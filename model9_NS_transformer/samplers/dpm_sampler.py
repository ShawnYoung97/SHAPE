"""SAMPLING ONLY."""

import torch

from .dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver


class DPMSolverSampler(object):
    def __init__(self, model, device,parameterization,**kwargs):
        super().__init__()
        self.model = model
        self.device=torch.device(device)
        self.parameterization=parameterization
        self.inference_batch_size = int(getattr(getattr(model, 'args', None), 'inference_batch_size', 512))
        if self.inference_batch_size < 0:
            raise ValueError('inference_batch_size must be nonnegative')
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)
        self.register_buffer('alphas_cumprod', to_torch(self.model.alphas_cumprod))

    def register_buffer(self, name, attr):
        if isinstance(attr, torch.Tensor):
            attr = attr.to(self.device)
        setattr(self, name, attr)

    @staticmethod
    def _conditioning_batch_size(conditioning):
        if isinstance(conditioning, dict):
            first = next(value for value in conditioning.values() if value is not None)
            return DPMSolverSampler._conditioning_batch_size(first)
        if isinstance(conditioning, (tuple, list)):
            return DPMSolverSampler._conditioning_batch_size(conditioning[0])
        return conditioning.shape[0]

    @staticmethod
    def _slice_condition(conditioning, start, stop):
        if conditioning is None:
            return None
        if isinstance(conditioning, dict):
            return {key: DPMSolverSampler._slice_condition(value, start, stop)
                    for key, value in conditioning.items()}
        if isinstance(conditioning, tuple):
            return tuple(DPMSolverSampler._slice_condition(value, start, stop)
                         for value in conditioning)
        if isinstance(conditioning, list):
            return [DPMSolverSampler._slice_condition(value, start, stop)
                    for value in conditioning]
        return conditioning[start:stop]

    @torch.no_grad()
    def _model_forward(self, x, t, conditioning=None):
        """Bound denoiser activations while retaining the original solver batch.

        The full initial noise, sample order, solver steps, and metric batch
        stay unchanged. Only independent denoiser evaluations are chunked.
        If another process consumes memory, halve the chunk and retry that
        part without redrawing noise or losing already evaluated trajectories.
        """
        if self.inference_batch_size == 0:
            return self.model.forward(x, t, conditioning)
        if t.numel() == 1:
            t = t.reshape(1).expand(x.shape[0])
        output = torch.empty_like(x)
        start = 0
        while start < x.shape[0]:
            size = min(self.inference_batch_size, x.shape[0] - start)
            stop = start + size
            chunk_condition = self._slice_condition(conditioning, start, stop)
            retry = False
            try:
                prediction = self.model.forward(x[start:stop], t[start:stop], chunk_condition)
            except torch.cuda.OutOfMemoryError:
                if size <= 1:
                    raise
                retry = True
            if retry:
                # Leave the exception handler first to release traceback-held
                # activation tensors before clearing the allocator cache.
                self.inference_batch_size = max(1, size // 2)
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                print(f'[Sampling] OOM: reducing denoiser trajectories per call '
                      f'to {self.inference_batch_size}; metric batch is unchanged')
                continue
            output[start:stop].copy_(prediction)
            del prediction
            start = stop
        return output

    @torch.no_grad()
    def sample(self,
               S,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        size = shape
        batch_size=size[0]
        if conditioning is not None:
            cbs = self._conditioning_batch_size(conditioning)
            if cbs != batch_size:
                print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

        # sampling
        # B,F, L = shape


        # print(f'Data shape for DPM-Solver sampling is {size}, sampling steps {S}')

        if x_T is None:
            img = torch.randn(size, device=self.device)
        else:
            img = x_T

        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)

        model_fn = model_wrapper(
            self._model_forward,
            ns,
            model_type=self.parameterization,
            guidance_type="classifier-free",
            condition=conditioning,
            unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )
        if self.parameterization=="noise":
            predict_x0=False
        elif self.parameterization=="x_start":
            predict_x0=True
        dpm_solver = DPM_Solver(model_fn, ns, predict_x0=predict_x0, thresholding=False)
        x = dpm_solver.sample(img, steps=S, skip_type="time_uniform", method="multistep", order=2, lower_order_final=True)
        # print('1',x.shape)
        return x



        

import torch
from model9_NS_transformer.condition_models import TimeFilter

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.cond_model_dict = {
            'TimeFilter': TimeFilter,
        }
        self.device = self._acquire_device()
        model,cond_pred_model= self._build_model()
        self.model = model.to(self.device)
        self.cond_pred_model=cond_pred_model.to(self.device)


    def _build_model(self):
        raise NotImplementedError
        return None, None, None

    def _acquire_device(self):
        # CUDA_VISIBLE_DEVICES is set by the launch script before CUDA starts.
        # Rewriting it here can change the meaning of an already selected GPU.
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
        print(f'Use device: {device}')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

"""D3U diffusion training, full SHAPE post-training, and paper evaluation."""

import copy
from contextlib import contextmanager
import gc
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import optim

from data_provider.data_factory import data_provider
from model9_NS_transformer.diffusion_models import diffuMTS
from model9_NS_transformer.exp.exp_basic import Exp_Basic
from model9_NS_transformer.losses.shape_loss import shape_preference_loss
from model9_NS_transformer.samplers.dpm_sampler import DPMSolverSampler
from utils.metrics import (metric, calc_quantile_CRPS_sum, crps_empirical,
                           interval_width_empirical, qice_empirical)
from utils.tools import EarlyStopping


class Exp_Main(Exp_Basic):
    def _build_model(self):
        model = diffuMTS.Model(self.args).float()
        conditioner = self.cond_model_dict[self.args.model].Model(self.args).float()
        path = (Path(self.args.pretrain_checkpoints) / self.args.model / 'all'
                / self.args.data_name / str(self.args.pred_len) / 'checkpoint.pth')
        conditioner.load_state_dict(torch.load(path, map_location=self.device), strict=True)
        conditioner.requires_grad_(False)
        return model, conditioner

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _load_model_state(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device), strict=True)

    def _sample_timesteps(self, count, preference=False):
        # Keep the original RNG streams: anchor timesteps on CPU, preference
        # timesteps on the model device. Both use symmetric timestep pairs.
        device = self.device if preference else torch.device('cpu')
        t = torch.randint(0, self.model.num_timesteps, (count // 2 + 1,), device=device)
        return torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:count].to(self.device)

    def _condition(self, batch_x, batch_y):
        decoder = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
        decoder = torch.cat([batch_y[:, :self.args.label_len, :], decoder], dim=1)
        with torch.no_grad():
            forecast, _, enc_out = self.cond_pred_model(batch_x, None, decoder, None)
        return batch_y[:, -self.args.pred_len:, :], forecast, enc_out

    def _diffusion_loss(self, truth, forecast, enc_out, t):
        residual = truth - forecast
        noise = torch.randn_like(residual)
        noisy = self.model.q_sample(residual, t, noise=noise)
        output = self.model(noisy, t, enc_out)
        target = residual if self.args.parameterization == 'x_start' else noise
        return (target - output).square().mean()

    def _preference_loss(self, batch_x, truth, forecast, enc_out, t, reference):
        # The old full-batch validity routing materialized contiguous tensors.
        # Preserve that layout while removing the unused routing branches.
        return shape_preference_loss(
            batch_x.contiguous(), truth.contiguous(), forecast.contiguous(),
            self.model, reference, enc_out.contiguous(), t,
            beta=self.args.beta_cp, parameterization=self.args.parameterization,
            projection_step=self.args.projection_step, n_patches=self.args.loss_patches,
            tail_len=self.args.ns_tail_len,
            loc_coefficients=self.args.shape_loc_coefficients,
            scale_coefficients=self.args.shape_scale_coefficients,
            eps=self.args.shape_epsilon,
        )

    def _stage2_batch(self, batch_x, batch_y, reference):
        anchor_t = self._sample_timesteps(batch_x.size(0))
        truth, forecast, enc_out = self._condition(batch_x, batch_y)
        diffusion = self._diffusion_loss(truth, forecast, enc_out, anchor_t)
        preference_t = self._sample_timesteps(batch_x.size(0), preference=True)
        preference, diagnostics = self._preference_loss(
            batch_x, truth, forecast, enc_out, preference_t, reference)
        return diffusion + self.args.lambda_dpo * preference, diffusion, preference, diagnostics

    @contextmanager
    def _validation_rng(self):
        cpu_state = torch.get_rng_state()
        gpu_state = torch.cuda.get_rng_state(self.device) if self.args.use_gpu else None
        torch.manual_seed(0)
        if self.args.use_gpu:
            torch.cuda.manual_seed(0)
        try:
            yield
        finally:
            torch.set_rng_state(cpu_state)
            if gpu_state is not None:
                torch.cuda.set_rng_state(gpu_state, self.device)

    def vali(self, vali_data, vali_loader, criterion=None):
        self.model.eval()
        self.cond_pred_model.eval()
        losses, counts = [], []
        with self._validation_rng(), torch.no_grad():
            for batch_x, batch_y, _, _ in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                t = self._sample_timesteps(batch_x.size(0))
                truth, forecast, enc_out = self._condition(batch_x, batch_y)
                loss = self._diffusion_loss(truth, forecast, enc_out, t)
                losses.append(loss.item())
                counts.append(batch_x.size(0))
        self.model.train()
        return float(np.average(losses, weights=counts))

    def _validate_preference(self, loader, reference):
        self.model.eval()
        self.cond_pred_model.eval()
        losses = []
        with self._validation_rng(), torch.no_grad():
            for index, (batch_x, batch_y, _, _) in enumerate(loader):
                if index >= self.args.stage2_vali_cp_batches:
                    break
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                truth, forecast, enc_out = self._condition(batch_x, batch_y)
                t = self._sample_timesteps(batch_x.size(0), preference=True)
                loss, _ = self._preference_loss(batch_x, truth, forecast, enc_out, t, reference)
                losses.append(loss.item())
        self.model.train()
        return float(np.mean(losses)) if losses else float('nan')

    def train(self, setting):
        _, train_loader = self._get_data('train')
        val_data, val_loader = self._get_data('val')
        test_data, test_loader = self._get_data('test')
        path = Path(self.args.checkpoints) / setting
        path.mkdir(parents=True, exist_ok=True)
        # The frozen conditioner is kept in its original Stage 1 training mode;
        # its dropout behavior therefore matches the supplied training pipeline.
        optimizer = optim.Adam([{'params': self.model.parameters()},
                                {'params': self.cond_pred_model.parameters()}],
                               lr=self.args.learning_rate)
        stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        history = {'Train': [], 'Val': [], 'Test': []}
        for epoch in range(self.args.train_epochs):
            self.model.train()
            self.cond_pred_model.train()
            losses = []
            for batch_x, batch_y, _, _ in train_loader:
                optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                t = self._sample_timesteps(batch_x.size(0))
                truth, forecast, enc_out = self._condition(batch_x, batch_y)
                loss = self._diffusion_loss(truth, forecast, enc_out, t)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            train_loss = float(np.mean(losses))
            val_loss = self.vali(val_data, val_loader)
            test_loss = self.vali(test_data, test_loader)
            print(f'Stage1 Epoch {epoch + 1}/{self.args.train_epochs} | '
                  f'Train={train_loss:.6f} Val={val_loss:.6f} Test={test_loss:.6f}')
            for label, value in zip(history, (train_loss, val_loss, test_loss)):
                history[label].append(value)
            stopping(val_loss, self.model, str(path))
            if stopping.early_stop or not np.isfinite(train_loss):
                break
        self._plot_training_curve(history, path / 'training_curve_s1.png', 'Stage 1 Training')
        self._load_model_state(path / 'checkpoint.pth')
        return self.model

    def train_stage2(self, setting):
        """Train the complete SHAPE objective for a fixed number of epochs."""
        _, train_loader = self._get_data('train')
        _, val_loader = self._get_data('val')
        source_setting = self.args.stage1_setting or setting
        source = Path(source_setting)
        if source_setting == source.name:
            source = Path(self.args.checkpoints) / source
        self._load_model_state(source / 'checkpoint.pth')
        self.cond_pred_model.requires_grad_(False)
        self.cond_pred_model.eval()
        reference = copy.deepcopy(self.model).eval().requires_grad_(False)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.stage2_lr)
        suffix = f'_{self.args.run_tag}' if self.args.run_tag else ''
        stage2_setting = f'{setting}_s2_rc0p0{suffix}'
        path = Path(self.args.checkpoints) / stage2_setting
        path.mkdir(parents=True, exist_ok=True)
        history = {'Train': [], 'Val_CP': []}
        print(f'[Stage2] SHAPE: L_diff + {self.args.lambda_dpo} * L_DPO; '
              f'epochs={self.args.stage2_epochs}, beta={self.args.beta_cp}')
        for epoch in range(self.args.stage2_epochs):
            self.model.train()
            self.cond_pred_model.eval()
            totals, anchors, preferences, valid_patches = [], [], [], []
            for batch_x, batch_y, _, _ in train_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                optimizer.zero_grad()
                loss, anchor, preference, diagnostics = self._stage2_batch(batch_x, batch_y, reference)
                totals.append(loss.item())
                anchors.append(anchor.item())
                preferences.append(preference.item())
                valid_patches.append(diagnostics['valid_patch_fraction'])
                loss.backward()
                optimizer.step()
            val_loss = self._validate_preference(val_loader, reference)
            train_loss = float(np.mean(totals))
            history['Train'].append(train_loss)
            history['Val_CP'].append(val_loss)
            print(f'Stage2 Epoch {epoch + 1}/{self.args.stage2_epochs} | '
                  f'Train={train_loss:.6f} L_diff={np.mean(anchors):.6f} '
                  f'L_DPO={np.mean(preferences):.6f} Val_CP={val_loss:.6f} '
                  f'valid_patch_fraction={np.mean(valid_patches):.3f}')
            torch.save(self.model.state_dict(), path / f'checkpoint_ep{epoch + 1}.pth')
        torch.save(self.model.state_dict(), path / 'checkpoint.pth')
        self._plot_training_curve(history, path / 'training_curve_s2.png', 'Stage 2 Training')
        print(f'[Stage2] Saved final epoch {self.args.stage2_epochs}: {path / "checkpoint.pth"}')
        return stage2_setting

    @staticmethod
    def _plot_training_curve(history, save_path, title='Training Curve'):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(8, 4))
        for label, values in history.items():
            axis.plot(range(1, len(values) + 1), values, label=label)
        axis.set(xlabel='Epoch', ylabel='Loss', title=title)
        axis.legend()
        figure.tight_layout()
        figure.savefig(save_path, dpi=120)
        plt.close(figure)

    def test(self, setting, test=0):
        """Evaluate the full chronological split with the paper batch-mean QICE."""
        test_data, test_loader = self._get_data('test')
        if test:
            self._load_model_state(Path(self.args.checkpoints) / setting / 'checkpoint.pth')
        preds, trues = [], []
        folder_path = str(Path(self.args.checkpoints).parent / 'test_results' / setting) + os.sep
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        # ── Per-group metrics — CSV preferred, online rho fallback ───────────
        # test loader uses shuffle=False, drop_last=False → indices are sequential
        _DEFAULT_GROUPS = ['low_ns', 'mid_ns', 'high_ns', 'severe_ns']
        _REGIME_GROUPS = ['in_domain', 'mild_shift', 'severe_shift']
        _ALL_GROUPS = list(_DEFAULT_GROUPS)
        ns_test_labels = None
        _ns_csv = os.path.join(getattr(self.args, 'ns_scores_dir', './ns_scores/'), 'test_ns.csv')
        if os.path.exists(_ns_csv):
            _ns_df = pd.read_csv(_ns_csv)
            ns_test_labels = _ns_df['ns_group'].values
            _expected_ns_len = len(test_data)
            if len(ns_test_labels) != _expected_ns_len:
                print(f"[Test][WARN] test_ns.csv length mismatch: "
                      f"csv={len(ns_test_labels)} vs test_data={_expected_ns_len}. "
                      f"Ignoring {_ns_csv}; regenerate ns_scores for pred_len={self.args.pred_len}.")
                ns_test_labels = None
            else:
                _seen_groups = set(map(str, np.unique(ns_test_labels)))
                if _seen_groups & set(_REGIME_GROUPS):
                    _ALL_GROUPS = [g for g in _REGIME_GROUPS if g in _seen_groups]
                else:
                    _ALL_GROUPS = [g for g in _DEFAULT_GROUPS if g in _seen_groups]
                _ALL_GROUPS += sorted(_seen_groups - set(_ALL_GROUPS))
                print(f"[Test] NS labels loaded ({len(ns_test_labels)} samples) from CSV; groups={_ALL_GROUPS}")
        if ns_test_labels is None:
            # Fast pre-pass: compute rho per sample without running the diffusion model.
            # test_loader is sequential so rho index aligns with sampling order below.
            print("[Test] test_ns.csv not found — computing NS groups online (fast pre-pass)")
            # Equation (26) is a window-level score: use all 96 historical
            # steps and the full future for Equation (26). SHAPE's
            # 48-step training tail and patch-level weights are separate.
            _EPS_NS = 1e-6
            _rho_list = []
            with torch.no_grad():
                for _bx, _by, _, _ in test_loader:
                    _x = _bx.double()
                    _y = _by[:, -self.args.pred_len:, :].double()
                    _mu_x = _x.mean(dim=1)
                    _sigma_x = _x.std(dim=1, unbiased=False)
                    _mu_y = _y.mean(dim=1)
                    _sigma_y = _y.std(dim=1, unbiased=False)
                    _rho = (((_mu_y - _mu_x).abs() / (_sigma_x + _EPS_NS))
                            + ((_sigma_y + _EPS_NS) / (_sigma_x + _EPS_NS)).log().abs()).mean(dim=1)
                    _rho_list.extend(_rho.numpy().tolist())
            _rho_arr = np.asarray(_rho_list)
            _threshold_csv = os.path.join(getattr(self.args, 'ns_scores_dir', './ns_scores/'),
                                          'regime_thresholds.csv')
            if os.path.exists(_threshold_csv):
                _threshold = pd.read_csv(_threshold_csv).iloc[0]
                if (str(_threshold['dataset']) not in {str(self.args.data), str(self.args.data_name)}
                        or not np.isclose(float(_threshold['gamma']), 1.0)):
                    raise ValueError('NS thresholds do not match this dataset/Equation (26); regenerate NS scores')
                ns_test_labels = np.where(
                    _rho_arr > float(_threshold['tau_severe']), 'severe_shift',
                    np.where(_rho_arr > float(_threshold['tau_mild']), 'mild_shift', 'in_domain'))
                _ALL_GROUPS = list(_REGIME_GROUPS)
                print('[Test] Using training-calibrated thresholds from', _threshold_csv)
            else:
                _pct = pd.Series(_rho_arr).rank(pct=True).values
                ns_test_labels = np.where(_pct >= 0.9, 'severe_ns',
                                 np.where(_pct >= 0.7, 'high_ns',
                                 np.where(_pct >= 0.3, 'mid_ns', 'low_ns')))
                print('[Test] No training thresholds: using within-test percentile groups')
            _cnts = {g: int((ns_test_labels == g).sum()) for g in _ALL_GROUPS}
            print(f"[Test] Online NS groups: {_cnts}")

        # Match the paper's original evaluation: compute ten-bin QICE inside
        # each chronological batch, then weight by that batch's window count.
        print(f'[Test] QICE protocol: batch_mean, bins=10, test_batch_size={self.args.test_batch_size}')
        g_stats = {g: dict(crps=0.0, qice=0.0, iw=0.0, mse=0.0, mae=0.0, n=0)
                   for g in _ALL_GROUPS}
        test_sample_offset = 0

        self.model.eval()
        self.cond_pred_model.eval()
        sampler = DPMSolverSampler(self.model, self.device, self.args.parameterization)
        print(f'[Test] DPM denoiser trajectories per call: {sampler.inference_batch_size}')
        total_mse = total_mae = total_samples = 0.0
        sum_crps = sum_qice = sum_iw = 0.0
        batch_start_time = time.time()
        with torch.no_grad():
            for index, (batch_x, batch_y, _, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                truth, forecast, enc_out = self._condition(batch_x, batch_y)
                count = int(self.model.diffusion_config.testing.n_z_samples /
                            self.model.diffusion_config.testing.n_z_samples_depart)
                chunks = []
                for _ in range(self.model.diffusion_config.testing.n_z_samples_depart):
                    # Preserve the original window-major trajectory order.
                    enc_tile = enc_out.repeat(count, 1, 1, 1, 1).transpose(0, 1).flatten(0, 1)
                    mean_tile = forecast.repeat(count, 1, 1, 1).transpose(0, 1).flatten(0, 1)
                    samples = sampler.sample(
                        S=self.args.DPMsolver_step, conditioning=enc_tile,
                        shape=(batch_x.size(0) * count, self.args.pred_len, self.args.c_out),
                        verbose=False, unconditional_guidance_scale=1.0,
                        unconditional_conditioning=None, eta=0., x_T=None)
                    samples = samples + mean_tile
                    chunks.append(samples.reshape(-1, count, self.args.pred_len, self.args.c_out).cpu().numpy())
                pred = np.concatenate(chunks, axis=1)
                true = truth.cpu().numpy()
                windows = pred.shape[0]
                sum_crps += crps_empirical(pred, true) * windows
                sum_qice += qice_empirical(pred, true) * windows
                sum_iw += interval_width_empirical(pred) * windows
                mae, mse, _, _, _ = metric(pred.mean(axis=1), true)
                total_mse += mse * windows
                total_mae += mae * windows
                total_samples += windows
                preds.append(pred.sum(-1))
                trues.append(true.sum(-1))
                labels = ns_test_labels[test_sample_offset:test_sample_offset + windows]
                for group in _ALL_GROUPS:
                    mask = labels == group
                    count_group = int(mask.sum())
                    if count_group == 0:
                        continue
                    group_pred, group_true = pred[mask], true[mask]
                    stats = g_stats[group]
                    stats['crps'] += crps_empirical(group_pred, group_true) * count_group
                    stats['qice'] += qice_empirical(group_pred, group_true) * count_group
                    stats['iw'] += interval_width_empirical(group_pred) * count_group
                    group_mae, group_mse, _, _, _ = metric(group_pred.mean(axis=1), group_true)
                    stats['mse'] += group_mse * count_group
                    stats['mae'] += group_mae * count_group
                    stats['n'] += count_group
                test_sample_offset += windows
                gc.collect()
                if index:
                    print(f'Testing: {index}/{len(test_loader)} cost time: '
                          f'{(time.time() - batch_start_time) / 60:.6f} min')
                    batch_start_time = time.time()

        print('total_samples',total_samples)
        avg_crps = sum_crps / total_samples
        avg_qice = sum_qice / total_samples
        avg_iw   = sum_iw   / total_samples
        mse_total = total_mse / total_samples
        mae_total = total_mae / total_samples
        print('NT metrc: CRPS:{:.4f}'.format(avg_crps))
        print('NT metrc: QICE:{:.4f}'.format(avg_qice))
        print('NT metrc: IW90:{:.4f}'.format(avg_iw))
        print('NT metrc: mse:{:.4f}, mae:{:.4f} '.format(mse_total, mae_total))
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds_save = np.array(preds)
        trues_save = np.array(trues)
        # Channels have already been summed above; retain a singleton channel
        # axis so calc_quantile_CRPS_sum does not also sum the time dimension.
        crps_sum=calc_quantile_CRPS_sum(preds[..., None],trues[..., None])
        print('NT metrc: CRPS_sum:{:.4f}'.format(crps_sum))


        np.save(folder_path + 'pred.npy', preds_save)
        np.save(folder_path + 'true.npy', trues_save)

        # ── Per-NS-group 结果 ─────────────────────────────────────────────────
        if ns_test_labels is not None:
            _hdr = f"{'Group':<14}  {'n':>6}  {'CRPS':>8}  {'QICE':>8}  {'IW90':>8}  {'MSE':>8}  {'MAE':>8}"
            print('\n── Per-NS-group metrics ─────────────────────────────────────────────')
            print(_hdr)
            print('─' * len(_hdr))
            for g in _ALL_GROUPS:
                s = g_stats[g]
                if s['n'] == 0:
                    continue
                print(f"{g:<14}  {s['n']:>6}  "
                      f"{s['crps']/s['n']:>8.4f}  {s['qice']/s['n']:>8.4f}  "
                      f"{s['iw']/s['n']:>8.4f}  {s['mse']/s['n']:>8.4f}  {s['mae']/s['n']:>8.4f}")
            # Combined rows for the two supported grouping schemes.
            _combine_specs = []
            if {'high_ns', 'severe_ns'}.issubset(g_stats):
                _combine_specs.append(('high+severe', ['high_ns', 'severe_ns']))
            if {'mild_shift', 'severe_shift'}.issubset(g_stats):
                _combine_specs.append(('shifted', ['mild_shift', 'severe_shift']))
            for _name, _members in _combine_specs:
                _h = {k: sum(g_stats[g][k] for g in _members)
                      for k in ('crps', 'qice', 'iw', 'mse', 'mae', 'n')}
                if _h['n'] > 0:
                    print('─' * len(_hdr))
                    print(f"{_name:<14}  {_h['n']:>6}  "
                          f"{_h['crps']/_h['n']:>8.4f}  {_h['qice']/_h['n']:>8.4f}  "
                          f"{_h['iw']/_h['n']:>8.4f}  {_h['mse']/_h['n']:>8.4f}  {_h['mae']/_h['n']:>8.4f}")
            print()
       

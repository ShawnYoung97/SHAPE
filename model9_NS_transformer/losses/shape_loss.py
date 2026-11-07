"""SHAPE preference construction and loss, Equations (11)--(24).

Location/scale transformations and validity checks use forecast space. The
denoiser operates on residuals around the frozen conditioning forecast.
"""

import math

import torch
import torch.nn.functional as F


def _validate_coefficients(coefficients, name):
    if len(coefficients) != 2:
        raise ValueError(f'{name} must contain an under- and over-correction coefficient')
    under, over = (float(value) for value in coefficients)
    if not (math.isfinite(under) and math.isfinite(over) and 0 <= under < 1 < over):
        raise ValueError(f'{name} must satisfy 0 <= c_minus < 1 < c_plus')
    return under, over


def build_shift_candidates(batch_x, y_true, tail_len=48,
                           loc_coefficients=(0.5, 1.5),
                           scale_coefficients=(0.5, 1.5), eps=1e-6):
    """Return the four forecast-space candidates [B, 4, H, C]."""
    if not 0 < tail_len <= batch_x.shape[1]:
        raise ValueError(f'tail_len must be in [1, {batch_x.shape[1]}], got {tail_len}')
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive')
    loc_coefficients = _validate_coefficients(loc_coefficients, 'loc_coefficients')
    scale_coefficients = _validate_coefficients(scale_coefficients, 'scale_coefficients')
    x_tail = batch_x[:, -tail_len:, :]
    mu_x = x_tail.mean(dim=1, keepdim=True)
    sigma_x = x_tail.std(dim=1, keepdim=True, unbiased=False)
    mu_y = y_true.mean(dim=1, keepdim=True)
    sigma_y = y_true.std(dim=1, keepdim=True, unbiased=False)
    candidates = []
    for coefficient in loc_coefficients:
        candidates.append(y_true + (coefficient - 1) * (mu_y - mu_x))
    ratio = (sigma_y + eps) / (sigma_x + eps)
    for coefficient in scale_coefficients:
        candidates.append(mu_y + ratio.pow(coefficient - 1) * (y_true - mu_y))
    return torch.stack(candidates, dim=1), ('loc', 'loc', 'scale', 'scale')


def patch_shift_weights(batch_x, y_true, tail_len=48, n_patches=4, eps=1e-6):
    """Equations (20)--(21), without additional score or weight clipping."""
    batch_size, horizon, channels = y_true.shape
    if n_patches <= 0 or horizon % n_patches:
        raise ValueError('n_patches must be positive and divide the forecast horizon')
    if not 0 < tail_len <= batch_x.shape[1]:
        raise ValueError('tail_len must select a nonempty part of the history')
    x_tail = batch_x[:, -tail_len:, :]
    mu_x = x_tail.mean(dim=1)[:, None, :]
    sigma_x = x_tail.std(dim=1, unbiased=False)[:, None, :]
    patches = y_true.reshape(batch_size, n_patches, horizon // n_patches, channels)
    mu_y = patches.mean(dim=2)
    sigma_y = patches.std(dim=2, unbiased=False)
    rho = ((mu_y - mu_x).abs() / (sigma_x + eps)
           + ((sigma_y + eps) / (sigma_x + eps)).log().abs()).mean(dim=-1)
    return (rho / (rho.mean(dim=1, keepdim=True) + eps)).detach()


@torch.no_grad()
def _project_residual(model, enc_out, residual, timestep, noise, parameterization):
    """Equation (15), using the existing deterministic short denoising budget.

    A timestep of 3 makes denoiser calls at 3, 2, and 1, as in the original
    script. No extra reverse-process noise is introduced between candidates.
    """
    steps = torch.full((residual.shape[0],), timestep, device=residual.device,
                       dtype=torch.long)
    noisy = model.q_sample(residual, steps, noise=noise)
    alphas = model.alphas_cumprod.to(device=residual.device, dtype=residual.dtype)
    for current in range(timestep, 0, -1):
        steps.fill_(current)
        prediction = model(noisy, steps, enc_out)
        alpha = alphas[current]
        if parameterization == 'x_start':
            clean = prediction
        else:
            clean = (noisy - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
        if current == 1:
            return clean
        predicted_noise = (noisy - alpha.sqrt() * clean) / (1 - alpha).sqrt()
        next_alpha = alphas[current - 1]
        noisy = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * predicted_noise


def select_preference_pair(winner, candidates, truth, n_patches=4):
    """Equations (17)--(19) and the strict patch validity rule of (24).

    Inputs are in a common space (forecasts, or residuals with the same offset).
    Invalid samples use a placeholder index but have every patch masked out.
    """
    batch_size, horizon, channels = truth.shape
    if n_patches <= 0 or horizon % n_patches:
        raise ValueError('n_patches must be positive and divide the forecast horizon')
    winner_mae = (winner - truth).abs().mean(dim=(1, 2))
    candidate_mae = (candidates - truth[:, None]).abs().mean(dim=(2, 3))
    gaps = candidate_mae - winner_mae[:, None]
    valid_candidates = gaps > 0
    valid_sequence = valid_candidates.any(dim=1)
    selected = gaps.masked_fill(~valid_candidates, float('inf')).argmin(dim=1)
    loser = candidates[torch.arange(batch_size, device=truth.device), selected]
    patch_shape = (batch_size, n_patches, horizon // n_patches, channels)
    winner_patch_mae = (winner - truth).abs().reshape(patch_shape).mean(dim=(2, 3))
    loser_patch_mae = (loser - truth).abs().reshape(patch_shape).mean(dim=(2, 3))
    valid_patch = valid_sequence[:, None] & (loser_patch_mae > winner_patch_mae)
    return loser, selected, valid_patch, winner_mae, candidate_mae


def weighted_preference_loss(margin, weights, valid_patch, beta=5.0, eps=1e-6):
    """Equation (24) per window, followed by the mean over valid windows."""
    valid_weights = weights * valid_patch
    weight_sum = valid_weights.sum(dim=1)
    valid_sample = valid_patch.any(dim=1) & (weight_sum > 0)
    per_sample = (valid_weights * -F.logsigmoid(beta * margin)).sum(dim=1) / (weight_sum + eps)
    return (per_sample * valid_sample).sum() / valid_sample.sum().clamp(min=1)


def shape_preference_loss(
    batch_x, y_true, y_T_mean, model, ref_model, enc_out, t,
    beta=5.0, parameterization='x_start', projection_step=3,
    n_patches=4, tail_len=48, loc_coefficients=(0.5, 1.5),
    scale_coefficients=(0.5, 1.5), eps=1e-6,
):
    """Equations (15)--(24), with the full four-candidate SHAPE objective.

    The caller adds the full-batch diffusion anchor. Pair construction and
    reference scores are detached; policy winner/loser scores carry gradients.
    """
    if parameterization not in ('x_start', 'noise'):
        raise ValueError(f'Unsupported parameterization: {parameterization!r}')
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('beta must be finite and positive')
    if y_true.shape != y_T_mean.shape:
        raise ValueError('The target and conditioning forecast must have identical shapes')
    if n_patches <= 0 or y_true.shape[1] % n_patches:
        raise ValueError('n_patches must be positive and divide pred_len')
    if not 1 <= projection_step < model.num_timesteps:
        raise ValueError('The projection timestep must satisfy 1 <= k* < timesteps')

    with torch.no_grad():
        candidates, families = build_shift_candidates(
            batch_x, y_true, tail_len, loc_coefficients, scale_coefficients, eps)
        truth_residual = y_true - y_T_mean
        candidate_residuals = candidates - y_T_mean[:, None]
        projection_noise = torch.randn_like(truth_residual)
        training_mode = model.training
        model.eval()
        try:
            winner = _project_residual(model, enc_out, truth_residual,
                                       projection_step, projection_noise, parameterization)
            projected_candidates = torch.stack([
                _project_residual(model, enc_out, seed, projection_step,
                                  projection_noise, parameterization)
                for seed in candidate_residuals.unbind(dim=1)
            ], dim=1)
        finally:
            model.train(training_mode)
        loser, selected, valid_patch, _, _ = select_preference_pair(
            winner, projected_candidates, truth_residual, n_patches)
        weights = patch_shift_weights(batch_x, y_true, tail_len, n_patches, eps)
        valid_weights = weights * valid_patch
        weight_sum = valid_weights.sum(dim=1)
        valid_sample = valid_patch.any(dim=1) & (weight_sum > 0)

        pair_noise = torch.randn_like(winner)
        winner_t = model.q_sample(winner, t, noise=pair_noise)
        loser_t = model.q_sample(loser, t, noise=pair_noise)
        winner_target = winner if parameterization == 'x_start' else pair_noise
        loser_target = loser if parameterization == 'x_start' else pair_noise
        winner_ref = ref_model(winner_t, t, enc_out)
        loser_ref = ref_model(loser_t, t, enc_out)

    winner_policy = model(winner_t, t, enc_out)
    loser_policy = model(loser_t, t, enc_out)
    batch_size, horizon, channels = y_true.shape
    patch_shape = (batch_size, n_patches, horizon // n_patches, channels)

    def patch_error(prediction, target):
        return (prediction - target).square().reshape(patch_shape).mean(dim=(2, 3))

    chosen_delta = patch_error(winner_policy, winner_target) - patch_error(winner_ref, winner_target)
    rejected_delta = patch_error(loser_policy, loser_target) - patch_error(loser_ref, loser_target)
    margin = rejected_delta - chosen_delta
    loss = weighted_preference_loss(margin, weights, valid_patch, beta, eps)
    selected_is_loc = torch.tensor([family == 'loc' for family in families],
                                   device=y_true.device)[selected]
    valid_count = valid_sample.sum().clamp(min=1)
    diagnostics = {
        'valid_sample': valid_sample.detach(),
        'valid_patch_fraction': valid_patch.float().mean().item(),
        'location_fraction': ((selected_is_loc & valid_sample).sum() / valid_count).item(),
    }
    return loss, diagnostics

import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

def quantile_loss(target, forecast, q: float) -> float:
    return 2 * np.sum(
        np.abs((forecast - target) * ((target <= forecast) * 1.0 - q))
    )

def calc_quantile_CRPS(all_preds, true): 

    target = true
    forecast = all_preds

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = np.sum(np.abs(target))
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(np.quantile(forecast[j : j + 1], quantiles[i], axis=1))
        q_pred = np.concatenate(q_pred, axis=0)
        q_loss = quantile_loss(target, q_pred, quantiles[i])
        CRPS += q_loss / denom
    return CRPS / len(quantiles)

def quantile_loss(target, forecast, q: float) -> float:
    return 2 * np.sum(
        np.abs((forecast - target) * ((target <= forecast) * 1.0 - q))
    )

def crps_empirical(pred, true):
    """
    Vectorized empirical CRPS via the energy score formula.

    Args:
        pred : np.ndarray [B, S, T, C]  — S probabilistic samples
        true : np.ndarray [B, T, C]     — ground truth

    Returns:
        float — mean CRPS over B, T, C
    """
    B, S, T, C = pred.shape
    # Term 1: E|X - y|
    term1 = np.abs(pred - true[:, np.newaxis]).mean(axis=1)          # [B, T, C]
    # Term 2: 0.5 * E|X - X'| via sorted-sample closed form
    sorted_pred = np.sort(pred, axis=1)                               # [B, S, T, C]
    idx     = np.arange(S, dtype=np.float32)                          # [S]
    weights = (2 * idx - S + 1) / (S ** 2)                           # [S]
    term2   = (sorted_pred * weights[np.newaxis, :, np.newaxis, np.newaxis]).sum(axis=1)  # [B, T, C]
    return float((term1 - term2).mean())

def interval_width_empirical(pred):
    """
    Mean 90% prediction interval width (95th - 5th percentile across samples).
    Lower is better given fixed coverage; compare alongside QICE.

    Args:
        pred : np.ndarray [B, S, T, C]

    Returns:
        float — mean interval width over B, T, C
    """
    q05 = np.quantile(pred, 0.05, axis=1)  # [B, T, C]
    q95 = np.quantile(pred, 0.95, axis=1)  # [B, T, C]
    return float((q95 - q05).mean())

# def qice_empirical(pred, true):
#     """
#     QICE: Quantile Interval Coverage Error.

#     For each quantile level q, measures |empirical_coverage - q|, then averages.
#     Lower is better (0 = perfectly calibrated).

#     Args:
#         pred : np.ndarray [B, S, T, C]
#         true : np.ndarray [B, T, C]

#     Returns:
#         float — mean QICE over quantile levels
#     """
#     quantiles = np.arange(0.05, 1.0, 0.05)
#     qice = 0.0
#     for q in quantiles:
#         q_hat = np.quantile(pred, q, axis=1)  # [B, T, C]
#         coverage = (true <= q_hat).mean()
#         qice += abs(coverage - q)
#     return qice / len(quantiles)

def qice_empirical(pred, true, n_bins=10):
    """
    QICE matching NsDiff implementation (histogram-style calibration).

    Evaluate one batch (or one group within a batch). The paper protocol
    averages these values using window counts; do not pool test-set histograms.

    Divides predicted samples into n_bins equal-probability bins (deciles by
    default) and measures how far the true values deviate from a uniform
    distribution across those bins.  Lower is better (0 = perfectly calibrated).

    Args:
        pred   : np.ndarray [B, S, T, C] — S probabilistic samples
        true   : np.ndarray [B, T, C]    — ground truth
        n_bins : int, number of quantile bins (default 10, matches NsDiff)

    Returns:
        float — mean |1/n_bins - actual_fraction| over bins
    """
    B, S, T, C = pred.shape
    # Flatten to (N, S) and (N,)
    preds_flat = pred.transpose(0, 2, 3, 1).reshape(-1, S)   # [B*T*C, S]
    targets_flat = true.reshape(-1)                            # [B*T*C]

    quantile_list = np.arange(n_bins + 1) * (100.0 / n_bins)  # [0, 10, ..., 100]
    y_pred_q = np.percentile(preds_flat, q=quantile_list, axis=1)  # [n_bins+1, N]

    # Count how many quantile boundaries each target exceeds (0 … n_bins+1)
    membership = ((targets_flat[np.newaxis, :] - y_pred_q) > 0).astype(int)  # [n_bins+1, N]
    y_true_bin = membership.sum(axis=0)  # [N], values in [0, n_bins+1]

    bin_counts = np.array([(y_true_bin == v).sum() for v in range(n_bins + 2)])
    # Absorb outliers into the edge bins
    bin_counts[1] += bin_counts[0]
    bin_counts[-2] += bin_counts[-1]
    bin_counts = bin_counts[1:-1]  # [n_bins]

    y_true_ratio = bin_counts / len(targets_flat)
    return float(np.abs(np.ones(n_bins) / n_bins - y_true_ratio).mean())

    
def calc_quantile_CRPS_sum(all_preds, true):

    target = true.sum(-1)
    forecast = all_preds

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = np.sum(np.abs(target))
    CRPS = 0
    for i in range(len(quantiles)):
        #targer(b,seq_len)
        #forecast(b,sampe,seq_len)
        q_pred = np.quantile(forecast.sum(-1),quantiles[i], axis=1)
      #   print(q_pred.shape)
      #   print(target.shape)
        q_loss = quantile_loss(target, q_pred, quantiles[i])
        CRPS += q_loss /denom
    return CRPS / len(quantiles)

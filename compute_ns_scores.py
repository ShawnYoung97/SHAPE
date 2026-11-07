"""
Compute per-sample non-stationarity scores for forecasting datasets.

Scores are computed on StandardScaler-normalized data (scaler fitted on train split only).

For each sample i:
  X_i : encoder input window  (seq_len  steps)
  Y_i : prediction target     (pred_len steps, no label overlap)

  rho_mu(X_i, Y_i)    = mean_d( |mu_Y^d - mu_X^d| / (sigma_X^d + eps) )
  rho_sigma(X_i, Y_i) = mean_d( |log( (sigma_Y^d + eps) / (sigma_X^d + eps) )| )
  rho(X_i, Y_i)       = rho_mu + gamma * rho_sigma

By default, ns_group is assigned by the percentile rank of rho within each split:
  >= 0.90  -> severe_ns
  >= 0.70  -> high_ns
  >= 0.30  -> mid_ns
  <  0.30  -> low_ns

Alternatively, --grouping train_calibrated keeps the same per-sample rho
definition, but assigns regimes using thresholds estimated from train rho:
  rho <= Q_train(mild_q)                     -> in_domain
  Q_train(mild_q) < rho <= Q_train(severe_q) -> mild_shift
  rho > Q_train(severe_q)                    -> severe_shift

--data controls the split borders:
  ETTm1 / ETTm2  : 15-min resolution (×4 steps per hour)
  ETTh1 / ETTh2  : hourly  resolution
  Solar           : headerless numeric matrix, using Dataset_Solar split borders
  others          : custom split, 70% train / 10% val / 20% test

Usage:
  python compute_ns_scores.py --data ETTm1 --pred_len 192
  python compute_ns_scores.py --data ETTh1 --pred_len 192 --data_path ETTh1.csv --output_dir ./ns_scores_etth1/
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
from numpy.lib.stride_tricks import as_strided


def make_sliding_windows(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Return a read-only strided view of shape (n_windows, window_size, D).
    Window i covers rows [i, i+window_size) of data.
    No data is copied.
    """
    n, D = data.shape
    n_windows = n - window_size + 1
    s0, s1 = data.strides
    return as_strided(data, shape=(n_windows, window_size, D), strides=(s0, s0, s1))


def assign_ns_group(percentiles: np.ndarray) -> list:
    groups = []
    for p in percentiles:
        if p >= 0.90:
            groups.append("severe_ns")
        elif p >= 0.70:
            groups.append("high_ns")
        elif p >= 0.30:
            groups.append("mid_ns")
        else:
            groups.append("low_ns")
    return groups


def assign_regime_group(rho: np.ndarray, tau_mild: float, tau_severe: float) -> list:
    groups = []
    for r in rho:
        if r > tau_severe:
            groups.append("severe_shift")
        elif r > tau_mild:
            groups.append("mild_shift")
        else:
            groups.append("in_domain")
    return groups


def main(args):
    # ── Load raw data ────────────────────────────────────────────────────────
    source_path = os.path.join(args.root_path, args.data_path)
    is_solar = args.data.lower() == 'solar'
    if is_solar:
        # solar_AL.txt has neither a header nor a date column. Reading it as
        # an ordinary forecasting CSV silently drops its first row and variable.
        values = pd.read_csv(source_path, header=None).to_numpy(dtype=np.float64)
    else:
        df_raw = pd.read_csv(source_path)
        values = df_raw.iloc[:, 1:].to_numpy(dtype=np.float64)

    # ── Split borders ─────────────────────────────────────────────────────────
    # Both ETTh and ETTm use the same 12/4/4-month ratio; only the step size differs.
    # border1 for val/test is shifted back by seq_len so the first sample's
    # input window starts at the exact split boundary.
    seq_len  = args.seq_len
    pred_len = args.pred_len

    if args.data in ("ETTh1", "ETTh2"):
        # Hourly resolution: 1 step = 1 hour
        # Train: 12×30×24 = 8 640  Val/Test: 4×30×24 = 2 880 each
        base = 12 * 30 * 24
        seg  = 4  * 30 * 24
        border1s = [0, base - seq_len, base + seg - seq_len]
        border2s = [base, base + seg, base + 2 * seg]
        split_desc = "ETT-hour fixed 12/4/4-month split"
    elif args.data in ("ETTm1", "ETTm2"):
        # 15-min resolution (ETTm1, ETTm2): 1 step = 15 min
        # Train: 12×30×24×4 = 34 560  Val/Test: 4×30×24×4 = 11 520 each
        base = 12 * 30 * 24 * 4
        seg  = 4  * 30 * 24 * 4
        border1s = [0, base - seq_len, base + seg - seq_len]
        border2s = [base, base + seg, base + 2 * seg]
        split_desc = "ETT-minute fixed 12/4/4-month split"
    elif is_solar:
        # Exactly match Dataset_Solar, including its separately rounded val size.
        n_raw = len(values)
        num_train = int(n_raw * 0.7)
        num_test = int(n_raw * 0.2)
        num_val = int(n_raw * 0.1)
        border1s = [0, num_train - seq_len, n_raw - num_test - seq_len]
        border2s = [num_train, num_train + num_val, n_raw]
        split_desc = 'Solar headerless matrix, Dataset_Solar 70/10/20 borders'
    else:
        # Match Dataset_Custom in data_provider/data_loader.py.
        # Traffic, Weather, Electricity, Exchange, etc. usually use this split.
        n_raw = len(values)
        num_train = int(n_raw * 0.7)
        num_test = int(n_raw * 0.2)
        num_val = n_raw - num_train - num_test
        border1s = [0, num_train - seq_len, n_raw - num_test - seq_len]
        border2s = [num_train, num_train + num_val, n_raw]
        split_desc = "custom 70/10/20 split"

    print(f"[split] {args.data}: {split_desc}")

    # ── Fit scaler on training data only ─────────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(values[border1s[0]:border2s[0]])
    data = scaler.transform(values)  # (T, D)

    os.makedirs(args.output_dir, exist_ok=True)
    eps   = args.epsilon
    gamma = args.gamma
    quantile_group_order = ["low_ns", "mid_ns", "high_ns", "severe_ns"]
    regime_group_order = ["in_domain", "mild_shift", "severe_shift"]
    split_rows = []
    count_rows = []

    for split_idx, split_name in enumerate(["train", "val", "test"]):
        split_data = data[border1s[split_idx]:border2s[split_idx]]  # (n_rows, D)
        n_samples  = len(split_data) - seq_len - pred_len + 1
        if n_samples <= 0:
            raise ValueError(
                f"{split_name} split is too short: rows={len(split_data)}, "
                f"seq_len={seq_len}, pred_len={pred_len}, n_samples={n_samples}. "
                f"Check --data / split strategy / dataset length."
            )

        # ── Sliding window views (zero-copy) ──────────────────────────────
        # X window i : split_data[i : i+seq_len]
        # Y window i : split_data[i+seq_len : i+seq_len+pred_len]
        X_wins = make_sliding_windows(split_data[:-pred_len], seq_len)   # (n_samples, seq_len,  D)
        Y_wins = make_sliding_windows(split_data[seq_len:],   pred_len)  # (n_samples, pred_len, D)

        # Bound std's temporary arrays for high-dimensional datasets such as
        # Solar. Chunking changes only memory use, not the score definition.
        chunk_size = getattr(args, 'chunk_size', 512)
        if chunk_size <= 0:
            raise ValueError('chunk_size must be positive')
        rho_mu = np.empty(n_samples, dtype=np.float64)
        rho_sigma = np.empty(n_samples, dtype=np.float64)
        for start in range(0, n_samples, chunk_size):
            stop = min(start + chunk_size, n_samples)
            mu_X = X_wins[start:stop].mean(axis=1)
            mu_Y = Y_wins[start:stop].mean(axis=1)
            sigma_X = X_wins[start:stop].std(axis=1)
            sigma_Y = Y_wins[start:stop].std(axis=1)
            rho_mu[start:stop] = np.mean(np.abs(mu_Y - mu_X) / (sigma_X + eps), axis=1)
            rho_sigma[start:stop] = np.mean(np.abs(np.log((sigma_Y + eps) / (sigma_X + eps))), axis=1)
        rho       = rho_mu + gamma * rho_sigma

        # ── Percentile rank within this split ─────────────────────────────
        # rankdata produces ranks 1..n; dividing by n gives (1/n .. 1)
        percentiles = rankdata(rho, method="average") / n_samples
        quantile_groups = assign_ns_group(percentiles)

        split_rows.append({
            "split_name": split_name,
            "n_samples": n_samples,
            "rho_mu": rho_mu,
            "rho_sigma": rho_sigma,
            "rho": rho,
            "percentiles": percentiles,
            "quantile_groups": quantile_groups,
        })

    train_rho = split_rows[0]["rho"]
    tau_mild = float(np.quantile(train_rho, args.mild_q))
    tau_severe = float(np.quantile(train_rho, args.severe_q))
    if not tau_mild < tau_severe:
        raise ValueError(
            f"Expected mild threshold < severe threshold, got "
            f"{tau_mild:.6f} >= {tau_severe:.6f}. Check --mild_q/--severe_q."
        )

    threshold_path = os.path.join(args.output_dir, "regime_thresholds.csv")
    pd.DataFrame([{
        "dataset": args.data,
        "gamma": gamma,
        "mild_q": args.mild_q,
        "severe_q": args.severe_q,
        "tau_mild": tau_mild,
        "tau_severe": tau_severe,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "n_channels": values.shape[1],
        "n_raw_rows": len(values),
        "source_file": os.path.basename(source_path),
        "input_format": 'solar_headerless' if is_solar else 'dated_csv',
    }]).to_csv(threshold_path, index=False)
    print(
        f"[regime thresholds] train-calibrated: "
        f"Q{args.mild_q:.2f}={tau_mild:.4f}, Q{args.severe_q:.2f}={tau_severe:.4f}"
    )

    for row in split_rows:
        split_name = row["split_name"]
        n_samples = row["n_samples"]
        rho_mu = row["rho_mu"]
        rho_sigma = row["rho_sigma"]
        rho = row["rho"]
        percentiles = row["percentiles"]
        quantile_groups = row["quantile_groups"]
        regime_groups = assign_regime_group(rho, tau_mild, tau_severe)

        if args.grouping == "percentile":
            ns_groups = quantile_groups
            group_order = quantile_group_order
        else:
            ns_groups = regime_groups
            group_order = regime_group_order

        df_out = pd.DataFrame({
            "sample_idx":     np.arange(n_samples, dtype=np.int64),
            "dataset":        args.data,
            "rho_mu":         rho_mu,
            "rho_sigma":      rho_sigma,
            "rho":            rho,
            "rho_percentile": percentiles,
            "ns_quantile_group": quantile_groups,
            "regime_group":   regime_groups,
            "ns_group":       ns_groups,
        })
        out_path = os.path.join(args.output_dir, f"{split_name}_ns.csv")
        df_out.to_csv(out_path, index=False)

        counts = df_out["ns_group"].value_counts().to_dict()
        ordered_counts = {g: int(counts.get(g, 0)) for g in group_order}
        count_rows.append({
            "dataset": args.data,
            "split": split_name,
            "n_samples": int(n_samples),
            **ordered_counts,
        })
        regime_counts = df_out["regime_group"].value_counts().to_dict()
        ordered_regime_counts = {g: int(regime_counts.get(g, 0)) for g in regime_group_order}
        print(
            f"[{split_name}] {n_samples} samples -> {out_path}\n"
            f"  rho : min={rho.min():.4f}  mean={rho.mean():.4f}  max={rho.max():.4f}\n"
            f"  ns_group({args.grouping}): {ordered_counts}\n"
            f"  regime_group: {ordered_regime_counts}\n"
        )

    count_path = os.path.join(args.output_dir, "ns_group_counts.csv")
    pd.DataFrame(count_rows).to_csv(count_path, index=False)
    print(f"[summary] group counts saved -> {count_path}")
    print(f"[summary] regime thresholds saved -> {threshold_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute window-level non-stationarity scores")
    parser.add_argument("--data",       type=str,   default="ETTm1",
                        help="ETTm1 | ETTm2 | ETTh1 | ETTh2 | Solar, or a custom dataset name (controls format/splits)")
    parser.add_argument("--root_path",  type=str,   default="./dataset/ETT-small/")
    parser.add_argument("--data_path",  type=str,   default="ETTm1.csv")
    parser.add_argument("--seq_len",    type=int,   default=96)
    parser.add_argument("--label_len",  type=int,   default=48)
    parser.add_argument("--pred_len",   type=int,   default=192)
    parser.add_argument("--epsilon",    type=float, default=1e-6)
    parser.add_argument("--gamma",      type=float, default=1.0)
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="maximum windows per statistics chunk (limits temporary memory)")
    parser.add_argument("--grouping",   type=str,   default="percentile",
                        choices=["percentile", "train_calibrated"],
                        help="which labels to write to ns_group; extra columns are always saved")
    parser.add_argument("--mild_q",     type=float, default=0.70,
                        help="train-rho quantile for in-domain vs mild shift")
    parser.add_argument("--severe_q",   type=float, default=0.90,
                        help="train-rho quantile for mild vs severe shift")
    parser.add_argument("--output_dir", type=str,   default="./ns_scores/")
    args = parser.parse_args()
    main(args)



'''
python compute_ns_scores.py --data ETTm1 --pred_len 192 --data_path ETTm1.csv --output_dir ./ns_scores/ETTm1/
'''

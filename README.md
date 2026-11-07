# SHAPE: Shift-Aware Preference Alignment for Probabilistic Time Series Forecasting

SHAPE aligns pretrained conditional diffusion forecasters using shift-aware preferences. It constructs location- and scale-shifted alternatives, projects them through the diffusion trajectory, and applies a reference-anchored, patch-wise preference objective alongside the denoising loss. Preference construction is used only during training.

This repository provides the main SHAPE implementation on **D³U**, with a runnable **ETTm1** example using 96 historical steps to forecast the next 192 steps.

## Installation

Use Python 3.11 and a CUDA-enabled PyTorch installation compatible with your GPU. Install the dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## Data and Checkpoints

Download `ETTm1.csv` from the [ETT dataset repository](https://github.com/zhouhaoyi/ETDataset) and place it at:

```text
dataset/ETT-small/ETTm1.csv
```

The release includes a pretrained point forecasting checkpoint in `pretrain_checkpoints/` and a shared Stage 1 diffusion checkpoint in `checkpoints/`. Keep their directory structure unchanged; the scripts load these checkpoints automatically.

## Usage

Run all commands from the repository root.

### Stage 2 Training and Evaluation

Train SHAPE for five alignment epochs from the supplied Stage 1 checkpoint, then evaluate on the ETTm1 test split:

```bash
GPU=0 SEEDS="2021" RUN_TAG_BASE=shape bash scripts/ettm1_s2.sh
```

To run the three seeds used in the paper sequentially, sharing the same Stage 1 checkpoint:

```bash
GPU=0 SEEDS="2021 2024 2027" RUN_TAG_BASE=shape bash scripts/ettm1_s2.sh
```

The paper reports results averaged over three seeds. The single-seed command runs one alignment and evaluation experiment.

### Evaluation Only

After Stage 2 training, evaluate the saved checkpoint using the same seed and run tag:

```bash
GPU=0 SEEDS="2021" RUN_TAG_BASE=shape RUN_MODE=test bash scripts/ettm1_s2.sh
```

Evaluation generates 100 forecast samples per window using DPM-Solver with 20 sampling steps. It reports CRPS, QICE, MSE, MAE, and interval width. QICE uses batch-mean aggregation; retain the default test batch size of 128 when comparing results.

Stage 2 checkpoints are saved under `checkpoints/`, logs under `logs/shape/`, and evaluation arrays under `test_results/`. The saved `pred.npy` and `true.npy` arrays contain sums over the variables.

### Optional: Train Stage 1

To train a new Stage 1 diffusion model using the supplied point forecasting checkpoint:

```bash
GPU=0 SEED=2021 bash scripts/ettm1_s1.sh
```

This command trains and evaluates Stage 1, saving its checkpoint under `outputs/stage1/`. To use it for alignment, set `STAGE1_SETTING` to the generated directory containing `checkpoint.pth` when launching `scripts/ettm1_s2.sh`. Otherwise, Stage 2 uses the bundled checkpoint.

### Optional: Precompute Shift Groups

For grouped test metrics, non-stationarity scores and percentile-based groups can be prepared with:

```bash
python compute_ns_scores.py \
    --data ETTm1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTm1.csv \
    --seq_len 96 \
    --pred_len 192 \
    --output_dir ./ns_scores_main/ETTm1
```

If these files are absent, evaluation computes the default test groups automatically.

## Contact

If you have any questions about this work or the code, please contact [Xiaoyang Li](mailto:xiaoyang.li01@adelaide.edu.au).

## Acknowledgement

This repository is built on [D³U](https://github.com/Torea-L/D3U). We thank the authors for their valuable code and efforts. Additional sources and third-party license texts are listed in [NOTICE.txt](NOTICE.txt) and [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES/).

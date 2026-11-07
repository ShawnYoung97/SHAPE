#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd "$repo_root"

GPU=${GPU:-0}
SEED=${SEED:-2021}
PYTHON_BIN=${PYTHON_BIN:-python}
checkpoint_root=${CHECKPOINT_ROOT:-./outputs/stage1}

seq_len=96
pred_len=192
model_name=TimeFilter

root_path_name=${ROOT_PATH:-./dataset/ETT-small}
data_path_name=${DATA_PATH:-ETTm1.csv}
model_id_name=ETTm1
data_name=ETTm1

pretrain_root=${PRETRAIN_ROOT:-./pretrain_checkpoints}

args=(
    --is_training
    --seed "$SEED"
    --root_path "$root_path_name"
    --data_path "$data_path_name"
    --model_id "${model_id_name}_${seq_len}_${pred_len}"
    --model "$model_name"
    --pretrain_checkpoints "$pretrain_root"
    --checkpoints "$checkpoint_root"
    --data "$data_name"
    --data_name "$model_id_name"
    --features M
    --seq_len "$seq_len"
    --label_len 48
    --pred_len "$pred_len"
    --enc_in 7
    --dec_in 7
    --c_out 7
    --e_layers_c 2
    --n_heads_c 4
    --d_model_c 256
    --d_ff 256
    --dropout 0.2
    --alpha 0.1
    --pos 1
    --patch_len 8
    --top_p 0.0
    --depth 1
    --d_model_d 128
    --num_workers "${NUM_WORKERS:-4}"
    --itr 1
    --train_epochs 100
    --patience 10
    --learning_rate 1e-4
    --batch_size 64
    --test_batch_size 128
    --timesteps 100
    --des Exp
    --denoise_model PatchDN
    --diffusion_config_dir ./model9_NS_transformer/configs/shape.yml
    --type_sampler DPM_solver
    --DPMsolver_step 20
    --gpu 0
    --parameterization x_start
    --inference_batch_size "${INFERENCE_BATCH_SIZE:-512}"
)

CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" -u ./runner9_NS_transformer.py "${args[@]}"

#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd "$repo_root"

GPU=${GPU:-0}
read -r -a SEEDS <<< "${SEEDS:-2021}"
RUN_TAG_BASE=${RUN_TAG_BASE:-shape}
RUN_MODE=${RUN_MODE:-stage2}
case "$RUN_MODE" in
    stage2|test) ;;
    *) printf 'RUN_MODE must be stage2 or test\n' >&2; exit 2 ;;
esac
PYTHON_BIN=${PYTHON_BIN:-python}
checkpoint_root=${CHECKPOINT_ROOT:-./checkpoints}
ns_dir=${NS_SCORES_DIR:-./ns_scores_main/ETTm1}
log_root=${LOG_ROOT:-./logs}

seq_len=96
pred_len=192
model_name=TimeFilter

root_path_name=${ROOT_PATH:-./dataset/ETT-small}
data_path_name=${DATA_PATH:-ETTm1.csv}
model_id_name=ETTm1
data_name=ETTm1

pretrain_root=${PRETRAIN_ROOT:-./pretrain_checkpoints}
base_setting=False_ts100_PatchDN_ETTm1_96_192_TimeFilter_ETTm1_ftM_sl96_ll48_pl192_dm256_nh4_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0
STAGE1_SETTING=${STAGE1_SETTING:-$base_setting}

for random_seed in "${SEEDS[@]}"; do
    run_tag="${RUN_TAG_BASE}_s${random_seed}"
    stage2_setting="${base_setting}_s2_rc0p0_${run_tag}"
    mode=$RUN_MODE
    log_file="$log_root/$RUN_TAG_BASE/ettm1_s2_${run_tag}.log"
    if [[ "$mode" == test ]]; then log_file="${log_file%.log}_test.log"; fi

    printf 'ETTm1 SHAPE | seed=%s | mode=%s | log=%s\n' "$random_seed" "$mode" "$log_file"

    args=(
        --seed "$random_seed"
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
        --timesteps 100
        --des Exp
        --denoise_model PatchDN
        --type_sampler DPM_solver
        --DPMsolver_step 20
        --gpu 0
        --parameterization x_start
        --stage1_setting "$STAGE1_SETTING"
        --beta_cp 5.0
        --stage2_lr 1e-6
        --stage2_epochs 5
        --batch_size 64
        --test_batch_size 128
        --inference_batch_size "${INFERENCE_BATCH_SIZE:-512}"
        --ns_tail_len 48
        --shape_loc_coefficients 0.5 1.5
        --shape_scale_coefficients 0.5 1.5
        --loss_patches 4
        --projection_step 3
        --lambda_dpo 0.1
        --ns_scores_dir "$ns_dir"
        --diffusion_config_dir ./model9_NS_transformer/configs/shape.yml
        --run_tag "$run_tag"
    )
    if [[ "$mode" == test ]]; then
        args+=(--test_setting "$stage2_setting")
    else
        args+=(--stage2_only)
    fi
    mkdir -p -- "$(dirname -- "$log_file")"
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" -u ./runner9_NS_transformer.py "${args[@]}" 2>&1 | tee "$log_file"
done

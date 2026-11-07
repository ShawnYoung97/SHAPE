"""Run D3U Stage 1, full SHAPE Stage 2, or checkpoint evaluation on ETTm1."""

import argparse
import random

import numpy as np
import setproctitle
import torch

from model9_NS_transformer.exp.exp_main import Exp_Main


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('true', '1', 'yes'):
        return True
    if value in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Expected true/false, yes/no, or 1/0')


def build_parser():
    parser = argparse.ArgumentParser(description='SHAPE on D3U: ETTm1 main experiment')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--is_training', action='store_true', help='train a Stage 1 diffusion model')
    mode.add_argument('--stage2_only', action='store_true', help='train full SHAPE from the shared Stage 1')
    parser.add_argument('--skip_test', action='store_true', help='skip evaluation after training')
    parser.add_argument('--test_setting', default='', help='checkpoint setting for evaluation only')
    parser.add_argument('--model_id', default='ETTm1_96_192')
    parser.add_argument('--model', choices=['TimeFilter'], default='TimeFilter', help='frozen D3U conditioning network')
    parser.add_argument('--data', choices=['ETTm1'], default='ETTm1')
    parser.add_argument('--data_name', default='ETTm1')
    parser.add_argument('--root_path', default='./dataset/ETT-small')
    parser.add_argument('--data_path', default='ETTm1.csv')
    parser.add_argument('--features', choices=['M'], default='M')
    parser.add_argument('--target', default='OT')
    parser.add_argument('--freq', default='h')
    parser.add_argument('--checkpoints', default='./checkpoints')
    parser.add_argument('--pretrain_checkpoints', default='./pretrain_checkpoints')
    parser.add_argument('--stage1_setting', default='', help='bare checkpoint name or project-relative Stage 1 directory')
    parser.add_argument('--diffusion_config_dir', default='./model9_NS_transformer/configs/shape.yml')
    parser.add_argument('--ns_scores_dir', default='./ns_scores_main/ETTm1')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=192)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model_c', type=int, default=256)
    parser.add_argument('--n_heads_c', type=int, default=4)
    parser.add_argument('--e_layers_c', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1)
    parser.add_argument('--patch_len', type=int, default=8)
    parser.add_argument('--denoise_model', choices=['PatchDN'], default='PatchDN')
    parser.add_argument('--d_model_d', type=int, default=128)
    parser.add_argument('--n_heads_d', type=int, default=8)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--padding_patch', choices=['end'], default='end')
    parser.add_argument('--cond_patch_num', type=int, default=None)
    parser.add_argument('--timesteps', type=int, default=100)
    parser.add_argument('--type_sampler', choices=['DPM_solver'], default='DPM_solver')
    parser.add_argument('--DPMsolver_step', type=int, default=20)
    parser.add_argument('--parameterization', choices=['x_start', 'noise'], default='x_start')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--inference_batch_size', type=int, default=512,
                        help='expanded trajectories per denoiser forward; 0 disables chunking')
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10, help='Stage 1 validation patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--stage2_lr', type=float, default=1e-6)
    parser.add_argument('--stage2_epochs', type=int, default=5)
    parser.add_argument('--stage2_vali_cp_batches', type=int, default=30, help='validation preference batches for logging only')
    parser.add_argument('--beta_cp', type=float, default=5.0)
    parser.add_argument('--lambda_dpo', type=float, default=0.1)
    parser.add_argument('--projection_step', type=int, default=3)
    parser.add_argument('--loss_patches', type=int, default=4)
    parser.add_argument('--ns_tail_len', type=int, default=48)
    parser.add_argument('--shape_loc_coefficients', type=float, nargs=2, default=[0.5, 1.5])
    parser.add_argument('--shape_scale_coefficients', type=float, nargs=2, default=[0.5, 1.5])
    parser.add_argument('--shape_epsilon', type=float, default=1e-6)
    parser.add_argument('--run_tag', default='', help='suffix for this Stage 2 run')
    parser.add_argument('--des', default='Exp')
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--embed', choices=['timeF'], default='timeF')
    parser.add_argument('--use_gpu', type=parse_bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2021)
    return parser


def make_setting(args, index=0):
    """Retain the bundled checkpoint names and existing Stage 2 run paths."""
    return (f'False_ts{args.timesteps}_{args.denoise_model}_{args.model_id}_{args.model}_'
            f'{args.data_name}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_'
            f'pl{args.pred_len}_dm{args.d_model_c}_nh{args.n_heads_c}_el{args.e_layers_c}_'
            f'dl1_df{args.d_ff}_fc3_eb{args.embed}_dtTrue_{args.des}_{index}')


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.beta_cp <= 0 or args.lambda_dpo <= 0:
        parser.error('Full SHAPE requires positive beta_cp and lambda_dpo')
    if args.stage2_epochs < 1 or args.stage2_vali_cp_batches < 1:
        parser.error('Stage 2 epoch and validation batch counts must be positive')
    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
    seed = np.random.randint(2147483647) if args.seed == -1 else args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.use_gpu:
        torch.cuda.set_device(args.gpu)
    setproctitle.setproctitle('SHAPE_D3U')
    print('Using seed:', seed)
    print(args)
    for index in range(args.itr):
        setting = make_setting(args, index)
        experiment = Exp_Main(args)
        if args.stage2_only:
            setting = experiment.train_stage2(setting)
        elif args.is_training:
            experiment.train(setting)
        else:
            setting = args.test_setting or setting
        if not args.skip_test:
            experiment.test(setting, test=1)
        if args.use_gpu:
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

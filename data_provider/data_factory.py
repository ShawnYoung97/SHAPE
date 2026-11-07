"""Chronological train/validation/test loaders for the ETTm1 main experiment."""

from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_ETT_minute


def data_provider(args, flag):
    if args.data != 'ETTm1' or flag not in ('train', 'val', 'test'):
        raise ValueError('This release provides the ETTm1 train/val/test example')
    dataset = Dataset_ETT_minute(
        root_path=args.root_path, data_path=args.data_path, flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features, target=args.target,
        timeenc=1, freq=args.freq)
    loader = DataLoader(
        dataset, batch_size=args.test_batch_size if flag == 'test' else args.batch_size,
        shuffle=flag == 'train', drop_last=flag == 'train', num_workers=args.num_workers)
    print(flag, len(dataset), 'windows;', len(loader), 'batches')
    return dataset, loader

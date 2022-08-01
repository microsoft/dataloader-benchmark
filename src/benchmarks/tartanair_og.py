import torch
from benchmarker import Benchmarker

from src.data.tartanair import build_loader
from src.data.tartanair.build import TartanAirNoTransform, TartanAirVideoDataset
from src.utils.opts import parse_args


def get_tartanair_dataset(args):
    train_dataset, _, train_loader, _, _ = build_loader(args)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    return train_dataset, train_loader


def get_tartanair_dataset_no_transform(args):
    transform = TartanAirNoTransform()
    dataset = TartanAirVideoDataset(
        ann_file=args.train_ann_file,
        clip_len=args.clip_len,
        seq_len=args.seq_len,
        modalities=args.modalities,
        transform=transform,
        video_name_keyword=None,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    return dataloader


def benchmark(args):
    _, train_dataloader = get_tartanair_dataset(args)

    benchmarker = Benchmarker()
    benchmarker.set_dataloader(train_dataloader)
    benchmarker.benchmark_tartanair(args)


def main(args):
    print("start")
    benchmark(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)

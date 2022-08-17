from time import perf_counter

from src.data.tartanair.build import TartanAirNoTransform, TartanAirVideoDataset, build_loader
from src.utils.opts import parse_args
from torch.utils.data import DataLoader

from benchmarker import Benchmarker


def get_dataloader(args):
    _, _, train_loader, _, _ = build_loader(args)
    return train_loader


def get_dataloader_no_transform(args):
    transform = TartanAirNoTransform()
    dataset = TartanAirVideoDataset(
        ann_file=args.train_ann_file,
        clip_len=args.clip_len,
        seq_len=args.seq_len,
        modalities=args.modalities,
        transform=transform,
        video_name_keyword=None,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    return dataloader


def benchmark(args):
    start = perf_counter()
    dataloader = get_dataloader(args)
    benchmarker = Benchmarker(verbose=args.verbose, library="pytorch", dataset="tartanair")
    benchmarker.set_dataloader(dataloader)
    benchmarker.benchmark_tartanair(args)
    end = perf_counter()
    print(f"{(start-end)/60.0=}")


def main(args):
    benchmark(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)

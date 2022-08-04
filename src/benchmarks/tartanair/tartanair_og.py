from tartanair_ops import get_tartanair_args

from src.benchmarks.benchmarker import Benchmarker
from src.benchmarks.common_opts import get_common_args
from src.data.tartanair import build_loader


def get_dataloader(args):
    _, _, train_loader, _, _ = build_loader(args)
    return train_loader


def benchmark(args):
    dataloader = get_dataloader(args)
    benchmarker = Benchmarker(verbose=args.verbose, library="pytorch", dataset="tartanair")
    benchmarker.set_dataloader(dataloader)
    benchmarker.benchmark_tartanair(args)


def main(args):
    benchmark(args)


if __name__ == "__main__":
    args = get_common_args()
    tartanair_args = get_tartanair_args()

    args.__dict__.update(tartanair_args.__dict__)

    main(args)

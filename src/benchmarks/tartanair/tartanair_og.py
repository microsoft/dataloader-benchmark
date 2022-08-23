import argparse
from timeit import default_timer as timer

from src.benchmarks.benchmarker import Benchmarker
from src.benchmarks.common_opts import add_common_args
from src.benchmarks.tartanair.tartanair_opts import add_tartanair_args
from src.data.tartanair import build_loader


def get_dataloader(args):
    _, _, train_loader, _, _ = build_loader(args)
    return train_loader


def benchmark(args):
    init_time = timer()
    dataloader = get_dataloader(args)
    benchmarker = Benchmarker(verbose=args.verbose, library="pytorch", dataset="tartanair", init_time=init_time)
    benchmarker.set_dataloader(dataloader)
    benchmarker.benchmark_tartanair(args)


def main(args):
    benchmark(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    common_group = parser.add_argument_group(title="common params")
    tartanair_group = parser.add_argument_group(title="tartanair params")

    add_common_args(common_group)
    add_tartanair_args(tartanair_group)

    args = parser.parse_args()

    main(args)

import argparse

from ffcv.fields.decoders import NDArrayDecoder
from ffcv.loader import Loader
from ffcv.transforms import ToTensor
from utils_ffcv import get_order_option

from src.benchmarks.benchmarker import Benchmarker
from src.benchmarks.climate.climate_opts import add_climate_args
from src.benchmarks.common_opts import add_common_args, add_ffcv_args


def get_dataloader(args):
    print(f"Dataset: ffcv\n \t {args.beton_file}")
    print(f"Order: {args.order}")
    print(f"OS cache: {args.os_cache}")

    if args.use == "forecast":
        PIPELINES = {
            "inputs": [NDArrayDecoder(), ToTensor()],
            "outputs": [NDArrayDecoder(), ToTensor()],
        }

    elif args.use == "pretrain":
        PIPELINES = {
            "trajs": [NDArrayDecoder(), ToTensor()],
        }

    dataloader = Loader(
        args.beton_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        order=get_order_option(args.order),
        os_cache=args.os_cache,
        pipelines=PIPELINES,
    )
    return dataloader


def benchmark(args):
    dataloader = get_dataloader(args)
    benchmarker = Benchmarker(verbose=args.verbose, dataset="climate", library="ffcv")
    benchmarker.set_dataloader(dataloader)
    benchmarker.benchmark_tartanair(args)


def main(args):
    benchmark(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    add_common_args(parser.add_argument_group("common args"))
    add_climate_args(parser.add_argument_group("climate args"))
    add_ffcv_args(parser.add_argument_group("ffcv args"))

    args = parser.parse_args()
    main(args)

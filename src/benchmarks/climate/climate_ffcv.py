import argparse
from distutils.util import strtobool

from climate_ops import get_climate_args
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.loader import Loader
from ffcv.transforms import ToTensor
from utils_ffcv import get_order_option

from src.benchmarks.benchmarker import Benchmarker
from src.benchmarks.common_opts import get_common_args


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
    benchmarker = Benchmarker(verbose=args.verbose, dataset=f"climate_{args.use}", library="ffcv")
    benchmarker.set_dataloader(dataloader)
    benchmarker.benchmark_tartanair(args)


def get_ffcv():
    parser = argparse.ArgumentParser(description="FFCV options")
    parser.add_argument("--beton_file", type=str, help="Dataset to use for benchmarking")
    parser.add_argument("--order", type=str, default="random", help="Ordering of data: random or quasi_random")
    parser.add_argument("--os_cache", type=lambda x: bool(strtobool(x)))
    args = parser.parse_args()

    return args


def main(args):
    benchmark(args)


if __name__ == "__main__":
    args = get_common_args()
    climate_args = get_climate_args()
    ffcv_args = get_ffcv()

    args.__dict__.update(climate_args.__dict__)
    args.__dict__.update(ffcv_args.__dict__)

    main(args)

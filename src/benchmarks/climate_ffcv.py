import argparse
import distutils

from benchmarker import Benchmarker
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.loader import Loader
from ffcv.transforms import ToTensor
from utils_ffcv import get_order_option


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

    benchmarker = Benchmarker()
    benchmarker.set_dataloader(dataloader)
    benchmarker.benchmark_tartanair(args)


def parse_args():
    parser = argparse.ArgumentParser(description="FFCV options")
    parser.add_argument("--benchmark_results_file", default="benchmark_results_climate.csv", type=str)
    parser.add_argument("--beton_file", type=str, help="Dataset to use for benchmarking")
    parser.add_argument("--order", type=str, default="random", help="Ordering of data: random or quasi_random")
    parser.add_argument("--os_cache", type=lambda x: bool(distutils.util.strtobool(x)))
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--use", type=str, default="forecast", help="Use forecast or pretrain")
    args = parser.parse_args()

    return args


def main(args):
    benchmark(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)

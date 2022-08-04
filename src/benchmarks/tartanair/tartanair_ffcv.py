import argparse
from distutils.util import strtobool

from ffcv.fields.decoders import NDArrayDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader
from ffcv.pipeline.compiler import Compiler
from ffcv.transforms import ToTensor
from tartanair_ops import get_tartanair_args
from utils_ffcv import get_order_option

from src.benchmarks.benchmarker import Benchmarker
from src.benchmarks.common_opts import get_common_args

Compiler.set_enabled(True)


def get_dataloader(args):
    # Dataset specific
    PIPELINES = {
        "rgb": [SimpleRGBImageDecoder(), ToTensor()],
        "depth": [NDArrayDecoder(), ToTensor()],
        "flow": [NDArrayDecoder(), ToTensor()],
    }

    order = get_order_option(args.order)

    dataloader = Loader(
        fname=args.beton_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        order=order,
        os_cache=args.os_cache,
        pipelines=PIPELINES,
    )

    return dataloader


def benchmark(args):
    dataloader = get_dataloader(args)
    benchmarker = Benchmarker(verbose=args.verbose, library="ffcv", dataset="tartanair")
    benchmarker.set_dataloader(dataloader)
    benchmarker.benchmark_tartanair(args)


def get_ffcv_args():
    parser = argparse.ArgumentParser(description="FFCV options")
    parser.add_argument(
        "--beton_file",
        type=str,
        default="/datadrive/localdatasets/tartanair-release1/tartan_abandonedfactory_ratnesh.beton",
        help="path to beton file",
    )
    parser.add_argument("--order", type=str, default="random", help="Ordering of data: random or quasi_random")
    parser.add_argument("--os_cache", type=lambda x: bool(strtobool(x)))
    args = parser.parse_args()
    return args


def main(args):
    benchmark(args)


if __name__ == "__main__":
    args = get_common_args()
    tartanair_args = get_tartanair_args()
    ffcv_args = get_ffcv_args()

    args.__dict__.update(tartanair_args.__dict__)
    args.__dict__.update(ffcv_args.__dict__)

    main(args)

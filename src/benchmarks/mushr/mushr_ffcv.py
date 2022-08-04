import argparse
from distutils.util import strtobool

from ffcv.fields.decoders import NDArrayDecoder
from ffcv.loader import Loader
from ffcv.pipeline.compiler import Compiler
from ffcv.transforms import ToTensor
from mushr_ops import get_mushr_args
from utils_ffcv import get_order_option

from src.benchmarks.benchmarker import Benchmarker
from src.benchmarks.common_opts import get_common_args

Compiler.set_enabled(True)


def get_dataloader(args):
    # Dataset specific
    PIPELINES = {
        "states": [NDArrayDecoder(), ToTensor()],
        "actions": [NDArrayDecoder(), ToTensor()],
        "poses": [NDArrayDecoder(), ToTensor()],
    }

    dataloader = Loader(
        fname=args.beton_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        order=get_order_option(args.order),
        os_cache=args.os_cache,
        pipelines=PIPELINES,
    )

    return dataloader


def benchmark(args):
    dataloader = get_dataloader(args)
    benchmarker = Benchmarker(verbose=args.verbose, dataset="mushr", library="ffcv")
    benchmarker.set_dataloader(dataloader)
    benchmarker.benchmark_mushr(args)


def get_custom_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--beton_file", type=str, default="./tartan_abandonedfactory_ratnesh.beton", help="path to beton file"
    )
    parser.add_argument("--os_cache", type=lambda x: bool(strtobool(x)))

    args = parser.parse_args()

    return args


def main(args):
    benchmark(args)


if __name__ == "__main__":
    args = get_common_args()
    mushr_args = get_mushr_args()
    custom_args = get_custom_args()

    args.__dict__.update(mushr_args.__dict__)
    args.__dict__.update(custom_args.__dict__)

    main(args)

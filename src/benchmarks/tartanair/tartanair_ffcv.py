import argparse

from ffcv.fields.decoders import NDArrayDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader
from ffcv.pipeline.compiler import Compiler
from ffcv.transforms import ToTensor
from utils_ffcv import get_order_option

from src.benchmarks.benchmarker import Benchmarker
from src.benchmarks.common_opts import add_common_args, add_ffcv_args
from src.benchmarks.tartanair.tartanair_opts import add_tartanair_args

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


def main(args):
    benchmark(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    add_common_args(parser.add_argument_group(title="common args"))
    add_tartanair_args(parser.add_argument_group(title="tartanair args"))
    add_ffcv_args(parser.add_argument_group(title="ffcv args"))

    args = parser.parse_args()

    main(args)

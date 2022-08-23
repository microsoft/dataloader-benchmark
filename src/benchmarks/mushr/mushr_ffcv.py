import argparse

from ffcv.fields.decoders import NDArrayDecoder
from ffcv.loader import Loader
from ffcv.pipeline.compiler import Compiler
from ffcv.transforms import ToTensor
from utils_ffcv import get_order_option

from src.benchmarks.benchmarker import Benchmarker
from src.benchmarks.common_opts import add_common_args, add_ffcv_args
from src.benchmarks.mushr.mushr_opts import add_mushr_args

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


def main(args):
    benchmark(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    add_common_args(parser.add_argument_group(title="common args"))
    add_mushr_args(parser.add_argument_group(title="mushr args"))
    add_ffcv_args(parser.add_argument_group(title="ffcv args"))

    args = parser.parse_args()

    main(args)

import argparse
import distutils

from benchmarker import Benchmarker
from ffcv.fields.decoders import NDArrayDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader
from ffcv.pipeline.compiler import Compiler
from ffcv.transforms import ToTensor
from ffcv_common import get_order_option_ffcv

Compiler.set_enabled(True)


def get_ffcv_dataloader(args):
    # Dataset specific
    PIPELINES = {
        "rgb": [SimpleRGBImageDecoder(), ToTensor()],
        "depth": [NDArrayDecoder(), ToTensor()],
        "flow": [NDArrayDecoder(), ToTensor()],
    }

    order = get_order_option_ffcv(args.order)

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
    dataloader = get_ffcv_dataloader(args)

    benchmarker = Benchmarker()
    benchmarker.set_dataloader(dataloader)
    benchmarker.benchmark_tartanair(args)


def get_parsed_args():
    parser = argparse.ArgumentParser(description="FFCV options")
    parser.add_argument("--dataset", type=str, default="tartanair", help="dataset type to use for benchmarking")
    parser.add_argument(
        "--beton_file", type=str, default="./tartan_abandonedfactory_ratnesh.beton", help="path to beton file"
    )
    parser.add_argument("--tartanair_ann_file", type=str, default="./train_ann_debug_ratnesh.json", help="")
    parser.add_argument("--order", type=str, default="random", help="Ordering of data: random or quasi_random")
    parser.add_argument("--os_cache", type=lambda x: bool(distutils.util.strtobool(x)))
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers")
    parser.add_argument(
        "--modalities",
        default=["image_left", "depth_left", "flow_flow"],
        help="list of modalities (strings)",
        nargs="+",
        type=str,
        choices=[
            "image_left",
            "image_right",
            "depth_left",
            "depth_right",
            "flow_mask",
            "flow_flow",
            "seg_left",
            "seg_right",
        ],
    )

    args = parser.parse_args()

    return args


def main(args):
    benchmark(args)


if __name__ == "__main__":
    args = get_parsed_args()
    main(args)

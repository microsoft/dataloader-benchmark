import argparse
import distutils

from benchmarker import Benchmarker
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.loader import Loader
from ffcv.pipeline.compiler import Compiler
from ffcv.transforms import ToTensor
from utils_ffcv import get_order_option

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

    benchmarker = Benchmarker()
    benchmarker.set_dataloader(dataloader)
    benchmarker.benchmark_mushr(args)


def main(args):
    benchmark(args)


def get_parsed_args():
    parser = argparse.ArgumentParser(description="FFCV options")
    parser.add_argument("--benchmark_results_file", default="benchmark_results_mushr.csv", type=str)
    parser.add_argument("--dataset", type=str, default="mushr", help="dataset type to use for benchmarking")
    parser.add_argument(
        "--beton_file", type=str, default="./tartan_abandonedfactory_ratnesh.beton", help="path to beton file"
    )
    parser.add_argument("--mushr_ann_file", type=str, default="./train_ann_pose.json", help="")
    parser.add_argument("--mushr_gt_map_file_name", type=str, default="bravern_floor.pgm", help="")
    parser.add_argument("--mushr_dir", type=str, default="./pretraining_data/hackathon_data_2p5_nonoise3", help="")
    parser.add_argument("--order", type=str, default="random", help="Ordering of data: random or quasi_random")
    parser.add_argument("--os_cache", type=lambda x: bool(distutils.util.strtobool(x)))
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_parsed_args()
    main(args)

import argparse
import distutils
from timeit import default_timer as timer

import mlflow
from ffcv.fields.decoders import NDArrayDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.compiler import Compiler
from ffcv.transforms import ToTensor

Compiler.set_enabled(True)


def parse_args():
    parser = argparse.ArgumentParser(description="FFCV options")
    parser.add_argument("--dataset", type=str, default="tartanair", help="dataset type to use for benchmarking")
    parser.add_argument(
        "--beton_file", type=str, default="./tartan_abandonedfactory_ratnesh.beton", help="path to beton file"
    )
    parser.add_argument("--tartanair_ann_file", type=str, default="./train_ann_debug_ratnesh.json", help="")
    parser.add_argument("--mushr_dir", type=str, default="./pretraining_data/hackathon_data_2p5_nonoise3", help="")
    parser.add_argument("--order", type=str, default="random", help="Ordering of data: random or quasi_random")
    parser.add_argument("--os_cache", type=lambda x: bool(distutils.util.strtobool(x)))
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers")

    args = parser.parse_args()

    return args


def get_order_option_ffcv(order):
    if order == "random":
        order_option = OrderOption.RANDOM
    elif order == "quasi_random":
        order_option = OrderOption.QUASI_RANDOM
    elif order == "sequential":
        order_option = OrderOption.SEQUENTIAL
    else:
        raise ValueError(f"Unknown order option: {order}")
    return order_option


def benchmark_tartanair_ffcv(args):
    print("==== TartanAir FFCV ====")
    start = timer()
    last = start

    # Dataset specific
    PIPELINES = {
        "rgb": [SimpleRGBImageDecoder(), ToTensor()],
        "depth": [NDArrayDecoder(), ToTensor()],
        "flow": [NDArrayDecoder(), ToTensor()],
    }

    loader = Loader(
        fname=args.beton_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        order=get_order_option_ffcv(args.order),
        os_cache=args.os_cache,
        pipelines=PIPELINES,
    )

    for batch_idx, batch in enumerate(loader):
        print(f"batch_idx: {batch_idx:05}")
        if batch_idx == 0:
            first = timer()
        print(f"{(timer() - last):.3f} secs for this batch")
        last = timer()

    last = timer()

    time_first_batch = first - start
    time_per_batch = (last - start) / batch_idx
    time_per_batch_without_first = (last - first) / (batch_idx - 1)

    print(f"{time_first_batch:.3f} secs for the first batch")
    print(f"{time_per_batch:.3f} secs per batch")
    print(f"{time_per_batch_without_first:.3f} secs per batch without counting first batch")

    mlflow.log_metric(key="time_per_batch_without_first", value=time_per_batch_without_first, step=0)
    mlflow.log_metric(key="time_per_batch", value=time_per_batch, step=0)
    mlflow.log_metric(key="time_first_batch", value=time_first_batch, step=0)


def main(args):
    benchmark_tartanair_ffcv(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)

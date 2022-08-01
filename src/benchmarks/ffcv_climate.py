import argparse
import distutils
from timeit import default_timer as timer

import mlflow
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.loader import Loader
from ffcv.transforms import ToTensor
from ffcv_common import get_order_option_ffcv


def parse_args():
    parser = argparse.ArgumentParser(description="FFCV options")
    parser.add_argument("--beton_file", type=str, help="Dataset to use for benchmarking")
    parser.add_argument("--order", type=str, default="random", help="Ordering of data: random or quasi_random")
    parser.add_argument("--os_cache", type=lambda x: bool(distutils.util.strtobool(x)))
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--use", type=str, default="forecast", help="Use forecast or pretrain")
    args = parser.parse_args()

    return args


def benchmark_climate_ffcv(args):
    print("===== Benchmarking =====")
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
    loader = Loader(
        args.beton_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        order=get_order_option_ffcv(args.order),
        os_cache=args.os_cache,
        pipelines=PIPELINES,
    )
    time_copy = 0.0
    num_batches = 0
    start = timer()
    last = start
    for idx, batch in enumerate(loader):
        start_copy = timer()
        if args.use == "forecast":
            x, y = batch[0].cuda(), batch[1].cuda()
        elif args.use == "pretrain":
            traj = batch[0].cuda()
        time_copy += timer() - start_copy
        num_batches += 1
        if idx == 0:
            first = timer()

    last = timer()

    time_copy_per_batch = time_copy / num_batches
    time_first_batch = first - start
    time_per_batch = (last - start) / num_batches
    time_per_batch_without_first = (last - first) / (num_batches - 1)
    print(f"{time_first_batch:.3f} secs for the first batch")
    print(f"{time_per_batch:.3f} secs per batch")
    print(f"{time_per_batch_without_first:.3f} secs per batch without counting first batch")
    print(f"{time_copy_per_batch:.3f} secs per batch for copying from cpu to gpu")
    mlflow.log_metric(key="num_workers", value=args.num_workers, step=0)
    mlflow.log_metric(key="batch_size", value=args.batch_size, step=0)
    mlflow.log_metric(key="time_per_batch_without_first", value=time_per_batch_without_first, step=0)
    mlflow.log_metric(key="time_per_batch", value=time_per_batch, step=0)
    mlflow.log_metric(key="time_first_batch", value=time_first_batch, step=0)
    mlflow.log_metric(key="time_copy_per_batch", value=time_copy_per_batch, step=0)


if __name__ == "__main__":
    benchmark_climate_ffcv(parse_args())

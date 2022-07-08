import argparse
import glob
import os
from timeit import default_timer as timer

import mlflow
import torch
import torchdata.datapipes as dp
from torch.utils.data import DataLoader

from climate.era5_datapipe import ERA5Forecast, ERA5Npy, ERA5Zarr, IndividualDataIter
from utils.utils import AverageMeter

# NPY = False
# NPY_PATH = "/mnt/data/1.40625/_yearly_np"
# ZARRY_PATH = "/mnt/data/1.40625_yearly"


def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    return inp, out


def get_datapipe(path, batchsize=32, NPY=False):
    if NPY:
        READER = ERA5Npy
        lister = dp.iter.FileLister(path)
    else:
        READER = ERA5Zarr
        lister = dp.iter.IterableWrapper(glob.glob(os.path.join(path, "*.zarr")))

    data = (
        IndividualDataIter(
            ERA5Forecast(
                READER(
                    lister.shuffle().sharding_filter(),  # shuffle at the year level  # needed for num_workers > 1
                    variables=["t", "u10", "v10"],
                )
            ),
        )
        .shuffle(buffer_size=1000)  # shuffle at the individual data level
        .batch(batchsize)
        .in_batch_shuffle()  # shuffle within a batch, probably not necessary
        .collate(collate_fn)
    )
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="FFCV options")
    parser.add_argument(
        "--dataset", type=str, default="npy", help="Dataset to use for benchmarking"
    )
    parser.add_argument("--datapath", type=str, default=None, help="Path to dataset")
    parser.add_argument(
        "--batchsize", type=int, default=32, help="Batchsize for dataloader"
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers for dataloader"
    )
    args = parser.parse_args()
    return args


def benchmark(args):
    print("===== Benchmarking =====")
    print(f"Dataset: {args.dataset}\n \t {args.datapath}")
    data = get_datapipe(args.datapath, args.batchsize, args.dataset == "npy")
    dl = DataLoader(data, batch_size=None, num_workers=args.num_workers)
    time_copy = 0.0
    num_batches = 0
    start = timer()
    last = start
    for idx, batch in enumerate(dl):
        start_copy = timer()
        x, y = batch[0].cuda(), batch[1].cuda()
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
    print(
        f"{time_per_batch_without_first:.3f} secs per batch without counting first batch"
    )
    print(f"{time_copy_per_batch:.3f} secs per batch for copying from cpu to gpu")
    mlflow.log_metric(key="num_workers", value=args.num_workers, step=0)
    mlflow.log_metric(key="batch_size", value=args.batchsize, step=0)
    mlflow.log_metric(
        key="time_per_batch_without_first", value=time_per_batch_without_first, step=0
    )
    mlflow.log_metric(key="time_per_batch", value=time_per_batch, step=0)
    mlflow.log_metric(key="time_per_batch", value=time_per_batch, step=0)
    mlflow.log_metric(key="time_copy_per_batch", value=time_copy_per_batch, step=0)


if __name__ == "__main__":
    benchmark(parse_args())

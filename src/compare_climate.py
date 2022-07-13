import argparse
import glob
import os
from timeit import default_timer as timer

import mlflow
import numpy as np
import torch
import torchdata.datapipes as dp
from torch.utils.data import DataLoader

from climate.era5_datapipe import (ERA5, NAME_MAP, ERA5Forecast, ERA5Npy,
                                   ERA5Zarr, IndividualDataIter,
                                   IndividualForecastDataIter)

# NPY = False
# NPY_PATH = "/mnt/data/1.40625/_yearly_np"
# ZARRY_PATH = "/mnt/data/1.40625_yearly"
# DEFAULT_VARS = list(NAME_MAP.values())
DEFAULT_VARS = ["z", "r", "u", "v", "t", "t2m", "u10", "v10"]


def collate_fn(batch):
    inp = np.stack([batch[i] for i in range(len(batch))])
    return inp


def collate_forecast_fn(batch):
    inp = np.stack([batch[i][0] for i in range(len(batch))])
    out = np.stack([batch[i][1] for i in range(len(batch))])
    return inp, out


def np_to_th(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, tuple):
        return tuple(np_to_th(i) for i in x)


def get_datapipe(path, batchsize=32, use="forecast", dataformat="npy"):
    if dataformat == "npy":
        READER = ERA5Npy
        lister = dp.iter.FileLister(path)
    elif dataformat == "zarr":
        READER = ERA5Zarr
        lister = dp.iter.IterableWrapper(glob.glob(os.path.join(path, "*.zarr")))
    else:
        raise NotImplementedError(f"Data {dataformat} is not implemented")

    if use == "pretrain":
        dataset_class = ERA5
        data_iter = IndividualDataIter
        cfun = collate_fn
    elif use == "forecast":
        dataset_class = ERA5Forecast
        data_iter = IndividualForecastDataIter
        cfun = collate_forecast_fn
    else:
        raise NotImplementedError("only pretrain/forecast use supported")

    data = (
        data_iter(
            dataset_class(
                READER(
                    lister.shuffle().sharding_filter(),  # shuffle at the year level  # needed for num_workers > 1
                    variables=DEFAULT_VARS,
                )
            ),
        )
        .shuffle(buffer_size=10000)  # shuffle at the individual data level
        .batch(batchsize)
        .in_batch_shuffle()  # shuffle within a batch, probably not necessary
        .collate(cfun)
        .map(np_to_th)
    )
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="FFCV options")
    parser.add_argument("--dataset", type=str, default="npy", help="Dataset to use for benchmarking")
    parser.add_argument("--datapath", type=str, default=None, help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batchsize for dataloader")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for dataloader")
    parser.add_argument("--use", type=str, default="forecast", help="Use forecast or pretrain")
    args = parser.parse_args()
    return args


def benchmark(args):
    print("===== Benchmarking =====")
    print(f"Dataset: {args.dataset}\n \t {args.datapath}")
    data = get_datapipe(args.datapath, args.batch_size, args.use, args.dataset)
    dl = DataLoader(data, batch_size=None, num_workers=args.num_workers)
    time_copy = 0.0
    num_batches = 0
    start = timer()
    last = start
    for idx, batch in enumerate(dl):
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
    mlflow.log_metric(key="time_per_batch", value=time_per_batch, step=0)
    mlflow.log_metric(key="time_copy_per_batch", value=time_copy_per_batch, step=0)


if __name__ == "__main__":
    benchmark(parse_args())

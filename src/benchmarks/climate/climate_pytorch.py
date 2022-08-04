import glob
import os

import numpy as np
import torch
import torchdata.datapipes as dp
from climate_ops import get_climate_args
from torch.utils.data import DataLoader

from src.benchmarks.benchmarker import Benchmarker
from src.benchmarks.common_opts import get_common_args
from src.data.climate.era5_datapipe import (
    ERA5,
    ERA5Forecast,
    ERA5Npy,
    ERA5Zarr,
    IndividualDataIter,
    IndividualForecastDataIter,
)

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


def get_dataloader(args):
    print(f"Dataset: {args.dataset}\n \t {args.data_dir}")
    data = get_datapipe(args.data_dir, args.batch_size, args.use, args.dataset)
    dataloader = DataLoader(data, batch_size=None, num_workers=args.num_workers)
    return dataloader


def benchmark(args):
    dataloader = get_dataloader(args)
    benchmarker = Benchmarker(verbose=args.verbose, dataset=f"climate_{args.use}", library="pytorch")
    benchmarker.set_dataloader(dataloader)
    benchmarker.benchmark_climate(args)


def main(args):
    benchmark(args)


if __name__ == "__main__":
    args = get_common_args()
    climate_args = get_climate_args()

    args.__dict__.update(climate_args.__dict__)

    main(args)

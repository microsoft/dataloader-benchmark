import glob
import os
import time

import torch
import torchdata.datapipes as dp
from torch.utils.data import DataLoader

from climate.era5_datapipe import ERA5Forecast, ERA5Npy, ERA5Zarr, IndividualDataIter
from utils.utils import AverageMeter

NPY = False
NPY_PATH = "/mnt/data/1.40625/_yearly_np"
ZARRY_PATH = "/mnt/data/1.40625_yearly"


def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    return inp, out


def get_datapipe(batchsize=32, NPY=False):
    if NPY:
        READER = ERA5Npy
        lister = dp.iter.FileLister(NPY_PATH)
    else:
        READER = ERA5Zarr
        lister = dp.iter.IterableWrapper(glob.glob(os.path.join(ZARRY_PATH, "*.zarr")))

    dp = (
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
    return dp


print("== NPY ==")
dp = get_datapipe(NPY=True)
time_start = time.time()
batch_time = AverageMeter()
ts = time.time()
for x, y in DataLoader(dp):
    batch_time.update(time.time() - ts)
    ts = time.time()

print(f"Time per batch: {batch_time.avg:.3f}")
print(f"Total time: {time.time() - time_start:.3f}")

print("== ZARR ==")
dp = get_datapipe(NPY=True)
time_start = time.time()
batch_time = AverageMeter()
ts = time.time()
for x, y in DataLoader(dp):
    batch_time.update(time.time() - ts)
    ts = time.time()

print(f"Time per batch: {batch_time.avg:.3f}")
print(f"Total time: {time.time() - time_start:.3f}")

import glob
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torchdata.datapipes as dp
from ffcv.fields import NDArrayField
from ffcv.writer import DatasetWriter
from torch.utils.data import DataLoader

from climate.era5_datapipe import ERA5Forecast, ERA5Npy, ERA5Zarr, IndividualDataIter


def collate_fn(batch):
    assert len(batch) == 1
    return batch[0]


def to_tuple(bb):
    return tuple(list(bb.values()))


def get_noshuffle_datapipe(path, batchsize=1, dataformat="npy"):
    if dataformat == "npy":
        READER = ERA5Npy
        lister = dp.iter.FileLister(path)
    elif dataformat == "zarr":
        READER = ERA5Zarr
        lister = dp.iter.IterableWrapper(glob.glob(os.path.join(path, "*.zarr")))
    else:
        raise NotImplementedError(f"Data {dataformat} is not implemented")

    data = (
        READER(
            lister,
            variables=["t", "u10", "v10"],
        )
        .batch(batchsize)
        .collate(collate_fn)
    )

    return data


def convert_climate(args):
    datap = get_noshuffle_datapipe(args.datapath, dataformat=args.dataset)
    data = None
    for dd in datap:
        data = dd
        for k, v in dd.items():
            print(k, v.shape)
        break

    writer = DatasetWriter(args.outfile, {k: NDArrayField(shape=v.shape, dtype=v.dtype) for k, v in data.items()})

    def pipeline(dd):
        return dd.map(to_tuple)

    writer.from_webdataset(datap, pipeline=pipeline)


def parse_args():
    parser = ArgumentParser(description="FFCV options")
    parser.add_argument("--datapath", type=str, help="Path to the data")
    parser.add_argument("--dataset", type=str, default="npy", choices=["npy", "zarr"])
    parser.add_argument("--outfile", type=str, help="Path to the output file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    convert_climate(parse_args())

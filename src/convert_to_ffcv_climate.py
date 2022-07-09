import glob
import os
from argparse import ArgumentParser
from functools import partial
from unittest.mock import DEFAULT

import numpy as np
import torch
import torchdata.datapipes as dp
from ffcv.fields import NDArrayField
from ffcv.writer import DatasetWriter, handle_sample
from torch.utils.data import DataLoader
from tqdm.contrib.concurrent import thread_map

from climate.era5_datapipe import (NAME_MAP, ERA5Forecast, ERA5Npy, ERA5Zarr,
                                   IndividualDataIter)

DEFAULT_VARS = list(NAME_MAP.values())


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
            variables=DEFAULT_VARS,
        )
        .batch(batchsize)
        .collate(collate_fn)
    )

    return data


def get_lister_dp(path, dataformat="npy"):
    if dataformat == "npy":
        lister = dp.iter.FileLister(path)
    elif dataformat == "zarr":
        lister = dp.iter.IterableWrapper(glob.glob(os.path.join(path, "*.zarr")))
    else:
        raise NotImplementedError(f"Data {dataformat} is not implemented")
    return lister


def count_samples_in_datapipe(file, pipeline):
    if isinstance(file, str):
        file = dp.iter.IterableWrapper([file])

    count = 0
    for dd in pipeline(file):
        count += 1
    return count


def worker_job_datapipe(
    input_queue, metadata_sm, metadata_type, fields, allocator, done_number, allocations_queue, pipeline
):
    # basically follows worker_job_webdataset with appropriate changes
    metadata = np.frombuffer(metadata_sm.buf, dtype=metadata_type)
    field_names = metadata_type.names
    # This `with` block ensures that all the pages allocated have been written
    # onto the file
    with allocator:
        while True:
            todo = input_queue.get()
            if todo is None:
                # No more work
                break

            file, offset = todo
            actpipe = dp.iter.IterableWrapper([file])
            done = 0
            for i, sample in enumerate(pipeline(actpipe)):
                done += 1
                dest_ix = offset + i
                handle_sample(sample, dest_ix, field_names, metadata, allocator, fields)

            with done_number.get_lock():
                done_number.value += done

    allocations_queue.put(allocator.allocations)


class DataPipeWriter(DatasetWriter):
    def from_datapipe(self, filelister, pipeline=None):
        counter = partial(count_samples_in_datapipe, pipeline=pipeline)
        lengths = list(thread_map(counter, filelister))
        total_len = sum(lengths)
        offsets = np.cumsum([0] + lengths)[:-1]
        todos = zip(filelister, offsets)
        self._write_common(total_len, todos, worker_job_datapipe, (pipeline,))


def convert_climate(args):
    datap = get_noshuffle_datapipe(args.datapath, dataformat=args.dataset)
    data = None
    for dd in datap:
        data = dd
        for k, v in dd.items():
            print(k, v.shape)
        break

    writer = DataPipeWriter(args.outfile, {k: NDArrayField(shape=v.shape, dtype=v.dtype) for k, v in data.items()})

    def pipeline(datapipe):
        if args.dataset == "npy":
            reader = ERA5Npy
        elif args.dataset == "zarr":
            reader = ERA5Zarr
        return reader(datapipe, variables=DEFAULT_VARS).batch(1).collate(collate_fn).map(to_tuple)

    filedp = get_lister_dp(args.datapath, dataformat=args.dataset)

    writer.from_datapipe(filedp, pipeline=pipeline)


def parse_args():
    parser = ArgumentParser(description="FFCV options")
    parser.add_argument("--datapath", type=str, help="Path to the data")
    parser.add_argument("--dataset", type=str, default="npy", choices=["npy", "zarr"])
    parser.add_argument("--outfile", type=str, help="Path to the output file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    convert_climate(parse_args())

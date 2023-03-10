import argparse
import os
from distutils.util import strtobool
from functools import partial
from random import shuffle
from timeit import default_timer as timer

import mlflow
import numpy as np
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from src.benchmarks.climate.climate_opts import add_climate_args
from src.benchmarks.common_opts import add_common_args

batch_size = 32


class ExternalPretrainInputCallableSharded:
    def __init__(self, data_dir, variables, batch_size, debug_print, debug_print_each_sample, shard_id=0, num_shards=1):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.variables = variables
        self.files = os.listdir(data_dir)
        shuffle(self.files)

        self.shard_id = shard_id
        self.dataset_size = len(self.files)
        self.shard_size = self.dataset_size // num_shards
        self.shard_offset = self.shard_size * self.shard_id
        self.num_samples_in_epoch = 8760 * 2 // num_shards
        # todo bug. num hours per year. parameterize
        # perhaps keep track of num files? leap years, 100/4, exceptions can be problematic
        self.perm = None  # permutation of indices
        self.last_seen_epoch = None  # so that we don't have to recompute the `self.perm` for every sample

        self.sample_idx_cum = 0
        self.file_idx = 0
        self.debug_print = debug_print
        self.debug_print_each_sample = debug_print_each_sample
        # self.buffer_size = buffer_size
        # self.buffer: List[np.ndarray] = []
        self.buffer = None

        self.load_next_file_into_buffer()

    # currently just loads one monthly file at a time
    def load_next_file_into_buffer(self):
        # shuffle every epoch
        # use self.perm to shuffle file order and hour order?

        self.fname = self.files[self.file_idx]
        if self.debug_print:
            print("\n\n\nload_next_file_into_buffer():")
            print(self.fname)
        start = timer()
        data = np.load(os.path.join(self.data_dir, self.fname))  # how to use fn.readers.numpy here
        self.buffer = np.concatenate([data[k] for k in self.variables], axis=1)
        # self.buffer.append(np.concatenate([data[k] for k in self.variables], axis=1))
        if self.debug_print:
            print(f"time taken to load file: {(timer() - start):.3f} seconds")
        # todo
        # mmap_curr = np.memmap(os.path.join(self.data_dir, self.fname), dtype='float32', mode='w+', shape=(,17,128,256)) # shape depends on month
        # append mmap to buffer

        # shuffle buuffle arrary on hour dim
        rand_indices_hourly = np.arange(self.buffer.shape[0])
        np.random.shuffle(rand_indices_hourly)
        self.buffer = self.buffer[rand_indices_hourly]

        self.sample_idx_curr_npy_file = 0
        self.max_sample_idx_curr_npy_file = self.buffer.shape[0] - 1
        self.file_idx += 1

    def __call__(self, sample_info):
        # shuffle every epoch
        if self.last_seen_epoch != sample_info.epoch_idx:
            self.last_seen_epoch = sample_info.epoch_idx
            self.perm = np.random.default_rng(seed=42 + sample_info.epoch_idx).permutation(self.shard_size)

        sample_idx = sample_info.idx_in_epoch
        if (sample_info.iteration >= self.num_samples_in_epoch) or (
            (self.file_idx - self.shard_offset) > (self.shard_size - 1)
        ):
            # Indicate end of the epoch
            raise StopIteration()

        sample = self.buffer[self.sample_idx_curr_npy_file, :]

        if self.sample_idx_curr_npy_file == self.max_sample_idx_curr_npy_file:
            self.load_next_file_into_buffer()
            if self.debug_print:
                print(
                    f"{self.fname}, self.sample_idx_curr_npy_file: {self.sample_idx_curr_npy_file}, self.max_sample_idx_curr_npy_file: {self.max_sample_idx_curr_npy_file}, self.sample_idx_cum: {self.sample_idx_cum}"
                )
            return sample

        self.sample_idx_curr_npy_file += 1
        self.sample_idx_cum += 1
        if self.debug_print_each_sample:
            print(
                f"{self.fname}, self.sample_idx_curr_npy_file: {self.sample_idx_curr_npy_file}, self.max_sample_idx_curr_npy_file: {self.max_sample_idx_curr_npy_file}, self.sample_idx_cum: {self.sample_idx_cum}"
            )
        return sample


@pipeline_def(batch_size=batch_size, num_threads=2, device_id=0, py_num_workers=6, py_start_method="spawn")
def get_parallel_callable_pretrain_pipeline_sharded(
    data_dir, variables, debug_print, debug_print_each_sample, shard_id, num_shards
):
    print("\n\n\nget_parallel_callable_pretrain_pipeline_sharded():")
    data = fn.external_source(
        source=ExternalPretrainInputCallableSharded(
            data_dir,
            variables,
            debug_print=debug_print,
            debug_print_each_sample=debug_print_each_sample,
            batch_size=batch_size,
            shard_id=shard_id,
            num_shards=num_shards,
            # reader_name=f"reader_{shard_id}",
        ),
        batch=False,
        parallel=True
        # num_outputs=1,
        # dtype=[types.FLOAT],
        # reader_name="Reader",
    )
    return data


def benchmark_parallel_callable_pretrain_pipeline_sharded(args):
    print("\n\n\nbenchmark_parallel_callable_pretrain_pipeline_sharded():")
    start = timer()
    last = start

    pipe_partial = partial(
        get_parallel_callable_pretrain_pipeline_sharded,
        data_dir=args.data_dir,
        variables=args.variables,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        device_id=args.device_id,
        debug_print=args.debug_print,
        debug_print_each_sample=args.debug_print_each_sample,
        py_num_workers=args.py_num_workers,
    )

    pipe_list = []

    for shard_id in range(args.num_shards):
        pipe_list.append(pipe_partial(shard_id=shard_id, num_shards=args.num_shards))

    # dali_iter = DALIGenericIterator(pipe, ["data"], reader_name="Reader")
    dali_iter = DALIGenericIterator(
        pipe_list, ["data"], size=(8760 * 2 / args.num_shards)
    )  # bug hardcoded incorrect num

    num_batches = 0
    for batch_idx, batch_list in enumerate(dali_iter):
        print(f"batch_idx: {batch_idx:05}, sample_idx: {args.batch_size*(num_batches+1):07}")
        if batch_idx == 0:
            first = timer()
            for shard_idx in range(args.num_shards):
                print(f"batch shape: {batch_list[shard_idx]['data'].shape}")
        for shard_idx in range(args.num_shards):
            print(f"batch[shard_idx={shard_idx}].shape: {batch_list[shard_idx]['data'].shape}")
        print(f"{(timer() - last):.3f} secs for these {args.num_shards} batches")
        print(f"{((timer() - last)/float(args.num_shards)):.3f} per batch")
        last = timer()
        num_batches += 1 * args.num_shards

    last = timer()

    time_first_batch = first - start
    time_per_batch = (last - start) / num_batches
    time_per_batch_without_first = (last - first) / num_batches

    print(f"{time_first_batch:.3f} secs for the first batch")
    print(f"{time_per_batch:.3f} secs per batch")
    print(f"{time_per_batch_without_first:.3f} secs per batch without counting first batch")

    mlflow.log_metric(key="time_per_batch_without_first", value=time_per_batch_without_first, step=0)
    mlflow.log_metric(key="time_per_batch", value=time_per_batch, step=0)
    mlflow.log_metric(key="time_first_batch", value=time_first_batch, step=0)


def add_dali_args(group):
    group.add_argument("--seed", type=int, default=42)
    group.add_argument("--num_threads", type=int, default=6)
    group.add_argument("--py_num_workers", type=int, default=2)
    group.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "gpu"]
    )  # use gpu for GPUDirect Storage Support. needs cuda>=11.4
    group.add_argument("--device_id", type=int, default=0)
    group.add_argument("--random_shuffle", default="yes", type=lambda x: bool(strtobool(x)))
    group.add_argument("--debug_print", default="no", type=lambda x: bool(strtobool(x)))
    group.add_argument("--debug_print_each_sample", default="no", type=lambda x: bool(strtobool(x)))
    group.add_argument("--prefetch_queue_depth", type=int, default=12)
    group.add_argument("--initial_fill", type=int, default=2)
    group.add_argument(
        "--data_dir",
        type=str,
        default="/datadrive/localdatasets/climate/1.40625deg_monthly_np/val",
    )
    group.add_argument("--is_amlt", default="no", type=lambda x: bool(strtobool(x)))
    group.add_argument("--read_ahead", default="yes", type=lambda x: bool(strtobool(x)))
    group.add_argument(
        "--cache_header_information",
        default="yes",
        type=lambda x: bool(strtobool(x)),
        help="If set to True, the header information for each file is cached, improving access speed.",
    )


def main(args):
    # test_parallel_callable_pretrain_single_batch_sharded_partial(args)
    benchmark_parallel_callable_pretrain_pipeline_sharded(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    add_common_args(parser.add_argument_group("common args"))
    add_climate_args(parser.add_argument_group("climate args"))
    add_dali_args(parser.add_argument_group("dali args"))

    args = parser.parse_args()
    main(args)

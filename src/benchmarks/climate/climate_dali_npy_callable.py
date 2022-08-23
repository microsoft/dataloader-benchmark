import argparse
import os
from distutils.util import strtobool

# from data.climate.era5_datapipe import ERA5Npy
from random import shuffle
from timeit import default_timer as timer

import numpy as np
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from src.benchmarks.benchmarker import Benchmarker
from src.benchmarks.climate.climate_opts import add_climate_args
from src.benchmarks.common_opts import add_common_args

batch_size = 32


class ExternalPretrainInputCallable:
    def __init__(self, data_dir, variables, batch_size, debug_print, debug_print_each_sample):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.variables = variables
        self.files = os.listdir(data_dir)
        shuffle(self.files)
        self.num_samples_in_epoch = 8760
        # todo bug. num hours per year. parameterize
        # perhaps keep track of num files? leap years, 100/4, exceptions can be problematic

        self.sample_idx_cum = 0
        self.file_idx = 0
        self.debug_print = debug_print
        self.debug_print_each_sample = debug_print_each_sample
        # self.buffer_size = buffer_size
        # self.buffer: List[np.ndarray] = []
        # todo buffer.append(next(generator))
        self.buffer = None

        self.load_next_file_into_buffer()

    # currently just loads one monthly file at a time
    def load_next_file_into_buffer(self):
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
        sample_idx = sample_info.idx_in_epoch
        if (sample_info.iteration >= self.num_samples_in_epoch) or (self.file_idx > (len(self.files) - 1)):
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
def get_parallel_callable_pretrain_pipeline(data_dir, variables, debug_print, debug_print_each_sample):
    print("\n\n\nget_parallel_callable_pretrain_pipeline():")
    data = fn.external_source(
        source=ExternalPretrainInputCallable(
            data_dir,
            variables,
            debug_print=debug_print,
            debug_print_each_sample=debug_print_each_sample,
            batch_size=batch_size,
        ),
        batch=False,
        parallel=True
        # num_outputs=1,
        # dtype=[types.FLOAT],
        # reader_name="Reader",
    )
    data = data.gpu()
    return data


def get_dataloader(args):
    pipe = get_parallel_callable_pretrain_pipeline(
        data_dir=args.data_dir,
        variables=args.variables,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        device_id=args.device_id,
        debug_print=args.debug_print,
        debug_print_each_sample=args.debug_print_each_sample,
        py_num_workers=args.num_workers,
        # random_shuffle=args.random_shuffle,
        # initial_fill=args.initial_fill,
        # read_ahead=args.read_ahead,
        # seed=args.seed,
    )

    dataloader = DALIGenericIterator(pipe, ["data"])
    return dataloader


def benchmark(args):
    dataloader = get_dataloader(args)
    benchmarker = Benchmarker(verbose=args.verbose, dataset="climate", library="dali_npy_callable")
    benchmarker.set_dataloader(dataloader)
    benchmarker.benchmark_climate_dali(args)


def add_dali_args(group):
    group.add_argument("--seed", type=int, default=42)
    group.add_argument("--num_threads", type=int, default=6)
    group.add_argument("--num_workers", type=int, default=6)
    group.add_argument(
        "--device", type=str, default="gpu", choices=["cpu", "gpu"]
    )  # use gpu for GPUDirect Storage Support. needs cuda>=11.4
    group.add_argument("--device_id", type=int, default=0)
    group.add_argument("--random_shuffle", default="yes", type=lambda x: bool(strtobool(x)))
    group.add_argument("--debug_print", default="no", type=lambda x: bool(strtobool(x)))
    group.add_argument("--debug_print_each_sample", default="no", type=lambda x: bool(strtobool(x)))
    group.add_argument("--verbose", default="no", type=lambda x: bool(strtobool(x)))
    group.add_argument("--prefetch_queue_depth", type=int, default=12)
    group.add_argument("--initial_fill", type=int, default=2)

    group.add_argument("--is_amlt", default="no", type=lambda x: bool(strtobool(x)))
    group.add_argument("--read_ahead", default="yes", type=lambda x: bool(strtobool(x)))
    group.add_argument(
        "--cache_header_information",
        default="yes",
        type=lambda x: bool(strtobool(x)),
        help="If set to True, the header information for each file is cached, improving access speed.",
    )


def main(args):
    benchmark(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    add_common_args(parser.add_argument_group("common args"))
    add_climate_args(parser.add_argument_group("climate args"))
    add_dali_args(parser.add_argument_group("dali args"))

    args = parser.parse_args()
    main(args)

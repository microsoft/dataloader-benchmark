import distutils
import os

# from data.climate.era5_datapipe import ERA5Npy
from random import shuffle
from timeit import default_timer as timer

import mlflow
import numpy as np
import nvidia.dali.fn as fn

# from nvidia.dali import pipeline_def
from nvidia.dali.pipeline.experimental import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator

batch_size = 32


class ExternalPretrainInputIterator:
    def __init__(self, data_dir, variables, batch_size_ext=32):
        self.data_dir = data_dir
        self.batch_size = batch_size_ext
        self.variables = variables
        self.files = os.listdir(data_dir)
        shuffle(self.files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    def __next__(self):
        batch = []
        for _ in range(self.batch_size):
            fname = self.files[self.i]
            data = np.load(os.path.join(self.data_dir, fname))
            # todo cant have dict. perhaps multiple pipelines
            # batch.append({k: data[k] for k in self.variables})
            # test with single variable
            batch.append(np.concatenate([data[k] for k in self.variables], axis=1))
            self.i = (self.i + 1) % self.n
        return batch


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
        self.buffer = None

        self.load_next_file_into_buffer()

    # currently just loads one monthly file at a time
    def load_next_file_into_buffer(self):
        self.fname = self.files[self.file_idx]
        print("\n\n\nload_next_file_into_buffer():")
        print(self.fname)
        if self.debug_print:
            print("\n\n\nload_next_file_into_buffer():")
            print(self.fname)
        start = timer()
        data = np.load(os.path.join(self.data_dir, self.fname))  # how to use fn.readers.numpy here
        self.buffer = np.concatenate([data[k] for k in self.variables], axis=1)
        # self.buffer.append(np.concatenate([data[k] for k in self.variables], axis=1))
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


@pipeline_def
def get_iterable_pretrain_pipeline(data_dir, variables):
    print("\n\n\nget_iterable_pretrain_pipeline():")
    data = fn.external_source(source=ExternalPretrainInputIterator(data_dir, variables, batch_size_ext=batch_size))
    return data


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
    return data


@pipeline_def
def get_climate_npy_pretrain_pipeline(data_dir, device):
    print("\n\n\nget_climate_npy_pretrain_pipeline():")
    data = fn.readers.numpy(
        device=device,
        file_root=data_dir,
        file_filter="*.npy",
        # random_shuffle=random_shuffle,
        # initial_fill=initial_fill,
        # read_ahead=read_ahead,
        name="Reader",
    )

    return data


def test_iterable_pretrain_single_batch(args):
    print("\n\n\ntest_iterable_pretrain_single_batch():")
    pipe = get_iterable_pretrain_pipeline(
        data_dir=args.data_dir,
        variables=args.variables,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        device_id=args.device_id,
        # random_shuffle=args.random_shuffle,
        # initial_fill=args.initial_fill,
        # read_ahead=args.read_ahead,
        # seed=args.seed,
    )
    pipe.build()
    pipe_out = pipe.run()
    batch = [np.array(pipe_out[0][sample_idx]) for sample_idx in range(args.batch_size)]
    for sample_idx, sample in enumerate(batch):
        print(f"sample {sample_idx:05}, variable {args.variables[0]}, shape: {sample.shape}")
        # for k in args.variables:
        # print(f"sample {sample_idx:05}, variable {args.}, shape: {sample[k].shape}")


def test_parallel_callable_pretrain_single_batch(args):
    print("\n\n\ntest_parallel_callable_pretrain_single_batch():")
    pipe = get_parallel_callable_pretrain_pipeline(
        data_dir=args.data_dir,
        variables=args.variables,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        device_id=args.device_id,
        debug_print=args.debug_print,
        debug_print_each_sample=args.debug_print_each_sample,
        py_num_workers=args.py_num_workers,
        # random_shuffle=args.random_shuffle,
        # initial_fill=args.initial_fill,
        # read_ahead=args.read_ahead,
        # seed=args.seed,
    )
    pipe.build()
    pipe_out = pipe.run()
    batch = [np.array(pipe_out[0][sample_idx]) for sample_idx in range(args.batch_size)]
    for sample_idx, sample in enumerate(batch):
        print(f"sample {sample_idx:05}, variable {args.variables[0]}, shape: {sample.shape}")
        # for k in args.variables:
        # print(f"sample {sample_idx:05}, variable {args.}, shape: {sample[k].shape}")


def test_pipe_npy_single_batch(args):
    print("\n\n\nget_climate_npy_pretrain_pipeline():")
    pipe = get_climate_npy_pretrain_pipeline(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        device=args.device,
        device_id=args.device_id,
        # random_shuffle=args.random_shuffle,
        # initial_fill=args.initial_fill,
        # read_ahead=args.read_ahead,
        # seed=args.seed,
    )
    pipe.build()
    pipe_out = pipe.run()
    batch = [np.array(pipe_out[0][sample_idx]) for sample_idx in range(args.batch_size)]
    for sample_idx, sample in enumerate(batch):
        print(
            f"sample {sample_idx:05}, shape: {sample.shape}"
            # f"sample {sample_idx:05}, shape: {sample.shape}, sample.min(): {sample.min()}, sample.max(): {sample.max()}, sample.mean(): {sample.mean()}"
        )


def benchmark_iterable_pretrain_pipeline(args):
    print("\n\n\nbenchmark_iterable_pretrain_pipeline():")
    start = timer()
    last = start

    pipe = get_iterable_pretrain_pipeline(
        data_dir=args.data_dir,
        variables=args.variables,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        device_id=args.device_id,
        debug_print=args.debug_print,
        # random_shuffle=args.random_shuffle,
        # initial_fill=args.initial_fill,
        # read_ahead=args.read_ahead,
        # seed=args.seed,
    )

    dali_iter = DALIGenericIterator(pipe, ["data"], reader_name="Reader")

    for batch_idx, batch_list in enumerate(dali_iter):
        print(f"batch_idx: {batch_idx:05}")
        if batch_idx == 0:
            first = timer()
            print(f"batch_list[0]['data'].shape: {batch_list[0]['data'].shape}")
        print(f"{(timer() - last):.3f} secs for this batch")
        last = timer()

    last = timer()

    time_first_batch = first - start
    time_per_batch = (last - start) / (batch_idx + 1)
    time_per_batch_without_first = (last - first) / (batch_idx + 1)

    print(f"{time_first_batch:.3f} secs for the first batch")
    print(f"{time_per_batch:.3f} secs per batch")
    print(f"{time_per_batch_without_first:.3f} secs per batch without counting first batch")

    mlflow.log_metric(key="time_per_batch_without_first", value=time_per_batch_without_first, step=0)
    mlflow.log_metric(key="time_per_batch", value=time_per_batch, step=0)
    mlflow.log_metric(key="time_first_batch", value=time_first_batch, step=0)


def benchmark_parallel_callable_pretrain_pipeline(args):
    print("\n\n\nbenchmark_parallel_callable_pretrain_pipeline():")
    start = timer()
    last = start

    pipe = get_parallel_callable_pretrain_pipeline(
        data_dir=args.data_dir,
        variables=args.variables,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        device_id=args.device_id,
        debug_print=args.debug_print,
        debug_print_each_sample=args.debug_print_each_sample,
        py_num_workers=args.py_num_workers,
        # random_shuffle=args.random_shuffle,
        # initial_fill=args.initial_fill,
        # read_ahead=args.read_ahead,
        # seed=args.seed,
    )

    # dali_iter = DALIGenericIterator(pipe, ["data"], reader_name="Reader")
    dali_iter = DALIGenericIterator(pipe, ["data"])

    num_batches = 0
    for batch_idx, batch_list in enumerate(dali_iter):
        print(f"batch_idx: {batch_idx:05}, sample_idx: {args.batch_size*(num_batches+1):07}")
        if batch_idx == 0:
            first = timer()
            print(f"batch shape: {batch_list[0]['data'].shape}")
        print(f"batch.shape: {batch_list[0]['data'].shape}")
        print(f"{(timer() - last):.3f} secs for this batch")
        last = timer()
        num_batches += 1

    last = timer()

    time_first_batch = first - start
    time_per_batch = (last - start) / (batch_idx + 1)
    time_per_batch_without_first = (last - first) / (batch_idx + 1)

    print(f"{time_first_batch:.3f} secs for the first batch")
    print(f"{time_per_batch:.3f} secs per batch")
    print(f"{time_per_batch_without_first:.3f} secs per batch without counting first batch")

    mlflow.log_metric(key="time_per_batch_without_first", value=time_per_batch_without_first, step=0)
    mlflow.log_metric(key="time_per_batch", value=time_per_batch, step=0)
    mlflow.log_metric(key="time_first_batch", value=time_first_batch, step=0)


def get_parsed_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_threads", type=int, default=6)
    parser.add_argument("--py_num_workers", type=int, default=1)
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "gpu"]
    )  # use gpu for GPUDirect Storage Support. needs cuda>=11.4
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--img_dim", type=int, default=224)
    parser.add_argument("--random_shuffle", default="yes", type=lambda x: bool(distutils.util.strtobool(x)))
    parser.add_argument("--debug_print", default="no", type=lambda x: bool(distutils.util.strtobool(x)))
    parser.add_argument("--debug_print_each_sample", default="no", type=lambda x: bool(distutils.util.strtobool(x)))
    parser.add_argument("--img_crop", type=int, default=448)
    parser.add_argument("--prefetch_queue_depth", type=int, default=12)
    parser.add_argument("--initial_fill", type=int, default=2)
    parser.add_argument(
        "--data_dir",
        type=str,
        # default="/datadrive/weatherstorage2datasets/1.40625deg_monthly_np/val",
        # default="/datadrive/localdatasets/climate/1.40625deg_monthly_npy/val/pretrain/",
        default="/datadrive/localdatasets/climate/1.40625deg_monthly_np/val",
    )
    parser.add_argument("--is_amlt", default="no", type=lambda x: bool(distutils.util.strtobool(x)))
    parser.add_argument("--read_ahead", default="yes", type=lambda x: bool(distutils.util.strtobool(x)))
    parser.add_argument(
        "--cache_header_information",
        default="yes",
        type=lambda x: bool(distutils.util.strtobool(x)),
        help="If set to True, the header information for each file is cached, improving access speed.",
    )
    parser.add_argument(
        "--variables",
        default=["z", "r", "u", "v", "t", "t2m", "u10", "v10"],
        help="list of ___ (strings)",
        nargs="+",
        type=str,
        choices=["z", "r", "u", "v", "t", "t2m", "u10", "v10"],
    )

    args = parser.parse_args()
    return args


def main():
    args = get_parsed_args()
    # test_iterable_pretrain_single_batch(args)
    # benchmark_iterable_pretrain_pipeline(args)

    # test_pipe_npy_single_batch(args)

    # test_parallel_callable_pretrain_single_batch(args)
    benchmark_parallel_callable_pretrain_pipeline(args)

    print("\n\n\n")


if __name__ == "__main__":
    main()

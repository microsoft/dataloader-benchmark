import argparse
import os
from distutils.util import strtobool
from random import shuffle

import numpy as np
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from src.benchmarks.benchmarker import Benchmarker
from src.benchmarks.climate.climate_opts import add_climate_args
from src.benchmarks.common_opts import add_common_args

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
            batch.append(np.concatenate([data[k] for k in self.variables], axis=1))
            self.i = (self.i + 1) % self.n
        return batch


# todo pass cmd line args to decorater with args
@pipeline_def(batch_size=batch_size, num_threads=6, device_id=0)
def get_iterable_pretrain_pipeline(data_dir, variables):
    print("\n\n\nget_iterable_pretrain_pipeline():")
    data = fn.external_source(source=ExternalPretrainInputIterator(data_dir, variables, batch_size_ext=batch_size))
    data = data.gpu()
    return data


def get_dataloader(args):
    pipe = get_iterable_pretrain_pipeline(
        data_dir=args.data_dir,
        variables=args.variables,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        device_id=args.device_id,
    )

    dataloader = DALIGenericIterator(pipe, ["data"])
    return dataloader


def benchmark(args):
    dataloader = get_dataloader(args)
    benchmarker = Benchmarker(verbose=args.verbose, dataset="climate", library="dali_npy_iterable")
    benchmarker.set_dataloader(dataloader)
    benchmarker.benchmark_climate(args)


def add_dali_args(group):
    group.add_argument("--seed", type=int, default=42)
    group.add_argument("--num_threads", type=int, default=6)
    group.add_argument("--py_num_workers", type=int, default=1)
    group.add_argument(
        "--device", type=str, default="gpu", choices=["cpu", "gpu"]
    )  # use gpu for GPUDirect Storage Support. needs cuda>=11.4
    group.add_argument("--device_id", type=int, default=0)
    group.add_argument("--random_shuffle", default="yes", type=lambda x: bool(strtobool(x)))
    group.add_argument("--debug_print", default="no", type=lambda x: bool(strtobool(x)))
    group.add_argument("--debug_print_each_sample", default="no", type=lambda x: bool(strtobool(x)))
    group.add_argument("--prefetch_queue_depth", type=int, default=1)
    group.add_argument("--initial_fill", type=int, default=2)

    group.add_argument("--is_amlt", default="no", type=lambda x: bool(strtobool(x)))
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

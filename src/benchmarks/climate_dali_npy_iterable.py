import distutils
import os

# from data.climate.era5_datapipe import ERA5Npy
from random import shuffle

import numpy as np
import nvidia.dali.fn as fn
from benchmarker import Benchmarker
from nvidia.dali import pipeline_def
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
    benchmarker = Benchmarker(verbose=args.verbose, dataset=f"climate_{args.use}", library="dali_npy_iterable")
    benchmarker.set_dataloader(dataloader)
    benchmarker.benchmark_climate(args)


def get_parsed_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--benchmark_results_file", default="benchmark_results_climate.csv", type=str)
    parser.add_argument("--verbose", default="no", type=lambda x: bool(distutils.util.strtobool(x)))
    parser.add_argument("--verbose", default="no", type=lambda x: bool(distutils.util.strtobool(x)))
    parser.add_argument("--use", type=str, default="pretrain", help="Use forecast or pretrain")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_threads", type=int, default=6)
    parser.add_argument("--py_num_workers", type=int, default=1)
    parser.add_argument(
        "--device", type=str, default="gpu", choices=["cpu", "gpu"]
    )  # use gpu for GPUDirect Storage Support. needs cuda>=11.4
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--img_dim", type=int, default=224)
    parser.add_argument("--random_shuffle", default="yes", type=lambda x: bool(distutils.util.strtobool(x)))
    parser.add_argument("--debug_print", default="no", type=lambda x: bool(distutils.util.strtobool(x)))
    parser.add_argument("--debug_print_each_sample", default="no", type=lambda x: bool(distutils.util.strtobool(x)))
    parser.add_argument("--img_crop", type=int, default=448)
    parser.add_argument("--prefetch_queue_depth", type=int, default=1)
    parser.add_argument("--initial_fill", type=int, default=2)
    parser.add_argument(
        "--data_dir",
        type=str,
        # default="/datadrive/weatherstorage2datasets/1.40625deg_monthly_np/val",
        # default="/datadrive/localdatasets/climate/1.40625deg_monthly_npy/val/pretrain/",
        default="/datadrive/localdatasets/climate/1.40625deg_monthly_np/val",
    )
    parser.add_argument("--is_amlt", default="no", type=lambda x: bool(distutils.util.strtobool(x)))
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


def main(args):
    benchmark(args)


if __name__ == "__main__":
    args = get_parsed_args()
    main(args)

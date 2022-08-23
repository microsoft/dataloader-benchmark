import argparse
from distutils.util import strtobool

import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def

from src.benchmarks.benchmarker import Benchmarker
from src.benchmarks.climate.climate_opts import add_climate_args
from src.benchmarks.common_opts import add_common_args

batch_size = 32


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


def benchmark(args):
    dataloader = get_dataloader(args)
    benchmarker = Benchmarker(verbose=args.verbose, dataset="climate", library="dali_npy")
    benchmarker.set_dataloader(dataloader)
    benchmarker.benchmark_climate(args)


def add_dali_args(group):
    group.add_argument("--seed", type=int, default=42)
    group.add_argument("--num_threads", type=int, default=6)
    group.add_argument("--py_num_workers", type=int, default=1)
    group.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "gpu"]
    )  # use gpu for GPUDirect Storage Support. needs cuda>=11.4
    group.add_argument("--device_id", type=int, default=0)
    group.add_argument("--random_shuffle", default="yes", type=lambda x: bool(strtobool(x)))
    group.add_argument("--debug_print", default="no", type=lambda x: bool(strtobool(x)))
    group.add_argument("--debug_print_each_sample", default="no", type=lambda x: bool(strtobool(x)))
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

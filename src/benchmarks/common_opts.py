import argparse
from distutils.util import strtobool


def get_common_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", default=6, type=int, help="number of cpu cores")
    parser.add_argument("--benchmark_results_file", default="benchmark_results_tartanair.csv", type=str)
    parser.add_argument("--verbose", default="no", type=lambda x: bool(strtobool(x)))

    args = parser.parse_args()
    return args

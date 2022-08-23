from distutils.util import strtobool


def add_common_args(group):
    group.add_argument("--batch_size", type=int, default=32)
    group.add_argument("--num_workers", default=6, type=int, help="number of cpu cores")
    group.add_argument("--benchmark_results_file", default="benchmark_results_tartanair.csv", type=str)
    group.add_argument("--verbose", default="no", type=lambda x: bool(strtobool(x)))


def add_ffcv_args(group):
    group.add_argument("--beton_file", type=str, help="path to beton file")
    group.add_argument("--order", type=str, default="random", help="Ordering of data: random or quasi_random")
    group.add_argument("--os_cache", type=lambda x: bool(strtobool(x)))

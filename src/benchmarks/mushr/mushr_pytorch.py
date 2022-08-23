import argparse

from torch.utils.data import DataLoader

from src.benchmarks.benchmarker import Benchmarker
from src.benchmarks.common_opts import add_common_args
from src.benchmarks.mushr.mushr_opts import add_mushr_args
from src.data.mushr.dataset_basic import MushrVideoDatasetPreload


def get_dataloader(args):
    pytorch_shuffle = args.order != "sequential"

    dataset = MushrVideoDatasetPreload(
        dataset_dir=args.mush_dir,
        ann_file_name=args.mushr_ann_file,
        transform=None,
        gt_map_file_name=args.mushr_gt_map_file_name,
        local_map_size_m=12,
        map_center=[-32.925, -37.3],
        map_res=0.05,
        state_type="pcl",
        clip_len=1,
        flatten_img=False,
        load_gt_map=False,
        rebalance_samples=False,
        num_bins=5,
        map_recon_dim=64,
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=pytorch_shuffle, num_workers=12)

    return dataloader


def benchmark(args):
    dataloader = get_dataloader(args)
    benchmarker = Benchmarker(verbose=args.verbose, dataset="mushr", library="pytorch")
    benchmarker.set_dataloader(dataloader)
    benchmarker.benchmark_mushr(args)


def main(args):
    benchmark(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    add_common_args(parser.add_argument_group(title="common args"))
    add_mushr_args(parser.add_argument_group(title="mushr args"))

    args = parser.parse_args()

    main(args)

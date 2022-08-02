import argparse
import distutils

from benchmarker import Benchmarker
from torch.utils.data import DataLoader

from src.data.mushr.dataset import MushrVideoDatasetPreload


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

    benchmarker = Benchmarker()
    benchmarker.set_dataloader(dataloader)
    benchmarker.benchmark_mushr(args)


def main(args):
    benchmark(args)


def get_parsed_args():
    parser = argparse.ArgumentParser(description="FFCV options")
    parser.add_argument("--benchmark_results_file", default="benchmark_results_mushr.csv", type=str)
    parser.add_argument("--dataset", type=str, default="mushr", help="dataset type to use for benchmarking")
    parser.add_argument(
        "--beton_file", type=str, default="./tartan_abandonedfactory_ratnesh.beton", help="path to beton file"
    )
    parser.add_argument("--mushr_ann_file", type=str, default="./train_ann_pose.json", help="")
    parser.add_argument("--mushr_gt_map_file_name", type=str, default="bravern_floor.pgm", help="")
    parser.add_argument("--mushr_dir", type=str, default="./pretraining_data/hackathon_data_2p5_nonoise3", help="")
    parser.add_argument("--order", type=str, default="random", help="Ordering of data: random or quasi_random")
    parser.add_argument("--os_cache", type=lambda x: bool(distutils.util.strtobool(x)))
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_parsed_args()
    main(args)

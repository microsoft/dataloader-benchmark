import argparse
import distutils
import time

import mlflow
import torch
from ffcv.fields.decoders import NDArrayDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.compiler import Compiler
from ffcv.transforms import ToTensor

from src.data.mushr.dataset import MushrVideoDatasetPreload
from src.data.mushr.dataset_disk import MushrVideoDataset
from src.data.tartanair.build import TartanAirNoTransform, TartanAirVideoDataset
from src.utils.utils import AverageMeter

Compiler.set_enabled(True)


def parse_args():
    parser = argparse.ArgumentParser(description="FFCV options")
    parser.add_argument("--dataset", type=str, default="tartanair", help="dataset type to use for benchmarking")
    parser.add_argument(
        "--beton_file", type=str, default="./tartan_abandonedfactory_ratnesh.beton", help="path to beton file"
    )
    parser.add_argument("--tartanair_ann_file", type=str, default="./train_ann_debug_ratnesh.json", help="")
    parser.add_argument("--mushr_ann_file", type=str, default="./train_ann_pose.json", help="")
    parser.add_argument("--mushr_gt_map_file_name", type=str, default="bravern_floor.pgm", help="")
    parser.add_argument("--mushr_dir", type=str, default="./pretraining_data/hackathon_data_2p5_nonoise3", help="")
    parser.add_argument("--order", type=str, default="random", help="Ordering of data: random or quasi_random")
    parser.add_argument("--os_cache", type=lambda x: bool(distutils.util.strtobool(x)))
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers")

    from pprint import pprint

    args = parser.parse_args()
    pprint(args)

    return args


def get_order_option_ffcv(order):
    if order == "random":
        order_option = OrderOption.RANDOM
    elif order == "quasi_random":
        order_option = OrderOption.QUASI_RANDOM
    elif order == "sequential":
        order_option = OrderOption.SEQUENTIAL
    else:
        raise ValueError(f"Unknown order option: {order}")
    return order_option


def benchmark_mushr_ffcv(args):
    time_start = time.time()

    # Dataset specific
    PIPELINES = {
        "states": [NDArrayDecoder(), ToTensor()],
        "actions": [NDArrayDecoder(), ToTensor()],
        "poses": [NDArrayDecoder(), ToTensor()],
    }

    loader = Loader(
        fname=args.beton_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        order=get_order_option_ffcv(args.order),
        os_cache=args.os_cache,
        pipelines=PIPELINES,
    )

    batch_time = AverageMeter()

    ts = time.time()
    for data in loader:
        batch_time.update(time.time() - ts)
        ts = time.time()

    print(f"Time per batch: {batch_time.avg:.3f}")
    print(f"Total time: {time.time() - time_start:.3f}")


def benchmark_tartanair_ffcv(args):
    print("==== TartanAir FFCV ====")
    time_start = time.time()

    # Dataset specific
    PIPELINES = {
        "rgb": [SimpleRGBImageDecoder(), ToTensor()],
        "depth": [NDArrayDecoder(), ToTensor()],
        "flow": [NDArrayDecoder(), ToTensor()],
    }

    loader = Loader(
        fname=args.beton_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        order=get_order_option_ffcv(args.order),
        os_cache=args.os_cache,
        pipelines=PIPELINES,
    )

    batch_time = AverageMeter()

    ts = time.time()
    for data in loader:
        batch_time.update(time.time() - ts)
        ts = time.time()

    print(f"Time per batch: {batch_time.avg:.3f}")
    print(f"Total time: {time.time() - time_start:.3f}")
    mlflow.log_metric(key="ffcv/batch_time", value=batch_time.avg, step=0)
    mlflow.log_metric(key="ffcv/total_time", value=time.time() - time_start, step=0)


def benchmark_tartanair_pytorch(args):
    print("==== TartanAir Disk ====")
    time_start = time.time()

    dataset = TartanAirVideoDataset(
        ann_file=args.tartanair_ann_file,
        clip_len=1,
        seq_len=1,
        modalities=["image_left", "depth_left", "flow_flow"],
        transform=TartanAirNoTransform(),
        video_name_keyword=None,
        ffcv=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    batch_time = AverageMeter()

    ts = time.time()
    for idx, batch in enumerate(dataloader):
        batch_time.update(time.time() - ts)
        ts = time.time()

    print(f"Time per batch: {batch_time.avg:.3f}")
    print(f"Total time: {time.time() - time_start:.3f}")
    mlflow.log_metric(key="vanilla/batch_time", value=batch_time.avg, step=0)
    mlflow.log_metric(key="vanilla/total_time", value=time.time() - time_start, step=0)


def benchmark_mushr_pytorch(args):
    pytorch_shuffle = args.order != "sequential"

    print("====MuSHR preload====")
    time_start = time.time()

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

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=pytorch_shuffle, num_workers=12)

    batch_time = AverageMeter()

    ts = time.time()
    for idx, batch in enumerate(dataloader):
        batch_time.update(time.time() - ts)
        ts = time.time()

    print(f"Time per batch: {batch_time.avg:.3f}")
    print(f"Total time: {time.time() - time_start:.3f}")


def benchmark_mushr_disk(args):
    pytorch_shuffle = args.order != "sequential"

    time_start = time.time()

    dataset = MushrVideoDataset(
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

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=pytorch_shuffle, num_workers=12)

    batch_time = AverageMeter()

    ts = time.time()
    for idx, batch in enumerate(dataloader):
        batch_time.update(time.time() - ts)
        ts = time.time()

    print(f"Time per batch: {batch_time.avg:.3f}")
    print(f"Total time: {time.time() - time_start:.3f}")


def main(args):
    if args.dataset == "mushr":
        benchmark_mushr_ffcv(args)
        benchmark_mushr_pytorch(args)

    elif args.dataset == "tartanair":
        benchmark_tartanair_ffcv(args)
        benchmark_tartanair_pytorch(args)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

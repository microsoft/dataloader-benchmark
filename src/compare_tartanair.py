from ffcv.fields.decoders import (FloatDecoder, NDArrayDecoder,
                                  SimpleRGBImageDecoder)
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.compiler import Compiler
from ffcv.transforms import ToTensor

Compiler.set_enabled(False)
import argparse
import distutils
import time

import torch

from tartanair.build import TartanAirVideoDataset
from utils import AverageMeter

parser = argparse.ArgumentParser(description="FFCV options")
parser.add_argument("--order", type=str, default="quasi_random", help="Ordering of data: random or quasi_random")
parser.add_argument("--os_cache", type=lambda x: bool(distutils.util.strtobool(x)), default="False")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--num_workers", type=int, default=12, help="Number of workers")

args = parser.parse_args()

if args.order == "random":
    order_option = OrderOption.RANDOM
elif args.order == "quasi_random":
    order_option = OrderOption.QUASI_RANDOM
elif args.order == "sequential":
    order_option = OrderOption.SEQUENTIAL
else:
    raise ValueError(f"Unknown order option: {args.order}")

print(f"====FFCV os_cache={str(args.os_cache)}====")
time_start = time.time()

# Dataset specific
PIPELINES = {
    "rgb": [SimpleRGBImageDecoder(), ToTensor()],
    "depth": [NDArrayDecoder(), ToTensor()],
    "flow": [NDArrayDecoder(), ToTensor()],
}

loader = Loader(
    "./tartan_abandonedfactory.beton",
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    order=order_option,
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

dataset = TartanAirVideoDataset(
    ann_file="/home/saihv/datasets/tartanair-release1/train_ann_abandonedfactory.json",
    clip_len=1,
    seq_len=1,
    data_types=["image_left", "depth_left", "flow_flow"],
    transform=None,
    video_name_keyword=None,
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=12)
batch_time = AverageMeter()

ts = time.time()
for idx, batch in enumerate(dataloader):
    batch_time.update(time.time() - ts)
    ts = time.time()

print(f"Time per batch: {batch_time.avg:.3f}")
print(f"Total time: {time.time() - time_start:.3f}")

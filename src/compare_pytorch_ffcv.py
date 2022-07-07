from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder
from ffcv.transforms import ToTensor
from ffcv.pipeline.compiler import Compiler
Compiler.set_enabled(False)
import tqdm 
import time
import torch
from mushr.dataset import MushrVideoDatasetPreload
from mushr.dataset_disk import MushrVideoDataset
import argparse
import distutils

parser = argparse.ArgumentParser(description='FFCV options')
parser.add_argument('--order', type=str, default='random', help='Ordering of data: random or quasi_random')
parser.add_argument('--os_cache', type=lambda x:bool(distutils.util.strtobool(x)))
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_workers', type=int, default=12, help='Number of workers')

args = parser.parse_args()

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if args.order == 'random':
    order_option = OrderOption.RANDOM
elif args.order == 'quasi_random':
    order_option = OrderOption.QUASI_RANDOM
elif args.order == 'sequential':
    order_option = OrderOption.SEQUENTIAL
else:
    raise ValueError('Unknown order option: {}'.format(args.order))

print("====FFCV os_cache=False====")
time_start = time.time()

# Dataset specific
PIPELINES = {'states': [NDArrayDecoder(), ToTensor()],
            'actions': [NDArrayDecoder(), ToTensor()],
            'poses': [NDArrayDecoder(), ToTensor()]}

loader = Loader('./mushr_train.beton',
                batch_size = args.batch_size,
                num_workers = args.num_workers,
                order = order_option,
                os_cache = args.os_cache,
                pipelines = PIPELINES)

batch_time = AverageMeter()

ts = time.time()
for data in loader:
    batch_time.update(time.time() - ts)
    ts = time.time()

print('Time per batch: {:.3f}'.format(batch_time.avg))
print('Total time: {:.3f}'.format(time.time() - time_start))

print("====FFCV os_cache=True====")

loader = Loader('./mushr_train.beton',
                batch_size = args.batch_size,
                num_workers = args.num_workers,
                order = order_option,
                os_cache = args.os_cache,
                pipelines = PIPELINES)

batch_time = AverageMeter()

ts = time.time()
for data in loader:
    batch_time.update(time.time() - ts)
    ts = time.time()

print('Time per batch: {:.3f}'.format(batch_time.avg))
print('Total time: {:.3f}'.format(time.time() - time_start))

pytorch_shuffle = (args.order != 'sequential')

print("====MuSHR preload====")
time_start = time.time()

dataset = MushrVideoDatasetPreload(
        dataset_dir="/home/saihv/pretraining_data/hackathon_data_2p5_nonoise3",
        ann_file_name="singlefile_train_ann_pose.json",
        transform=None,
        gt_map_file_name="bravern_floor.pgm",
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

print('Time per batch: {:.3f}'.format(batch_time.avg))
print('Total time: {:.3f}'.format(time.time() - time_start))

print("====MuSHR disk====")
time_start = time.time()

dataset = MushrVideoDataset(
        dataset_dir="/home/saihv/pretraining_data/hackathon_data_2p5_nonoise3",
        ann_file_name="train_ann_pose.json",
        transform=None,
        gt_map_file_name="bravern_floor.pgm",
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

print('Time per batch: {:.3f}'.format(batch_time.avg))
print('Total time: {:.3f}'.format(time.time() - time_start))
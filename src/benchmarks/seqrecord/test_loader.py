"""check if data read from seqrecord is same as data read by original tartanair dataloader
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchdata.datapipes as dp
from src.benchmarks.seqrecord.dataload_utils import InputConfig
from src.benchmarks.seqrecord.seqrecord import SeqRecord
from src.benchmarks.seqrecord.tartanloader import IterTartanAIRDatapipe, list2array
from src.data.tartanair.build import TartanAirVideoTransform
from src.data.tartanair.tartanair_video import TartanAirVideoDataset

# load original tartanair dataset
modalities = ["image_left", "depth_left", "flow_flow"]
json_filepath = "/datadrive/azure_mounted_data/commondataset2/tartanair-release1/train_ann_abandonedfactory_easy.json"
transform = TartanAirVideoTransform()
dataset = TartanAirVideoDataset(
    json_filepath,
    clip_len=1,
    seq_len=16,
    modalities=modalities,
    transform=transform,
    video_name_keyword=None,
)

# load iter style datapip
rootdir = "/datadrive/azure_mounted_data/commondataset2/tartanair-release1/abandonedfactory/records"
record = SeqRecord.load_record_from_dict(rootdir)
record.rootdir = rootdir
segment_len = 16

# configs from input modalities
config_path = "/home/azureuser/AutonomousSystemsResearch/dataloader-benchmark/src/benchmarks/seqrecord/config.yaml"
with open(config_path, mode="r") as f:
    import yaml

    config = yaml.safe_load(f)["inputs"]

input_config = InputConfig()
for key, modal in config.items():
    modal["kwargs"]["name"] = key
    input_config.add_input(modal)
datapipe = IterTartanAIRDatapipe(record, segment_len, None)
datapipe = dp.iter.Mapper(datapipe, fn=list2array)
datapipe = dp.iter.Mapper(datapipe, fn=input_config.train_transform)

# mannually checking a few
item_idx = 0
old_item = dataset[item_idx]
new_item = next(itertools.islice(datapipe, item_idx, None))

frame_idx = 0
fig, axs = plt.subplots(nrows=2, ncols=len(modalities), squeeze=False)
for i, modality in enumerate(modalities):
    old_frame = old_item[modality][0].permute(1, 0, 2, 3)[frame_idx]
    new_frame = new_item[modality][frame_idx]
    print(f"{modality=}")
    print(old_frame.shape)
    print(new_frame.shape)
    if old_frame.size(0) < 3:
        old_frame = old_frame[0]  # only show the first channel
        axs[0, i].imshow(np.asarray(old_frame))
    else:
        axs[0, i].imshow(np.asarray(old_frame.permute(1, 2, 0)))
    if new_frame.dim() == 2:
        axs[1, i].imshow(np.asarray(new_frame))
    elif new_frame.size(0) < 3:
        new_frame = new_frame[0]  # only show the first channel
        axs[1, i].imshow(np.asarray(new_frame))
    else:
        axs[1, i].imshow(np.asarray(new_frame.permute(1, 2, 0)))
plt.savefig(f"test_dataload_{item_idx}.png")

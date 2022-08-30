"""check if data read from seqrecord is same as data read by original tartanair dataloader
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchdata.datapipes as dp
from src.benchmarks.seqrecord.dataload_utils import InputConfig
from src.benchmarks.seqrecord.seqrecord import SeqRecord
from src.benchmarks.seqrecord.tartanloader import (IterTartanAIRDatapipe,
                                                   list2array)
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
# mannually checking a few (transform)
# check size, range, dtype and value of rgb image
item_idx = 5
old_item = dataset[item_idx]
new_item = next(itertools.islice(datapipe, item_idx, None))

frame_idx = 5
fig, axs = plt.subplots(nrows=2, ncols=len(modalities), squeeze=False)
# rgb
modality = 'image_left'
old_rgb = old_item[modality][0].permute(1, 0, 2, 3)[frame_idx]
new_rgb = new_item[modality][frame_idx]
print(old_rgb.dtype, old_rgb.size(), old_rgb.min(), old_rgb.max())
print(new_rgb.dtype, new_rgb.size(), new_rgb.min(), new_rgb.max())
old_rgb = old_rgb.permute(1, 2, 0)
new_rgb = new_rgb.permute(1, 2, 0)
axs[0, 0].imshow(np.asarray(old_rgb))
axs[1, 0].imshow(np.asarray(new_rgb))
torch.testing.assert_allclose(old_rgb, new_rgb)
modality = 'depth_left'
old_depth = old_item[modality][0].permute(1, 0, 2, 3)[frame_idx]
new_depth = new_item[modality][frame_idx]
old_depth = old_depth[0]
print(old_depth.dtype, old_depth.size(), old_depth.min(), old_depth.max())
print(new_depth.dtype, new_depth.size(), new_depth.min(), new_depth.max())
axs[0, 1].imshow(np.asarray(old_depth))
axs[1, 1].imshow(np.asarray(new_depth))
modality = 'flow_flow'
old_flow = old_item[modality][0].permute(1, 0, 2, 3)[frame_idx]
new_flow = new_item[modality][frame_idx]
old_flow = old_flow[0]
new_flow = new_flow[0, :, :]
print(old_flow.dtype, old_flow.size(), old_flow.min(), old_flow.max())
print(new_flow.dtype, new_flow.size(), new_flow.min(), new_flow.max())

axs[0, 2].imshow(np.asarray(old_flow))
axs[1, 2].imshow(np.asarray(new_flow))
torch.testing.assert_allclose(old_depth, new_depth)
torch.testing.assert_allclose(old_flow, new_flow)

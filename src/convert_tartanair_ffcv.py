from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, RGBImageField
import numpy as np
from tartanair.build import TartanAirVideoDataset

dataset = TartanAirVideoDataset(
            ann_file = "/home/saihv/datasets/tartanair-release1/train_ann_abandonedfactory.json",
            clip_len=1,
            seq_len=1,
            data_types=["image_left", "depth_left", "flow_flow"],
            transform=None,
            video_name_keyword=None,
        )

rgb, depth, flow = dataset[0]

writer = DatasetWriter("./tartan_abandonedfactory_jpg.beton", 
            {'rgb': RGBImageField(write_mode='raw'),
            'depth': NDArrayField(shape=(480, 640), dtype=np.dtype('float32')), 
            'flow': NDArrayField(shape=(480, 640, 2), dtype=np.dtype('float32'))}, num_workers=1)

writer.from_indexed_dataset(dataset)

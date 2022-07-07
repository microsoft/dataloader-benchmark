from dataset_basic import MushrVideoDatasetPreload
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField
import numpy as np

dataset = MushrVideoDatasetPreload(
            dataset_dir="/home/saihv/pretraining_data/hackathon_data_2p5_nonoise3",
            ann_file_name="singlefile_train_ann_pose_debug.json",
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

state, action, pose = dataset[0]

print(state.shape)
print(action.shape)
print(pose.shape)

writer = DatasetWriter("./mushr_train_debug.beton", 
            {'states': NDArrayField(shape=(1, 720, 2), dtype=np.dtype('float32')), 
            'actions': NDArrayField(shape=(1, 1), dtype=np.dtype('float32')), 
            'poses': NDArrayField(shape=(1, 3), dtype=np.dtype('float32'))})

writer.from_indexed_dataset(dataset)
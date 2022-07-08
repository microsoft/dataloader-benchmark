from argparse import ArgumentParser

import numpy as np
from ffcv.fields import NDArrayField, RGBImageField
from ffcv.writer import DatasetWriter

from mushr.dataset_basic import MushrVideoDatasetPreload
from tartanair.build import TartanAirVideoDataset


def convert_mushr(dataset_dir, ann_file_name, gt_map_file_name, output_beton_file):
    dataset = MushrVideoDatasetPreload(
        dataset_dir=dataset_dir,
        ann_file_name=ann_file_name,
        transform=None,
        gt_map_file_name=gt_map_file_name,
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

    writer = DatasetWriter(
        output_beton_file,
        {
            "states": NDArrayField(shape=(1, 720, 2), dtype=np.dtype("float32")),
            "actions": NDArrayField(shape=(1, 1), dtype=np.dtype("float32")),
            "poses": NDArrayField(shape=(1, 3), dtype=np.dtype("float32")),
        },
    )

    writer.from_indexed_dataset(dataset)


def convert_tartanair(
    ann_file,
    output_beton_file,
):
    dataset = TartanAirVideoDataset(
        ann_file=ann_file,
        clip_len=1,
        seq_len=1,
        modalities=["image_left", "depth_left", "flow_flow"],
        transform=None,
        video_name_keyword=None,
        ffcv=False,
        return_mask_position=False,
    )

    rgb, depth, flow = dataset[0]

    writer = DatasetWriter(
        output_beton_file,
        {
            "rgb": RGBImageField(write_mode="raw"),
            "depth": NDArrayField(shape=(480, 640), dtype=np.dtype("float32")),
            "flow": NDArrayField(shape=(480, 640, 2), dtype=np.dtype("float32")),
        },
        num_workers=1,
    )

    writer.from_indexed_dataset(dataset)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--tartanair_ann", type=str, default="/home/saihv/datasets/tartanair-release1/train_ann_abandonedfactory.json"
    )
    parser.add_argument("--tartanair_output_beton_file", type=str, default="./tartan_abandonedfactory_jpg.beton")

    parser.add_argument(
        "--mushr_dataset_dir", type=str, default="/home/saihv/pretraining_data/hackathon_data_2p5_nonoise3"
    )
    parser.add_argument("--mushr_ann_file_name", type=str, default="singlefile_train_ann_pose_debug.json")
    parser.add_argument("--mushr_gt_map_file_name", type=str, default="bravern_floor.pgm")
    parser.add_argument("--mushr_output_beton_file", type=str, default="./mushr_train_debug.beton")


def main(args):
    convert_tartanair(ann_file=args.tartanair_ann, output_beton_file=args.output_beton_file)

    convert_mushr(
        dataset_dir=args.mushr_dataset_dir,
        ann_file_name=args.mushr_ann_file_name,
        gt_map_file_name=args.mushr_gt_map_file_name,
        output_beton_file=args.output_beton_file,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)

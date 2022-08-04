import argparse


def get_tartanair_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modalities",
        default=["image_left", "depth_left", "flow_flow"],
        help="list of modalities (strings)",
        nargs="+",
        type=str,
        choices=[
            "image_left",
            "image_right",
            "depth_left",
            "depth_right",
            "flow_mask",
            "flow_flow",
            "seg_left",
            "seg_right",
        ],
    )
    parser.add_argument(
        "--train_ann_file",
        default="/datadrive/localdatasets/tartanair-release1/train_ann_debug_ratnesh.json",
        type=str,
    )
    parser.add_argument(
        "--train_transform",
        default="TartanAirNoTransform",
        type=str,
        choices=["TartanAirNoTransform", "TartanAirVideoTransform", "TartanAirVideoTransformWithAugmentation"],
    )
    parser.add_argument("--use_val", default=False, type=bool)
    parser.add_argument("--seq_len", default=16, type=int, help="number of frames in each video block")
    parser.add_argument("--num_seq", default=1, type=int, help="number of video blocks")

    parser.add_argument("--video_name_keyword", default=None)
    parser.add_argument("--pin_memory", default=True, type=bool)
    parser.add_argument("--img_dim", default=224, type=int)
    parser.add_argument("--img_crop", default=448, type=int)
    parser.add_argument("--flip", default=False, type=bool)

    args = parser.parse_args()
    return args

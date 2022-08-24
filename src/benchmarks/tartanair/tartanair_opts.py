def add_tartanair_args(group):
    group.add_argument(
        "--modalities",
        default=["image_left", "depth_left", "seg_left", "flow_flow", "flow_mask"],
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
    group.add_argument(
        "--train_ann_file",
        default="/datadrive/localdatasets/tartanair-release1/train_ann_debug_ratnesh.json",
        type=str,
    )
    group.add_argument(
        "--train_transform",
        default="TartanAirNoTransform",
        type=str,
        choices=["TartanAirNoTransform", "TartanAirVideoTransform", "TartanAirVideoTransformWithAugmentation"],
    )
    group.add_argument("--use_val", default=False, type=bool)
    group.add_argument("--seq_len", default=16, type=int, help="number of frames in each video block")
    group.add_argument("--num_seq", default=1, type=int, help="number of video blocks")

    group.add_argument("--video_name_keyword", default=None)
    group.add_argument("--pin_memory", default=True, type=bool)
    group.add_argument("--img_dim", default=224, type=int)
    group.add_argument("--img_crop", default=448, type=int)
    group.add_argument("--flip", default=False, type=bool)

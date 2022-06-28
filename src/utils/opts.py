import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--work_dir", type=str, default="work_dirs/pretrain/rgb2rgb")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--backbone", type=str, default="vit")
    parser.add_argument("--local_rank", type=int, default=0)

    # ================================== added arguments for TaratanAIR ==============
    parser.add_argument("--img", default=False, type=bool)
    parser.add_argument("--dataloader_mode", default="all", type=str)
    parser.add_argument("--visual_aug", default=False, type=bool)
    parser.add_argument("--flip", default=False, type=bool)
    parser.add_argument(
        "--modalities",
        default=["image_left"],
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

    parser.add_argument("--use_memory", default=False, help="use multimodal fusion memory")
    parser.add_argument("--use_external", default=False, type=bool, help="use memory as external info")
    parser.add_argument("--use_pred", default=False, type=bool, help="use predictive contrastive loss.")
    parser.add_argument("--use_inst", default=False, type=bool, help="use instance contrastive loss.")
    parser.add_argument("--use_img2img", default=False, type=bool, help="compute img2img loss")
    parser.add_argument(
        "--use_flow_both",
        default=False,
        type=bool,
        help="training rgb->flow both pred and instance.",
    )
    parser.add_argument(
        "--flow_inst_only",
        default=False,
        type=bool,
        help="only use instance loss for flow.",
    )
    # parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument("--use_val", default=False, type=bool)
    # parser.add_argument('--data_path', default= '../Data/tartanair-release1/')
    parser.add_argument(
        "--train_ann_file",
        default="/datadrive/commondatasets/tartanair-release1/train_ann_debug.json",
        type=str,
    )
    parser.add_argument("--val_ann_file", default=" ", type=str)
    parser.add_argument("--benchmark_results_file", default="benchmark_results.csv", type=str)
    parser.add_argument("--train_transform", default="", type=str)
    parser.add_argument(
        "--train_transform_aug",
        default="TartanAirVideoTransformWithAugmentation",
        type=str,
    )
    parser.add_argument("--val_transform", default=False, type=bool)
    parser.add_argument("--video_name_keyword", default=None)
    parser.add_argument("--pin_memory", default=True, type=bool)
    parser.add_argument("--img_dim", default=224, type=int)
    parser.add_argument("--img_crop", default=448, type=int)
    parser.add_argument("--seq_len", default=1, type=int, help="number of frames in each video block")
    parser.add_argument("--num_seq", default=16, type=int, help="number of video blocks")
    parser.add_argument("-j", "--workers", default=0, type=int)

    args = parser.parse_args()
    return args

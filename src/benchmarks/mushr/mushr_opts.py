def add_mushr_args(group):
    group.add_argument("--mushr_ann_file", type=str, default="./train_ann_pose.json", help="")
    group.add_argument("--mushr_gt_map_file_name", type=str, default="bravern_floor.pgm", help="")
    group.add_argument("--mushr_dir", type=str, default="./pretraining_data/hackathon_data_2p5_nonoise3", help="")
    group.add_argument("--order", type=str, default="random", help="Ordering of data: random or quasi_random")

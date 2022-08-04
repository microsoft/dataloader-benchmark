import argparse


def get_mushr_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--mushr_ann_file", type=str, default="./train_ann_pose.json", help="")
    parser.add_argument("--mushr_gt_map_file_name", type=str, default="bravern_floor.pgm", help="")
    parser.add_argument("--mushr_dir", type=str, default="./pretraining_data/hackathon_data_2p5_nonoise3", help="")
    parser.add_argument("--order", type=str, default="random", help="Ordering of data: random or quasi_random")

    args = parser.parse_args()
    return args

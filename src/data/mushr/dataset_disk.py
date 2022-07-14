import json
import math
import os

import cv2
import numpy as np
import torch
from src.data.mushr.dataset_utils import norm_angle, process_relative_poses_deltas_incremental
from PIL import Image
from skimage.transform import resize


class MushrVideoDataset(torch.utils.data.Dataset):
    """Imitation learning (IL) video dataset in mushr lidar."""

    def __init__(
        self,
        dataset_dir,
        ann_file_name,
        transform,
        gt_map_file_name,
        local_map_size_m=20,
        map_center=[-32.925, -37.3],
        map_res=0.05,
        state_type="hits",
        clip_len=8,
        flatten_img=False,
        load_gt_map=False,
        rebalance_samples=False,
        num_bins=15,
        map_recon_dim=64,
        dataset_fraction=1.0,
    ):
        self.dataset_dir = dataset_dir
        self.clip_len = clip_len
        self.flatten_img = flatten_img
        self.state_type = state_type
        self.map_res = map_res
        self.map_center = map_center
        self.local_map_size_m = local_map_size_m
        self.local_map_size_px = self.local_map_size_m / self.map_res
        self.map_recon_dim = map_recon_dim
        self.dataset_fraction = dataset_fraction

        # Load annotation file.
        # Format:
        # {
        #     'type': 'video',
        #     'ann': {
        #         'video name 1': [
        #             {'timestamp': timestamp, 'img_rel_path': img_rel_path, 'flow_rel_path': flow_rel_path, 'vel': vel},
        #             ...
        #         ]
        #         ...
        #     }
        # }

        ann_path = os.path.join(dataset_dir, ann_file_name)
        with open(ann_path) as f:
            self.ann = json.load(f)
        assert self.ann["type"] == "mushr_sim_pretrain"

        # Generate clip indices. Format: (video name, start frame index).
        self.clip_indices = []
        self.clip_actions = []
        self.video_names = []
        max_num_videos = int(dataset_fraction * len(self.ann["ann"]))
        for video_idx, video_name in enumerate(self.ann["ann"]):
            if video_idx > max_num_videos:
                break
            video = self.ann["ann"][video_name]
            self.video_names.append(video_name)
            # dir_name = video_name['dir_name']
            # episode_name = str(video_name['episode_number'])
            # video = '/'.join([dir_name, 'processed_images', episode_name])
            if len(video) >= clip_len:
                for start_frame_index in range(len(video) - clip_len + 1):
                    self.clip_indices.append((video_name, start_frame_index))
                    self.clip_actions.append(video[start_frame_index + clip_len - 1]["action"])

        # Other settings.
        self.transform = transform
        self.num_bins = num_bins

    def __len__(self):
        return len(self.clip_indices)

    def norm_angle(self, angle):
        # normalize all actions
        act_max = 0.38
        act_min = -0.38
        return 2.0 * (angle - act_min) / (act_max - act_min) - 1.0

    def crop_map_new(self, center_pose):
        cx = center_pose[0]
        cy = center_pose[1]
        rect = ((cx, cy), (self.local_map_size_px, self.local_map_size_px), 90 - math.degrees(center_pose[2]))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # get width and height of the detected rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])
        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been
        # straightened
        dst_pts = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")
        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(self.orig_map, M, (width, height))
        return warped

    def pose2pixel(self, pose):
        # assume pose has [x, y, theta]. is given in m with respect to map frame
        # we are ignoring rotation of map frame for now
        col_x = pose[0] / self.map_res - self.map_center[0] / self.map_res
        row_y = -pose[1] / self.map_res - self.map_center[1] / self.map_res
        return int(col_x), int(row_y)

    def get_full_eval_trajectory(self, traj_index=None):
        # process the index, cut off at the max length of trajectories
        if traj_index is None:
            traj_index = torch.randint(len(self.video_names), (1,)).item()
        else:
            traj_index = min(len(self.video_names) - 1, traj_index)
        # get all the clip indices for that video
        video_name = self.video_names[traj_index]
        video = self.ann["ann"][video_name]
        items = []
        if len(video) >= 2 * self.clip_len:
            for start_frame_index in range(len(video) - self.clip_len + 1):
                items.append(self.get_item_internal(video_name, start_frame_index))
        else:
            raise NotImplementedError()
        return items

    def __getitem__(self, index):
        # Get annotation.
        video_name, start_frame_index = self.clip_indices[index]

        return self.get_item_internal(video_name, start_frame_index)

    def get_item_internal(self, video_name, start_frame_index):
        states = []
        acts = []
        poses = []
        for frame_index in range(start_frame_index, start_frame_index + self.clip_len):
            frame_ann = self.ann["ann"][video_name][frame_index]

            # load the state representation
            if self.state_type == "hits":
                state_rel_path = frame_ann["img_bev_rel_path"]
            elif self.state_type == "occupancy":
                state_rel_path = frame_ann["img_occupancy_rel_path"]
            elif self.state_type == "pcl":
                state_rel_path = frame_ann["pcl_rel_path"]
            else:
                raise RuntimeError("Data type not supported!")
            state_path = os.path.join(self.dataset_dir, state_rel_path)

            states = np.load(state_path)

            act = frame_ann["action"]
            act = np.array(self.norm_angle(act), dtype=np.float32)
            acts.append(act)

            pose = frame_ann["pose"]
            pose = np.array(pose, dtype=np.float32)
            poses.append(pose)

            # get the middle pose
            # subtract 1 from clip len because had to add one for localization
            if frame_index == start_frame_index + int((self.clip_len - 1) / 2) - 1:
                middle_pose = np.array(frame_ann["pose"])

        if self.state_type == "hits" or self.state_type == "occupancy":
            if self.flatten_img:
                img_seq = [torch.from_numpy(item) for item in states]
                states = torch.stack(img_seq, 0)  # shape: [C] -> [D, C]
            else:
                states = torch.stack(states, dim=0)  # Shape: [C,H,W] -> [N,C,H,W].
        elif self.state_type == "pcl":
            states = [torch.from_numpy(item) for item in states]
            states = torch.stack(states, dim=0)  # Shape: [C,2] -> [N,C,2].
        else:
            raise RuntimeError("Data type not supported!")

        # torch_processing_time = time.time()
        # print(f"torch_processing_time = {torch_processing_time-reading_time}")

        # process the action information
        act_seq = np.asarray(acts)[np.newaxis, :]

        poses = np.asarray(poses)
        pose_seq = poses

        return states, act_seq, pose_seq

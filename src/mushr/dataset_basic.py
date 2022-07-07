import json
import math
import os

import cv2
import numpy as np
import torch
from mushr.dataset_utils import norm_angle, process_relative_poses_deltas_incremental
from PIL import Image
from skimage.transform import resize


class MushrVideoDatasetPreload(torch.utils.data.Dataset):
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
        self.transform = transform

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

        max_num_videos = int(dataset_fraction * len(self.ann["ann"]))

        # Generate clip indices. Format: (video name, start frame index).
        self.clip_indices = []
        self.clip_actions = []
        self.video_names = []

        # read the entire dataset and append to the main dict
        for video_idx, video_name in enumerate(self.ann["ann"]):
            if video_idx % 200 == 0:
                print("Done loading {:.2f} % of the videos".format(float(video_idx / (max_num_videos + 1)) * 100.0))

            if video_idx > max_num_videos:
                break

            video = self.ann["ann"][video_name]
            self.video_names.append(video_name)

            if len(video["data"]) >= clip_len:
                for start_frame_index in range(len(video["data"]) - clip_len + 1):
                    self.clip_indices.append((video_name, start_frame_index))
                    self.clip_actions.append(video["data"][start_frame_index + clip_len - 1]["action"])

            # read the npy file that contains all the state information
            if self.state_type == "hits" or self.state_type == "occupancy":
                state_path = video["metadata"]["all_images_path"]
            elif self.state_type == "pcl":
                state_path = video["metadata"]["all_pcls_path"]
            else:
                raise RuntimeError("Data type not supported!")
            state_path = os.path.join(self.dataset_dir, state_path)
            states = np.load(state_path)

            for frame_idx, frame in enumerate(video["data"]):
                # read the state information and append to the main dictionary:

                # load the state info
                if self.state_type == "hits" or self.state_type == "occupancy":
                    if self.flatten_img:
                        img = states[frame_idx, :]
                        img = 2.0 * img / 254.0 - 1.0
                        img = np.array(img, dtype=np.float32).flatten()
                        frame["state"] = img
                    else:
                        img = states[frame_idx, :]
                        img = Image.fromarray(img)
                        frame["state"] = self.transform(img)
                elif self.state_type == "pcl":
                    frame["state"] = states[frame_idx, :]
                else:
                    raise RuntimeError("Data type not supported!")

        # Other settings.
        self.num_bins = num_bins
        # create sampling classes if needed
        self.rebalance_samples = rebalance_samples
        if self.rebalance_samples:
            self.clip_actions = np.array(self.clip_actions)
            # create bins for classes
            bins = np.linspace(self.clip_actions.min(), self.clip_actions.max(), num=self.num_bins)
            # insert indices in each bin
            binplace = np.digitize(self.clip_actions, bins, right=True)
            self.binned_indices = []
            sum = 0
            for i in range(self.num_bins):
                self.binned_indices.append(np.where(binplace == i)[0])
                print(len(self.binned_indices[i]))
                sum += len(self.binned_indices[i])

        # open the GT bravern map if there is one
        self.load_gt_map = load_gt_map
        if self.load_gt_map:
            gt_map_name = os.path.join(dataset_dir, gt_map_file_name)
            self.orig_map = cv2.imread(gt_map_name, -1)
            self.orig_map[self.orig_map > 0] = 255  # make the gray area become white, as free space
            self.orig_map = -(2.0 * self.orig_map / 255.0 - 1.0).astype(np.float32)  # invert colors and normalize to -1,1 range
            # change the color scheme to 0 free to 1 occupied
            self.orig_map[self.orig_map > -1] = 1

    def crop_map_new(self, center_pose):
        cx = center_pose[0]
        cy = center_pose[1]

        rect = (
            (cx, cy),
            (self.local_map_size_px, self.local_map_size_px),
            90 - math.degrees(center_pose[2]),
        )
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # get width and height of the detected rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])
        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been
        # straightened
        dst_pts = np.array(
            [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
            dtype="float32",
        )
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

    def __len__(self):
        return len(self.clip_indices)

    def get_full_eval_trajectory(self, traj_index=None):
        # process the index, cut off at the max length of trajectories
        if traj_index is None:
            traj_index = torch.randint(len(self.video_names), (1,)).item()
        else:
            traj_index = min(len(self.video_names) - 1, traj_index)
        # get all the clip indices for that video
        video_name = self.video_names[traj_index]
        video = self.ann["ann"][video_name]["data"]
        items = []
        if len(video) >= 2 * self.clip_len:
            for start_frame_index in range(len(video) - self.clip_len + 1):
                items.append(self.get_item_internal(video_name, start_frame_index))
        else:
            raise NotImplementedError()
        return items

    def __getitem__(self, index):

        # Transform the index into a weighted sample
        if self.rebalance_samples:
            idx_bin = np.random.randint(len(self.binned_indices))  # choose a bin
            idx_clip = np.random.randint(len(self.binned_indices[idx_bin]))  # choose random element from bin
            index = self.binned_indices[idx_bin][idx_clip]

        # Get annotation.
        video_name, start_frame_index = self.clip_indices[index]

        return self._get_item_internal(video_name, start_frame_index)

    def _get_item_internal(self, video_name, start_frame_index):

        # start_time = time.time()

        states = []
        acts = []
        poses = []
        for frame_index in range(start_frame_index, start_frame_index + self.clip_len):
            frame_ann = self.ann["ann"][video_name]["data"][frame_index]

            # load the state info
            states.append(frame_ann["state"])

            # Load angles and normalize.
            act = frame_ann["action"]
            act = np.array(norm_angle(act), dtype=np.float32)
            acts.append(act)

            # before_pose_time = time.time()
            pose = frame_ann["pose"]
            pose = np.array(pose, dtype=np.float32)
            poses.append(pose)

            # get the middle pose
            # subtract 1 from clip len because had to add one for localization
            if frame_index == start_frame_index + int((self.clip_len - 1) / 2) - 1:
                middle_pose = np.array(frame_ann["pose"])

        states = np.asarray(states).astype(np.float32)

        # process the action information
        act_seq = np.asarray(acts)[np.newaxis, :]

        poses = np.array(poses)
        # poses = process_relative_poses_deltas_incremental(poses)

        pose_seq = poses

        return states, act_seq, pose_seq

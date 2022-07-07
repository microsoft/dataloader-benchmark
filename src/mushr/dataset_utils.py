import numpy as np
import math


def norm_angle(angle):
    # normalize all actions
    act_max = 0.38
    act_min = -0.38
    return 2.0 * (angle - act_min) / (act_max - act_min) - 1.0


def process_relative_poses_deltas_incremental(poses):
    num_poses = poses.shape[0]
    new_poses = np.zeros(shape=(num_poses - 1, 3))
    # iterate but skip the first point (because it outputs a zero delta)
    for i in range(num_poses - 1):
        new_poses[i, :] = calc_relative_pose(poses[i], poses[i + 1])
    return new_poses


def wrap_to_pi(a):
    # return the angle in the -pi, pi range
    m = int(a / (2.0 * math.pi))
    a = a - m * 2.0 * math.pi
    if a > math.pi:
        a -= 2.0 * math.pi
    elif a < -math.pi:
        a += 2.0 * math.pi
    return a


def angle_diff(a1, a2):
    # we want the smallest a1-a2 result, with the correct sign
    a1 = wrap_to_pi(a1)
    a2 = wrap_to_pi(a2)
    diff = a1 - a2
    if diff < -math.pi:
        diff += 2 * math.pi
    if diff > math.pi:
        diff -= 2.0 * math.pi
    return diff


def calc_relative_pose(pose_base, pose_desired):
    # pose_base and pose_desired are vectors of size 3 with X,Y,Theta in world frame
    # elements of homogeneous matrix expressing point from local frame into world frame coords
    R = np.array(
        [
            [np.cos(pose_base[2]), -np.sin(pose_base[2])],
            [np.sin(pose_base[2]), np.cos(pose_base[2])],
        ]
    )
    t = np.array([[pose_base[0]], [pose_base[1]]])
    # we create the inverse matrix to obtain the pose of a point from world coords into the base frame
    R_new = np.transpose(R)  # faster than computing np.linalg.inv(R)
    t_new = -np.matmul(R_new, t)
    T = np.array(
        [
            [R_new[0, 0], R_new[0, 1], t_new[0, 0]],
            [R_new[1, 0], R_new[1, 1], t_new[1, 0]],
            [0, 0, 1],
        ]
    )
    pose_world = np.array([[pose_desired[0]], [pose_desired[1]], [1]])
    pose_local = np.matmul(T, pose_world)
    delta_angle = angle_diff(pose_desired[2], pose_base[2])
    return np.array([pose_local[0, 0], pose_local[1, 0], delta_angle])


def transform_poses(current_pose, delta_pose_pred):
    # elements of homogeneous matrix expressing point from local frame into world frame coords
    current_angle = current_pose[2]
    R = np.array(
        [
            [np.cos(current_angle), -np.sin(current_angle)],
            [np.sin(current_angle), np.cos(current_angle)],
        ]
    )
    t = np.array([[current_pose[0]], [current_pose[1]]])
    T = np.array([[R[0, 0], R[0, 1], t[0, 0]], [R[1, 0], R[1, 1], t[1, 0]], [0, 0, 1]])
    # now transform the position of the next point from local to world frame
    pose_local = np.array([[delta_pose_pred[0]], [delta_pose_pred[1]], [1]])
    pose_world = np.matmul(T, pose_local)
    next_pose_in_world_frame = np.zeros(shape=(3,))
    next_pose_in_world_frame[0] = pose_world[0, 0]
    next_pose_in_world_frame[1] = pose_world[1, 0]
    next_pose_in_world_frame[2] = wrap_to_pi(current_angle + delta_pose_pred[2])
    return next_pose_in_world_frame

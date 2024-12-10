import os
import numpy as np


def rotation_matrix_to_axis_angle(rot_matrices):
    """Convert rotation matrices to axis-angle representation."""
    angles = np.arccos((np.trace(rot_matrices) - 1) / 2.0)  # Extract angle from trace
    return angles

data_root = "vlr_data/param/"
split_file = "vlr_data/val_split.txt"
split = [index.strip() for index in open(split_file)]

for s in split:
    data = np.load(os.path.join(data_root, s + "_params.npz"))
    hand_pose = data["hand_pose"]
    bending_angles = np.stack([
        rotation_matrix_to_axis_angle(hand_pose[0, joint]) 
        for joint in range(15)
    ])
    
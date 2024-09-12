import os
from pathlib import Path
import cv2
import numpy as np
from pyquaternion import Quaternion


class BONNDataset:
    def __init__(self, data_dir, *_, **__):
        self.data_dir = Path(data_dir)

        # Load depth frames
        depth_path = os.path.join(self.data_dir,"depth")
        rgb_path = os.path.join(self.data_dir,"rgb")
        self.depth_frames = sorted([os.path.join(depth_path, x) for x in os.listdir(depth_path) if ".png" in x])
        self.rgb_frames = sorted([os.path.join(rgb_path, x) for x in os.listdir(rgb_path) if ".png" in x])
        self.rgb_timestamps = np.asarray([float(str(os.path.basename(x)).split(".png")[0]) for x in self.rgb_frames])
        self.depth_timestamps = np.asarray([float(str(os.path.basename(x)).split(".png")[0]) for x in self.depth_frames])
        # matches
        self.matches = self.get_matches(self.depth_timestamps, self.rgb_timestamps)
        # depth scale factor
        self.depth_scale = 5000.0
        # Load GT poses
        gt_list = np.loadtxt(fname=os.path.join(self.data_dir, "groundtruth.txt"), dtype=str)
        self.gt_poses = self.load_poses(gt_list)

    def __len__(self):
        #  return len(self.matches)
        return len(self.matches)  # temp remove and uncomment above

    def get_matches(self, src_timestamps, target_timestamps):
        indices = np.abs(
            (
                np.subtract.outer(
                    src_timestamps.astype(np.float64),
                    target_timestamps.astype(np.float64),
                )
            )
        )
        src_matches = np.arange(len(src_timestamps))
        target_matches = np.argmin(indices, axis=1)
        _, unique_indxs = np.unique(target_matches, return_index=True)
        matches = np.vstack((src_matches[unique_indxs], target_matches[unique_indxs])).T
        return matches

    def load_poses(self, gt_list):
        gt_indices = self.get_matches(
            gt_list[:, 0].astype(np.float64),
            self.depth_timestamps[self.matches[:, 0]].astype(np.float64),
        )
        xyz = gt_list[gt_indices[:, 0]][:, 1:4]

        rotations = np.array(
            [
                Quaternion(x=x, y=y, z=z, w=w).rotation_matrix
                for x, y, z, w in gt_list[gt_indices[:, 0]][:, 4:]
            ]
        )
        num_poses = rotations.shape[0]
        poses = np.eye(4, dtype=np.float64).reshape(1, 4, 4).repeat(num_poses, axis=0)
        poses[:, :3, :3] = rotations
        poses[:, :3, -1] = xyz
        return poses

    def get_intrinsics(self) -> np.ndarray:
        #  intrinsics = np.array(([fx, 0, cx], [0, fy, cy], [0, 0, 1]))  # focal length x
        cx = 315.593520
        cy = 237.756098
        fx = 542.822841
        fy = 542.576870
        return np.array(([fx, 0, cx], [0, fy, cy], [0, 0, 1]))

    def __getitem__(self, idx):
        depth_idx, rgb_idx = self.matches[idx]
        rgb = str(self.rgb_frames[rgb_idx])
        #  print(str(self.data_dir / self.rgb_frames[rgb_idx][-1]))
        # read images and process depth
        depth = (self.depth_frames[depth_idx])
        return rgb, depth

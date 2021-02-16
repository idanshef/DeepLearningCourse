import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from robotcar_dataset_sdk.image import load_image
from robotcar_dataset_sdk.camera_model import CameraModel
from robotcar_dataset_sdk.build_pointcloud import build_pointcloud


class RobotCarDataset(Dataset):
    def __init__(self, data_dir):
        # self.image_paths = utils.get_image_paths(data_dir)
        # self.num_of_samples = len(self.image_paths)
        self.camera_model = CameraModel(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"), images_dir)

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = load_image(image_path, self.camera_model)

        pointcloud, reflectance = build_pointcloud(lidar_dir, poses_file, extrinsics_dir, start_time, end_time)

        img_tensor = torch.from_numpy(np.array([image]))
        structure_tensor = torch.from_numpy(pointcloud)

        return {'image': img_tensor, 'structure': structure_tensor}
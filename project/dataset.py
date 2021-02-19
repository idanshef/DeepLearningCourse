import os
import cv2
import numpy as np
import torch
import utils
import random
from torch.utils.data import Dataset
from robotcar_dataset_sdk.image import load_image
from robotcar_dataset_sdk.build_pointcloud import build_pointcloud
from robotcar_dataset_sdk.camera_model import CameraModel


class RobotCarDataset(Dataset):
    def __init__(self, data_dir, models_dir, poses_file_path):
        self.poses_file_path = poses_file_path
        
        data = DataTreeStructure(data_dir)
        self.samples_couple_list = create_dataset(data)
        self.num_of_couples = len(self.samples_couple_list)
        
        self.camera_model = CameraModel(models_dir, data.images_dir)


    def __len__(self):
        return self.num_of_couples

    def __getitem__(self, idx):
        samples_couple = self.samples_couple_list[idx]

        return {'samples': {'Xi': samples_couple.Xi.load_sample(self.camera_model, self.poses_file_path), 
                            'Xj': samples_couple.Xj.load_sample(self.camera_model, self.poses_file_path)}, 
                'label': samples_couple.calculate_match()}


class Sample:
    def __init__(self, img_path, lidar_dir, max_diff_sec=5):
        self.max_diff_sec = max_diff_sec
        self.img_path = img_path
        self.start_time, self.end_time = self._calculate_structure_times()
        self.lidar_dir = lidar_dir
    
    def _calculate_structure_times(self):
        lidar_scans_timestamps = np.array([np.fromstring(val[:-4], dtype=np.int64) for val in os.listdir(self.lidar_dir)])
        relevant_lidar = lidar_scans_timestamps[abs(lidar_scans_timestamps - np.fromstring(self.img_path[:-4], dtype=np.int64)) <= self.max_diff_sec]
        return relevant_lidar.min(), relevant_lidar.max()
    
    def load_sample(self, camera_model, poses_file_path):
        extrinsics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extrinsics")

        img_mat = load_image(self.img_path, camera_model)
        pointcloud, reflectance = build_pointcloud(self.lidar_dir, poses_file_path, extrinsics_dir, self.start_time, self.end_time)
        
        I = torch.from_numpy(img_mat)
        S = torch.from_numpy(pointcloud)
        
        return I, S


class SamplesCouple:
    def __init__(self, Ii_path, Ij_path, lidar_dir):
        self.Xi = Sample(Ii_path, lidar_dir)
        self.Xj = Sample(Ij_path, lidar_dir)
    
    def calculate_match(self, ):
        ## Calculate
        return True

class DataTreeStructure:
    def __init__(self, data_dir):
        self.images_dir = os.path.join(data_dir, "stereo", "centre")
        self.lidar_dir = os.path.join(data_dir, "ldmrs")


def create_dataset(data):
    samples_couple_list = list()
    images_list = os.listdir(data.images_dir)
    
    N = len(images_list) * 2
    for i in range(N):
        img_couple_paths = [os.path.join(data.images_dir, val) for val in random.sample(images_list, 2)]
        samples_couple_list.append(SamplesCouple(img_couple_paths[0], img_couple_paths[1], data.lidar_dir))
    
    return samples_couple_list
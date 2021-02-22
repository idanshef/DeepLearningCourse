import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from geopy.distance import distance
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from robotcar_dataset_sdk.image import load_image
from robotcar_dataset_sdk.camera_model import CameraModel
from robotcar_dataset_sdk.build_pointcloud import build_pointcloud


class RobotCarDataset(Dataset):
    def __init__(self, data_dir):
        self.data = DatasetStructure(data_dir)
        
        self.samples_couple_list = create_dataset(self.data)
        self.num_of_couples = len(self.samples_couple_list)
        
        self.extrinsics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extrinsics")
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.camera_model = CameraModel(models_dir, self.data.images_dir)
        

    def __len__(self):
        return self.num_of_couples

    def __getitem__(self, idx):
        samples_couple = self.samples_couple_list[idx]
        Ii, Gi = samples_couple.Xi.load_sample(self.camera_model, self.data, self.extrinsics_dir)
        Ij, Gj = samples_couple.Xj.load_sample(self.camera_model, self.data, self.extrinsics_dir)
        
        # Ii = (2*self.to_tensor(Ii)-1)/2
        # Ij = (2*self.to_tensor(Ij)-1)/2
        
        # Gi = pointcloud_to_voxel_grid(Gi)
        # Gj = pointcloud_to_voxel_grid(Gj)

        return {'Ii': Ii, 'Gi': Gi, 'Ij': Ij, 'Gj': Gj, 'label': samples_couple.match}


class Sample:
    def __init__(self, img_path, data, max_diff_sec=5):
        self.img_path = img_path
        self.lidar_dir = data.lidar_dir
        
        relevant_lidar = data.lidar_scans_timestamps[abs(data.lidar_scans_timestamps - int(os.path.basename(self.img_path)[:-4])) <= max_diff_sec * 1e6]
        self.start_time, self.end_time = relevant_lidar.min(), relevant_lidar.max()
        # self.to_tensor = ToTensor()
    
    def load_sample(self, camera_model, data, extrinsics_dir):
        img_mat = load_image(self.img_path, camera_model)
        pointcloud, reflectance = build_pointcloud(data.lidar_dir, data.poses_file_path, extrinsics_dir, self.start_time, self.end_time)
        
        I = (2 * (torch.from_numpy(img_mat).permute(2, 0, 1)/255.) - 1) / 2
        G = torch.from_numpy(pointcloud)
        
        return I, G


class SamplesCouple:
    def __init__(self, Ii_path, Ij_path, data):
        self.Xi = Sample(Ii_path, data)
        self.Xj = Sample(Ij_path, data)
        self.match = self._calculate_match(data.gps_df)
    
    def _calculate_match(self, gps_df):
        Xi_closest_time_idx = abs(gps_df['timestamp'] - int(os.path.basename(self.Xi.img_path)[:-4])).argmin()
        Xj_closest_time_idx = abs(gps_df['timestamp'] - int(os.path.basename(self.Xj.img_path)[:-4])).argmin()
        
        Xi_lat_long = (gps_df['latitude'][Xi_closest_time_idx], gps_df['longitude'][Xi_closest_time_idx])
        Xj_lat_long = (gps_df['latitude'][Xj_closest_time_idx], gps_df['longitude'][Xj_closest_time_idx])
        
        return distance(Xi_lat_long, Xj_lat_long).m <= 20

class DatasetStructure:
    def __init__(self, data_dir):
        self.images_dir = os.path.join(data_dir, "stereo", "centre")
        self.images_list = list(filter(lambda val: val[0] == "1", os.listdir(self.images_dir)))
        
        self.lidar_dir = os.path.join(data_dir, "ldmrs")
        lidar_scans_paths = list(filter(lambda val: val[0] == "1", os.listdir(self.lidar_dir)))
        self.lidar_scans_timestamps = np.array([int(val[:-4]) for val in lidar_scans_paths])
        
        gps_dir = os.path.join(data_dir, "gps")
        self.poses_file_path = os.path.join(gps_dir, "ins.csv")
        self.gps_df = pd.read_csv(self.poses_file_path)
    

def create_dataset(data):
    samples_couple_list = list()
    
    N = len(data.images_list) * 2
    for i in range(N):
        img_couple_paths = [os.path.join(data.images_dir, val) for val in random.sample(data.images_list, 2)]
        samples_couple_list.append(SamplesCouple(img_couple_paths[0], img_couple_paths[1], data))
    
    return samples_couple_list
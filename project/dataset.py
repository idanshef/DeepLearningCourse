import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from geopy.distance import distance
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import utils
from robotcar_dataset_sdk.image import load_image
from robotcar_dataset_sdk.camera_model import CameraModel
from robotcar_dataset_sdk.build_pointcloud import build_pointcloud


class RobotCarDataset(Dataset):
    def __init__(self, data_dir, structure_time_span, match_threshold):
        self.samples_list , self.full_gps_df_rad = utils.build_samples_list(data_dir, structure_time_span)        
        self.extrinsics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extrinsics")
        self.match_threshold = match_threshold
        self.to_tensor = torch.ToTensor()

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx_i):
        Ii, Gi = self._load_sample(idx_i)
        
        if random.random() > 0.5:
            idx_j = _get_match_idx(idx_i)
            y = 1
        else:
            idx_j = self._get_non_match_idx(idx_i)
            y = -1
        
        Ij, Gj = self._load_sample(idx_j)

        return {'Ii': Ii, 'Gi': Gi, 'Ij': Ij, 'Gj': Gj, 'is_match': y}
    
    def _load_sample(self, idx):
        curr_sample = self.samples_list[idx]
        I = load_image(curr_sample['I'], curr_sample['camera'])
        I = (2 * self.to_tensor(I) - 1) / 2
        
        pointcloud, reflectance = build_pointcloud(curr_sample['lidar_dir'], curr_sample['poses_path'], 
                                                   self.extrinsics_dir, curr_sample['start_time'], curr_sample['end_time'])
        pointcloud = np.array(pointcloud[:-1]).transpose()
        G = torch.from_numpy(utils.create_voxel_grid_from_point_cloud(pointcloud)).unsqueeze(0)
        
        return I,G
    
    def _calc_match_idxs(self, idx):
        Xi = self.samples_list[idx]
        Xi_lat_long = np.array([Xi['latitude'], Xi['logitude']])
        Xi_lat_long_mat = np.tile(Xi_lat_long, (self.full_gps_df.shape[0], 1))
        Xi_lat_long_mat_rad = np.array(list(map(np.radians, Xi_lat_long_mat)))
        
        lat1, lon1 = self.full_gps_df_rad[:, 0], self.full_gps_df_rad[:, 1]
        lat2, lon2 = Xi_lat_long_mat_rad[:, 0], Xi_lat_long_mat_rad[:, 1]
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        distance_m = 6367 * 1e3 * 2 * np.arcsin(np.sqrt(np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2))
        
        return np.where(distance_m <= self.match_threshold)[0]
    
    def _get_match_idx(self, idx_i):
        match_idxs = self._calc_match_idxs(idx_i, self.match_threshold)
        while True:
            idx_j = random.choice(match_idxs)
            if self.samples_list[idx_i]['date'] != self.samples_list[idx_j]['date']:
                break
        return idx_j
    
    def _get_non_match_idx(self, idx_i):
        while True:
            idx_j = random.randrange(self.__len__())
            if utils.is_match(self.samples_list[idx_i], self.samples_list[idx_j], self.match_threshold) == False:
                break
        return idx_j

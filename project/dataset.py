import os
import torch
import random
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import utils
from robotcar_dataset_sdk.image import load_image
from robotcar_dataset_sdk.build_pointcloud import build_pointcloud


class RobotCarDataset(Dataset):
    def __init__(self, dataset_df, cameras_model, match_threshold):
        self.samples_df = dataset_df
        self.camera_model_dict = cameras_model
        self.gps_lat_long_rad = np.radians(self.samples_df[['latitude', 'longitude']].to_numpy())
        self.extrinsics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extrinsics")
        self.match_threshold = match_threshold
        self.to_tensor = ToTensor()
    
    @classmethod
    def subset_of_dataset(cls, dataset, idxs):
        return cls(dataset.samples_df.iloc[idxs, :].reset_index(drop=True), 
                   dataset.camera_model_dict, dataset.match_threshold)

    def calc_matches_idxs(self, idxs, include_same_day=False):
        subset_gps_mat = self.gps_lat_long_rad[idxs, :]
        subset_repeat = subset_gps_mat.repeat(self.gps_lat_long_rad.shape[0], axis=0)
        tile_gps_mat = np.tile(self.gps_lat_long_rad, (len(idxs),1))
        
        matches_bool = self._calc_matches_bool(subset_repeat, tile_gps_mat, self.match_threshold).reshape(len(idxs), -1)
        matches_rows, matches_cols = np.where(matches_bool)
        
        remove_idx = lambda idxs_list, idx: idxs_list[idxs_list != idxs[idx]]
        matches_idxs = [remove_idx(matches_cols[matches_rows==val],val) for val in range(len(idxs))]
        
        if include_same_day==False:
            exclude_same_day = lambda matches_df, idx: list(matches_df.loc[
                matches_df['date'] != self.samples_df.iloc[idx]['date']].index.values)
            matches_idxs = [exclude_same_day(self.samples_df.iloc[matches_list], idx_matches) 
                            for matches_list, idx_matches in zip(matches_idxs, idxs)]
        
        return matches_idxs

    def calc_matches_bool(self, idx_i, idx_j):
        assert len(idx_i) == len(idx_j), f"Lists length does not match: {len(idx_i)=}, {len(idx_j)=}"
        gps_i = self.gps_lat_long_rad[idx_i, :]
        gps_j = self.gps_lat_long_rad[idx_j, :]
        return self._calc_matches_bool(gps_i, gps_j, self.match_threshold)
    
    def calc_matches_mat(self, idxs):
        subset_gps_mat = self.gps_lat_long_rad[idxs, :]
        subset_repeat = subset_gps_mat.repeat(len(idxs), axis=0)
        tile_gps_mat = np.tile(subset_gps_mat, (len(idxs),1))
        matches_vec = self._calc_matches_bool(tile_gps_mat, subset_repeat, self.match_threshold)
        return matches_vec.reshape((len(idxs),-1))

    def calc_matches_to_point(self, lat, long, radius):
        lat_long = np.array([[lat, long]])
        repeat_lat_long = lat_long.repeat(self.gps_lat_long_rad.shape[0], axis=0)
        return self._calc_matches_bool(repeat_lat_long, self.gps_lat_long_rad, radius)

    def _calc_matches_bool(self, lat_long_i, lat_long_j, match_threshold_m):
        lat1, lon1 = lat_long_i[:, 0], lat_long_i[:, 1]
        lat2, lon2 = lat_long_j[:, 0], lat_long_j[:, 1]
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        distance_m = 6367 * 1e3 * 2 * np.arcsin(np.sqrt(np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2))
        
        matches_bool = (distance_m <= match_threshold_m)
        return matches_bool
    
    def get_items_at(self, idxs):
        I, G = None, None
        for i in idxs:
            Ii, Gi = self._load_sample(i)
            Ii, Gi = Ii.unsqueeze(0), Gi.unsqueeze(0)
            
            if I is None:
                I = Ii.clone()
                G = Gi.clone()
            else:
                I = torch.cat((I, Ii), dim=0)
                G = torch.cat((G, Gi), dim=0)
        return I, G


    def _load_sample(self, idx):
        curr_sample = self.samples_df.loc[idx]

        I = cv2.resize(load_image(curr_sample['image_path'], self.camera_model_dict[curr_sample['date']])[:600,:,:], (640,300), interpolation=cv2.INTER_CUBIC)
        I = (2 * self.to_tensor(I) - 1) / 2

        pointcloud, reflectance = build_pointcloud(curr_sample['lidar_dir'], curr_sample['poses_path'], 
                                                   self.extrinsics_dir, curr_sample['timestamps'])
        pointcloud = np.array(pointcloud[:-1]).transpose()
        G = torch.from_numpy(utils.create_voxel_grid_from_point_cloud(pointcloud)).unsqueeze(0)

        return I, G

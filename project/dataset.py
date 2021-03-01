import os
import torch
import random
import numpy as np
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

    def calc_match_idxs(self, idx):
        Xi = self.samples_df.loc[idx]
        Xi_lat_long = np.array([Xi['latitude'], Xi['longitude']])
        Xi_lat_long_mat = np.tile(Xi_lat_long, (self.gps_lat_long_rad.shape[0], 1))
        Xi_lat_long_mat_rad = np.array(list(map(np.radians, Xi_lat_long_mat)))
        
        lat1, lon1 = lat_long_rad[:, 0], lat_long_rad[:, 1]
        lat2, lon2 = Xi_lat_long_mat_rad[:, 0], Xi_lat_long_mat_rad[:, 1]
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        distance_m = 6367 * 1e3 * 2 * np.arcsin(np.sqrt(np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2))
        
        matches_bool = distance_m <= self.match_threshold
        matches_idxs = np.where(matches_bool)[0]
        non_matches_idxs = np.where(not matches_bool)[0]
        
        remove_idx = lambda idxs: np.delete(idxs, idx)
        matches_idxs = remove_idx(matches_idxs)
        non_matches_idxs = remove_idx(non_matches_idxs)
        return matches_idxs, non_matches_idxs

    def get_item(self, idx_i):
        Ii, Gi = self._load_sample(idx_i)
        
        match_idxs = self._calc_match_idxs(idx_i)
        if len(match_idxs) == 0:
            raise Exception("FML")
        if random.random() > 0.5:
            idx_j = self._get_match_idx(idx_i, match_idxs)
            Ij, Gj = self._load_sample(idx_j)
            Ij = Ij.expand(self.N, *Ij.shape)
            Gj = Gj.expand(self.N, *Gj.shape)
            y = torch.ones(self.N)
        else:
            idx_j = self._get_non_match_idx(idx_i, match_idxs)
            Ij, Gj = self._load_multiple_samples(idx_j)
            y = torch.ones(self.N) * -1

        return {'Ii': Ii, 'Gi': Gi, 'Ij': Ij, 'Gj': Gj, 'is_match': y}

    def _load_sample(self, idx):
        curr_sample = self.samples_df.loc[idx]

        I = load_image(curr_sample['image_path'], self.camera_model_dict[curr_sample['date']])[:600,:,:]
        I = (2 * self.to_tensor(I) - 1) / 2

        pointcloud, reflectance = build_pointcloud(curr_sample['lidar_dir'], curr_sample['poses_path'], 
                                                   self.extrinsics_dir, curr_sample['timestamps'])
        pointcloud = np.array(pointcloud[:-1]).transpose()
        G = torch.from_numpy(utils.create_voxel_grid_from_point_cloud(pointcloud)).unsqueeze(0)

        return I, G

    def _load_multiple_samples(self, idxs):
        I, G = None, None
        for idx in idxs:
            I_curr, G_curr = self._load_sample(idx)
            I_curr, G_curr = I_curr.unsqueeze(0), G_curr.unsqueeze(0)
            if I is None:
                I, G = I_curr, G_curr
            else:
                I = torch.cat((I, I_curr), 0)
                G = torch.cat((G, G_curr), 0)
        
        return I, G


    
    def _get_match_idx(self, idx_i, match_idxs):
        while True:
            idx_j = random.choice(match_idxs) # TODO: possible situaltion were match idxs is empty after few iterations
            if self.samples_df.loc[idx_i]['date'] != self.samples_df.loc[idx_j]['date']:
                break
            match_idxs = np.delete(match_idxs, np.where(match_idxs == idx_j))
        return idx_j
    
    def _get_non_match_idx(self, idx_i, match_idxs):
        non_matches_idxs = np.arange(self.__len__())
        non_matches_idxs = np.delete(non_matches_idxs, match_idxs)
        super_batch = random.sample(list(non_matches_idxs), self.N)
        return super_batch

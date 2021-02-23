import numpy as np


def create_voxel_grid_from_point_cloud(point_cloud, center_m, grid_resolution = (96, 96, 48), volume_size = (40, 40, 20)):
    voxel_grid = np.zeros(grid_resolution)
    filter_point_cloud = abs(point_cloud - np.tile(center_m, (point_cloud.shape[0], 1)))
    
    filter_0 = filter_point_cloud[:, 0] < volume_size[0]/2
    filter_1 = filter_point_cloud[:, 1] < volume_size[1]/2
    filter_2 = filter_point_cloud[:, 2] < volume_size[2]/2
    filter_point_cloud = point_cloud[filter_0 & filter_1 & filter_2, :]
    
    trans_P = lambda P_m: (np.array(grid_resolution)/2) + np.floor(
        (P_m - center_m) * (np.array(grid_resolution) / np.array(volume_size)))
    
    points_axis = np.unique(list(map(trans_P, filter_point_cloud)), axis=0).astype(int)
    np.add.at(voxel_grid, tuple(points_axis.T), 1)
    
    return voxel_grid
import os
import random
import numpy as np
import pandas as pd
from itertools import zip_longest
from robotcar_dataset_sdk.camera_model import CameraModel


def create_dataset_df(data_dir, structure_time_span, dataset_csv):
    data_fields = ['date', 'lidar_dir', 'image_path', 'poses_path', 'timestamps', 'latitude', 'longitude']
    if dataset_csv is not None and os.path.exists(dataset_csv):
        df = pd.read_csv(dataset_csv, sep=',', converters={"timestamps": lambda x: list(map(int, x.strip("[]").replace("'","").split(", ")))})
    else:
        df = build_samples_df(data_dir, structure_time_span, data_fields)
        if dataset_csv is not None:
            df.to_csv(dataset_csv, sep=',')
    
    # TODO: filter data
    # df = df[df['date'].isin(['2015-03-10-14-18-10', '2014-07-14-14-49-50', '2014-11-18-13-20-12', '2014-12-09-13-21-02'])]
    df = df[df['timestamps'].str.len()>1].reset_index()
    return df


def build_samples_df(data_dir, structure_time_span, fields):
    df = pd.DataFrame(columns=fields)
    for _, data_dates, _ in os.walk(data_dir):
        data_dates = data_dates
        break
    
    for data_date in data_dates:
        date_path = os.path.join(data_dir, data_date)
        lidar_dir = os.path.join(date_path, 'ldmrs')
        img_dir = os.path.join(date_path, 'stereo', 'centre')
        poses_file_path = os.path.join(date_path, "gps", "ins.csv")
        gps_df = pd.read_csv(poses_file_path)
        
        lidar_timestamps = np.array(list(map(lambda val: int(val[:-4]), os.listdir(lidar_dir))))
        
        for img_name in os.listdir(img_dir):
            data_dict = dict()
            data_dict['date'] = data_date
            data_dict['lidar_dir'] = lidar_dir
            data_dict['image_path'] = os.path.join(img_dir, img_name)
            data_dict['poses_path'] = poses_file_path
            
            img_timestamp = int(img_name[:-4])
            matching_timestamps = list(lidar_timestamps[abs(lidar_timestamps-img_timestamp) <= round(structure_time_span/2) * 1e6])
            data_dict['timestamps'] = matching_timestamps if len(matching_timestamps)>4 else [0]
            
            closest_time_idx = abs(gps_df['timestamp'] - img_timestamp).argmin()
            data_dict['latitude'] = gps_df['latitude'][closest_time_idx]
            data_dict['longitude'] = gps_df['longitude'][closest_time_idx]

            df = df.append(data_dict, ignore_index=True)
    return df


def load_cameras(data_dir):
    camera_model_dict = dict()
    for data_date in os.listdir(data_dir):
        if os.path.isfile(os.path.join(data_dir, data_date)):
            continue
        date_path = os.path.join(data_dir, data_date)
        img_dir = os.path.join(date_path, 'stereo', 'centre')
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        camera_model_dict[data_date] = CameraModel(models_dir, img_dir)
    return camera_model_dict


def split_idxs_to_train_val_idxs(dataset, validate_lat_long_radius_m, train_lat_long_radius_m):
    validate_lat, validate_long, validate_radius_m = validate_lat_long_radius_m
    train_lat, train_long, train_radius_m = train_lat_long_radius_m
    
    is_match_val = dataset.calc_matches_to_point(np.radians(validate_lat), np.radians(validate_long), validate_radius_m)
    is_match_train = dataset.calc_matches_to_point(np.radians(train_lat), np.radians(train_long), train_radius_m)
    val_set_idxs = np.where(is_match_val)[0]
    train_set_idxs = np.where(is_match_train)[0]
    return train_set_idxs, val_set_idxs

def split_data_to_groups_size_k(dataset, k):
    df_idxs_values = []
    df_idxs_values[:] = dataset.samples_df.index.values
    random.shuffle(df_idxs_values)
    group_size_k_list = list(map(list,zip_longest(*(iter(df_idxs_values),) * k)))
    num_of_none_values = (k - (len(df_idxs_values) % k))
    if num_of_none_values != 0:
        group_size_k_list[-1] = group_size_k_list[-1][:-num_of_none_values]
    return group_size_k_list


def create_voxel_grid_from_point_cloud(point_cloud, grid_resolution = (96, 96, 48), volume_size = (40, 40, 20)):
    center_m = np.zeros(3)
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

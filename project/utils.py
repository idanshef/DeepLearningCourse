import numpy as np


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

def build_samples_list(data_dir, structure_time_span):
    data_list = []
    full_gps_df_rad = None
    for data_date in os.listdir(data_dir):
        date_path = os.path.join(data_dir, data_date)
        lidar_dir = os.path.join(date_path, 'ldmrs')
        img_dir = os.path.join(date_path, 'stereo', 'centre')
        poses_file_path = os.path.join(date_path, "gps", "ins.csv")
        gps_df = pd.read_csv(poses_file_path)
        
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        camera_model = CameraModel(models_dir, img_dir)
        
        for img_name in os.listdir(img_dir):
            data_dict = dict()
            data_dict['date'] = data_date
            data_dict['lidar_dir'] = lidar_dir
            data_dict['camera'] = camera_model
            data_dict['I'] = os.path.join(img_dir, img_name)
            data_dict['poses_path'] = poses_file_path
            
            img_timestamp = int(img_name[:-4])
            data_dict['start_time'] = str(curr_timestamp + val - round(structure_time_span/2) * 1e6)
            data_dict['end_time'] = str(curr_timestamp + val + round(structure_time_span/2) * 1e6)
            
            closest_time_idx = abs(gps_df['timestamp'] - img_timestamp).argmin()
            data_dict['latitude'] = gps_df['latitude'][closest_time_idx]
            data_dict['longitude'] = gps_df['longitude'][closest_time_idx]

            data_list.append(data_dict)
            
            curr_lat_long = np.array(list(map(np.radians, [data_dict['latitude'], data_dict['longitude']])))
            if full_gps_df_rad is None:
                full_gps_df_rad = curr_lat_long
            else:
                full_gps_df_rad = np.concatenate(full_gps_df_rad, curr_lat_long)
    
    return data_list, full_gps_df_rad

def is_match(Xi, Xj, threshold_m):
    Xi_lat_long = (Xi['latitude'], Xi['logitude'])
    Xj_lat_long = (Xj['latitude'], Xj['logitude'])
    return distance(Xi_lat_long, Xj_lat_long).m <= threshold_m

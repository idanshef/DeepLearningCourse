import os
from datetime import datetime
import torch
import torch.nn as nn
from itertools import zip_longest
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from dataset import RobotCarDataset
from net_models import CompoundNet, MarginBasedLoss
import utils
import numpy as np
import random
import time


def init_data(data_dir, dataset_csv, structure_time_span, match_threshold, validate_lat_long_radius_m, train_lat_long_radius_m):    
    dataset_df = utils.create_dataset_df(data_dir, structure_time_span, dataset_csv)
    cameras = utils.load_cameras(data_dir)
    
    full_dataset = RobotCarDataset(dataset_df, cameras, match_threshold)
    train_idxs, val_idxs = utils.split_idxs_to_train_val_idxs(full_dataset, validate_lat_long_radius_m, train_lat_long_radius_m)
    select_subset_df = lambda set_idxs: dataset_df.iloc[set_idxs, :].reset_index()
    
    return {'val': RobotCarDataset(select_subset_df(val_idxs), cameras, match_threshold),
            'train': RobotCarDataset(select_subset_df(train_idxs), cameras, match_threshold)}


def run_model(I, G, device, model):
    I = I.to(device=device, dtype=torch.float32)
    G = G.to(device=device, dtype=torch.float32)
        
    pred_descriptors = model(I, G).to(device='cpu', dtype=torch.float32)
    del I, G
    
    return pred_descriptors


def hard_mine(model, device, I, G, batch_size, k, matches_mat):
    pred_descriptors_group = None
    
    half_batch_size = int(batch_size/2)
    samples_group_list = list(map(np.array,zip_longest(*(iter(range(I.shape[0])),) * half_batch_size)))
    num_of_none_values = (half_batch_size - (I.shape[0] % half_batch_size)) % half_batch_size
    if num_of_none_values != 0:
        samples_group_list[-1] = samples_group_list[-1][:-num_of_none_values].astype(int)
        
    for samples_group in samples_group_list:
        pred_descriptors = run_model(I[samples_group], G[samples_group], device, model)
        
        if pred_descriptors_group is None:
            pred_descriptors_group = pred_descriptors.clone()
        else:
            pred_descriptors_group = torch.cat((pred_descriptors_group, pred_descriptors), dim=0)
    
    descriptor_size = pred_descriptors_group.shape[1]
    super_batch_size = I.shape[0]
    repeat_pred = pred_descriptors_group.repeat(1,super_batch_size).view(-1,descriptor_size)
    d_L1 = torch.sum(torch.abs(repeat_pred - pred_descriptors_group.repeat(super_batch_size,1)), dim=1).view(super_batch_size,-1)
    d_L1[matches_mat==True] = torch.finfo(torch.float32).max
    d_L1[torch.tril(torch.ones(d_L1.shape),diagonal=-1)==0] = torch.finfo(torch.float32).max
    
    min_d = torch.min(d_L1, dim=0)
    top_half_k_matches = torch.topk(-min_d.values, int(super_batch_size/2))
    
    i_idxs = min_d.indices[top_half_k_matches.indices]
    j_idxs = top_half_k_matches.indices

    return i_idxs, j_idxs


def balance_batch(I_k, G_k, dataset_orig, i_j_labels_hard_k, j_idxs_match_orig_list):
    i_idxs_hard_k, j_idxs_hard_k, labels_hard = i_j_labels_hard_k
    
    random_order_hard = np.arange(len(i_idxs_hard_k))
    random.shuffle(random_order_hard)
    i_idxs_hard_k = i_idxs_hard_k[random_order_hard]
    j_idxs_hard_k = j_idxs_hard_k[random_order_hard]
    labels_hard = torch.tensor(labels_hard[random_order_hard])
    
    i_idxs_match_k = np.arange(len(j_idxs_match_orig_list))
    num_of_matches = int(len(i_idxs_match_k)/2)
    random_order_match = [i for i,x in enumerate(j_idxs_match_orig_list) if len(x)!=0]
    random.shuffle(random_order_match)
    i_idxs_match_k = np.array(i_idxs_match_k)[random_order_match]
    i_idxs_match_k = i_idxs_match_k[:num_of_matches]
    
    j_idxs_match_orig_list = list(np.array(j_idxs_match_orig_list, dtype=object)[random_order_match])
    j_idxs_match_orig = np.array([random.choice(j_idxs_match_orig_list[i]) for i in range(num_of_matches)])
    I_j_match, G_j_match = dataset_orig.get_items_at(j_idxs_match_orig)
    
    Ii = torch.cat((I_k[i_idxs_hard_k], I_k[i_idxs_match_k]), dim=0)
    Gi = torch.cat((G_k[i_idxs_hard_k], G_k[i_idxs_match_k]), dim=0)
    
    Ij = torch.cat((I_k[j_idxs_hard_k], I_j_match), dim=0)
    Gj = torch.cat((G_k[j_idxs_hard_k], G_j_match), dim=0)
    
    labels_match = torch.tensor([True] * num_of_matches)
    labels_super_batch = torch.stack((labels_hard, labels_match)).transpose(0, 1).reshape(1,-1)[0]
    
    half_batch_size = int(batch_size/2)
    reorder_super_batch = torch.arange(Ii.shape[0]).view(2,-1).transpose(0,1).reshape(1,-1)[0]
    reorder_super_batch = list(map(list,zip_longest(*(iter(reorder_super_batch),) * half_batch_size)))
    
    num_of_none_values = (half_batch_size - (Ii.shape[0] % half_batch_size)) % half_batch_size
    if num_of_none_values != 0:
        reorder_super_batch[-1] = reorder_super_batch[-1][:-num_of_none_values]
    
    Ii_batch_list, Ij_batch_list = [], []
    Gi_batch_list, Gj_batch_list = [], []
    labels_batch_list = []
    for i in range(len(reorder_super_batch)):
        random_order = np.arange(len(reorder_super_batch[i]))
        random.shuffle(random_order)
        
        new_order = torch.tensor(reorder_super_batch[i])[random_order]
        Ii_batch_list.append(Ii[new_order])
        Ij_batch_list.append(Ij[new_order])
        Gi_batch_list.append(Gi[new_order])
        Gj_batch_list.append(Gj[new_order])
        labels_batch_list.append(labels_super_batch[new_order])
        
    return Ii_batch_list, Ij_batch_list, Gi_batch_list, Gj_batch_list, labels_batch_list


def run_groups_through_model(model, device, writer, is_train, dataset, k, batch_size, threshold=None, optimizer=None):
    if is_train:
        assert optimizer is not None, "optimizer must have value on train"
    else:
        assert threshold is not None, "threshold must have value on evalutaion"
    
    groups_size_k_list = utils.split_data_to_groups_size_k(dataset, k)
    global_steps = 0
    if is_train:
        epoch_loss = 0
    else:
        epoch_accuracy = 0
        
    for group_idx, i_idxs_match_orig in enumerate(groups_size_k_list):
        print(f"\tGroup: {group_idx+1}/{len(groups_size_k_list)}")
        group_start = time.time()
        group_steps = 0
        
        if is_train:
            group_loss = 0
        else:
            group_accuracy = 0
        
        get_idxs_in_original_set = lambda idxs: np.array(i_idxs_match_orig)[idxs]
        j_idxs_match_orig_list = dataset.calc_matches_idxs(i_idxs_match_orig)
        
        I_k, G_k = dataset.get_items_at(i_idxs_match_orig)
        matches_mat = dataset.calc_matches_mat(i_idxs_match_orig)
        
        with torch.no_grad():
            i_idxs_hard_k, j_idxs_hard_k = hard_mine(model, device, I_k, G_k, batch_size, k, matches_mat)
        
        labels_hard = dataset.calc_matches_bool(get_idxs_in_original_set(i_idxs_hard_k), 
                                                get_idxs_in_original_set(j_idxs_hard_k))
        i_j_labels_hard_k = (i_idxs_hard_k, j_idxs_hard_k, labels_hard)
        
        Ii_batch_list, Ij_batch_list, Gi_batch_list, Gj_batch_list, labels_batch_list = balance_batch(
            I_k, G_k, dataset, i_j_labels_hard_k, j_idxs_match_orig_list)
        
                
        for batch_idx in range(len(labels_batch_list)):
            Ii_batch, Ij_batch = Ii_batch_list[batch_idx], Ij_batch_list[batch_idx]
            Gi_batch, Gj_batch = Gi_batch_list[batch_idx], Gj_batch_list[batch_idx]
            labels_batch = torch.ones(Ii_batch.shape[0])
            labels_batch[labels_batch_list[batch_idx]==False] = -1

            I_batch = torch.cat((Ii_batch, Ij_batch), dim=0)
            G_batch = torch.cat((Gi_batch, Gj_batch), dim=0)
            pred_descriptors = run_model(I_batch, G_batch, device, model)
            
            half_descriptors_size = int(pred_descriptors.shape[0]/2)
            pred_descriptors_i = pred_descriptors[:half_descriptors_size]
            pred_descriptors_j = pred_descriptors[half_descriptors_size:]
            
            if is_train:
                loss = loss_func(pred_descriptors_i, pred_descriptors_j, labels_batch)
                epoch_loss += loss.item()
                group_loss += loss.item()
            else:
                d_L1 = torch.sum(torch.abs(pred_descriptors_i - pred_descriptors_j), dim=1)
                pred_labels = torch.ones(d_L1.shape).to(device='cpu', dtype=torch.float32)
                pred_labels[d_L1 > threshold] = -1
                accuracy = len(torch.where(pred_labels==labels_batch)[0])/len(pred_labels)
                epoch_accuracy += accuracy
                group_accuracy += accuracy
            
            global_steps += 1
            group_steps += 1

            if is_train:
                print(f"\t\tBatch: {batch_idx+1}/{len(labels_batch_list)}, loss: {loss.item()}")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                print(f"\t\tBatch: {batch_idx+1}/{len(labels_batch_list)}")
            
        group_end = time.time()
        if is_train:
            mean_group_loss = group_loss/group_steps
            print(f"\tGroup time: {group_end - group_start}, loss: {mean_group_loss}")
            writer.add_scalar("Group loss-train", mean_group_loss, group_idx)
        else:
            mean_group_accuracy = group_accuracy/group_steps
            print(f"\tGroup time: {group_end - group_start}, accuracy: {mean_group_accuracy}")
            writer.add_scalar("Group Evaluation", mean_group_accuracy, group_idx)
    
    if is_train:
        return epoch_loss/global_steps
    return epoch_accuracy/global_steps


def train_net(model, dataset_dict, batch_size, epochs, optimizer, loss_func, device, k, threshold):
    writer = SummaryWriter()
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
        model.train()
        mean_loss = run_groups_through_model(model, device, writer, True, dataset_dict['train'], k, batch_size, optimizer=optimizer)
        with torch.no_grad():
            model.eval()
            val_accuracy = run_groups_through_model(model, device, writer, False, dataset_dict['val'], k, batch_size, threshold=threshold)
        
        writer.add_scalar("Loss-train", mean_loss, epoch)
        writer.add_scalar("Evaluation", val_accuracy, epoch)
        
        print(f"Validate Accuracy: {val_accuracy}")

    writer.close()
    return model


if __name__ == "__main__":
    
    data_dir = r"/media/idansheffer/multi_view_hd/DeepLearning/data2"
    weights_dir = r"/media/idansheffer/multi_view_hd/DeepLearning/weights"
    dataset_path = os.path.join(data_dir,'dataset_2_timestamps.csv')
    structure_time_span = 2
    match_threshold_m = 5
    batch_size = 12
    epochs = 10
    k = 300
    alpha = 1
    m = 0.5
    threshold = 1
    
    # validate_lat_long_radius_m = (51.76065874460691, -1.2674376580131264, 70)
    validate_lat_long_radius_m = (51.76065874460691, -1.2674376580131264, 65)
    # train_lat_long_radius_m = (51.75785736610417, -1.256094136690952, 100)
    train_lat_long_radius_m = (51.75785736610417, -1.256094136690952, 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_model = CompoundNet()
    net_model = net_model.to(device=device)
    loss_func = MarginBasedLoss(alpha=alpha, m=m)
    optimizer = optim.SGD(net_model.parameters(), lr=0.01, weight_decay=1e-8)

    dataset_dict = init_data(data_dir, dataset_path, structure_time_span, match_threshold_m, validate_lat_long_radius_m, train_lat_long_radius_m)
    # matches_list = dataset_dict['train'].calc_matches_idxs([100])
    # dataset_dict['train'].get_items_at([100,1325,2180,3560,10000])
    net_model = train_net(net_model, dataset_dict, batch_size, epochs, optimizer, loss_func, device, k, threshold)

    weights_path = os.path.join(weights_dir, f"weights_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pt")
    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)

    torch.save(net_model.state_dict(), weights_path)

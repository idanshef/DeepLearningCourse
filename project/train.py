import os
from datetime import datetime
import torch
import torch.nn as nn
from torch import optim
from dataset import RobotCarDataset
from net_models import CompoundNet, MarginBasedLoss
import utils
import numpy as np
import random


def init_data(data_dir, dataset_csv, val_percent, structure_time_span, match_threshold):
    dataset_df = utils.create_dataset_df(data_dir, structure_time_span, dataset_csv)
    cameras = utils.load_cameras(data_dir)
    
    train_idxs, val_idxs = utils.split_idxs_to_train_val_idxs(dataset_df, val_percent/100)
    select_subset_df = lambda set_idxs: dataset_df.iloc[set_idxs, :].reset_index()
    
    return {'val': RobotCarDataset(select_subset_df(val_idxs), cameras, match_threshold),
            'train': RobotCarDataset(select_subset_df(train_idxs), cameras, match_threshold)}


def hard_mine(model, dataset_k, batch_size, k):
    pred_descriptors_k = None
    for i in range(int(k/batch_size)):
        if i != int(k/batch_size)-1:
            curr_samples = list(np.arange(i*batch_size, (i+1)*batch_size))
        else:
            curr_samples = list(np.arange(i*batch_size, k))
            
        dataset_i = RobotCarDataset.subset_of_dataset(dataset_k, curr_samples)
        I, G = dataset_i.get_items()
        
        I = I.to(device=device, dtype=torch.float32)
        G = G.to(device=device, dtype=torch.float32)
        
        pred_descriptors_i = model(I, G)
        pred_descriptors_i = pred_descriptors_i.to(device='cpu', dtype=torch.float32)
        I = I.to(device='cpu', dtype=torch.float32)
        G = G.to(device='cpu', dtype=torch.float32)
        
        if pred_descriptors_k is None:
            pred_descriptors_k = pred_descriptors_i.clone()
        else:
            pred_descriptors_k = torch.cat((pred_descriptors_k, pred_descriptors_i), dim=0)
    
    descriptor_size = pred_descriptors_k.shape[1]
    repeat_pred = pred_descriptors_k.repeat(1,k).view(-1,descriptor_size)
    d_L1 = torch.sum(torch.abs(repeat_pred - pred_descriptors_k.repeat(k,1)), dim=1).view(k,-1)
    d_L1[torch.tril(torch.ones(d_L1.shape),diagonal=-1)==0] = torch.finfo(torch.float32).max
    
    min_d = torch.min(d_L1, dim=0)
    top_k = torch.topk(-min_d.values, k)
    
    i_idxs = min_d.indices[top_k.indices]
    j_idxs = top_k.indices
    labels = dataset_k.calc_matches_bool(i_idxs, j_idxs)

    return i_idxs, j_idxs, labels


def balance_batch(i_idxs, j_idxs, labels):
    pos_labels = np.where(labels == True)
    neg_labels = np.where(labels == False)
    pos_i_idx, neg_i_idx = i_idxs[pos_labels], i_idxs[neg_labels]
    pos_j_idx, neg_j_idx = j_idxs[pos_labels], j_idxs[neg_labels]
    
    number_of_batches = int(2*k/batch_size) + (0 if k % batch_size == 0 else 1)
    
    i_batch_idxs_list = [[] for i in range(number_of_batches)]
    j_batch_idxs_list = [[] for i in range(number_of_batches)]
    labels_batch_list = [[] for i in range(number_of_batches)]
    
    list_idx, val_idx = 0, 0
    while len(pos_i_idx) != val_idx:
        i_batch_idxs_list[list_idx].append(pos_i_idx[val_idx])
        j_batch_idxs_list[list_idx].append(pos_j_idx[val_idx])
        labels_batch_list[list_idx].append(1)
        list_idx = (list_idx+1) % number_of_batches
        val_idx += 1
    
    list_idx, val_idx = 0, 0
    while len(neg_i_idx) != val_idx:
        reverse_list_idx = number_of_batches - 1 - list_idx
        i_batch_idxs_list[reverse_list_idx].append(neg_i_idx[val_idx])
        j_batch_idxs_list[reverse_list_idx].append(neg_j_idx[val_idx])
        labels_batch_list[reverse_list_idx].append(-1)
        list_idx = (list_idx+1) % number_of_batches
        val_idx += 1
    
    for i in range(number_of_batches):
        list_size = len(i_batch_idxs_list[i])
        rand_order = random.sample(list(np.arange(list_size)), list_size)
        i_batch_idxs_list[i] = [i_batch_idxs_list[i][val] for val in rand_order]
        j_batch_idxs_list[i] = [j_batch_idxs_list[i][val] for val in rand_order]
        labels_batch_list[i] = [labels_batch_list[i][val] for val in rand_order]
    
    return i_batch_idxs_list, j_batch_idxs_list, labels_batch_list


def train_net(model, dataset_dict, batch_size, epochs, optimizer, loss_func, device, n, k):
    
    for epoch in range(epochs):
        print(f"Epoch: {epoch}/{epochs}")

        model.train()
        epoch_loss = 0
        
        n_idxs_groups_size_k = utils.split_data_to_n_groups_size_k(dataset_dict['train'], n, k)
        
        for group_idx, groups_idxs in enumerate(n_idxs_groups_size_k):
            print(f"\tGroup: {group_idx}/{len(n_idxs_groups_size_k)}")
            optimizer.zero_grad()
            
            dataset_k = RobotCarDataset.subset_of_dataset(dataset_dict['train'], groups_idxs)
            with torch.no_grad():
                i_idxs, j_idxs, labels = hard_mine(model, dataset_k, batch_size, k)
            i_batch_idxs_list, j_batch_idxs_list, labels_batch_list = balance_batch(i_idxs, j_idxs, labels)
            
            for batch_idx, batch_tuple in enumerate(zip(i_batch_idxs_list, j_batch_idxs_list, labels_batch_list)):
                print(f"\t\tBatch: {batch_idx}/{len(i_batch_idxs_list)}")
                i_batch_idxs, j_batch_idxs, labels_batch = batch_tuple
                
                dataset_i = RobotCarDataset.subset_of_dataset(dataset_k, i_batch_idxs)
                dataset_j = RobotCarDataset.subset_of_dataset(dataset_k, j_batch_idxs)
                
                Ii, Gi = dataset_i.get_items()
                Ij, Gj = dataset_j.get_items()
                
                I = torch.cat((Ii, Ij), dim=0).to(device=device, dtype=torch.float32)
                G = torch.cat((Gi, Gj), dim=0).to(device=device, dtype=torch.float32)
                labels_batch = torch.Tensor(labels_batch).to(device=device, dtype=torch.float32)
                
                pred_descriptors = model(I, G)
                del I, G
                
                pred_descriptors_i = pred_descriptors[:int(pred_descriptors.shape[0]/2)]
                pred_descriptors_j = pred_descriptors[int(pred_descriptors.shape[0]/2):]
                loss = loss_func(pred_descriptors_i, pred_descriptors_j, labels_batch)
                epoch_loss += loss.item()
            
                loss.backward()
                optimizer.step()

        #     global_step += 1
        #     print(f"Batch loading time: {end-start}")
        #     print("Loss: {0}".format(loss.item()))
        #     print('done batch {0}'.format(global_step))

        # TODO: fix tensorboard
        # writer.add_scalar("Loss-train", epoch_loss / len(data_loaded['train']), global_step)

    # writer.close()
    return model


if __name__ == "__main__":
    
    data_dir = r"/media/idansheffer/multi_view_hd/DeepLearning/data2"
    dataset_path = os.path.join(data_dir,'dataset_2_timestamps.csv')
    structure_time_span = 2
    match_threshold = 5
    val_percent = 10
    batch_size = 12
    epochs = 20
    n = 100
    k = 60
    alpha = 1e-3
    m = 0.5e-3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_model = CompoundNet()
    net_model = net_model.to(device=device)
    loss_func = MarginBasedLoss(alpha=alpha, m=m)
    nn.MarginRankingLoss
    optimizer = optim.SGD(net_model.parameters(), lr=0.01, weight_decay=1e-8)

    dataset_dict = init_data(data_dir, dataset_path, val_percent, structure_time_span, match_threshold)
    net_model = train_net(net_model, dataset_dict, batch_size, epochs, optimizer, loss_func, device, n, k)

    weights_path = f"weights_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pt"
    if not os.path.isdir(net_weights_dir):
        os.makedirs(net_weights_dir)

    torch.save(net_model.state_dict(), weights_path)

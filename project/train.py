import os
from datetime import datetime
import torch
import torch.nn as nn
from torch import optim
from dataset import RobotCarDataset
import utils
import numpy as np


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
        
        with torch.no_grad():
            pred_descriptors_i = model(I, G)
        pred_descriptors_i = pred_descriptors_i.to(device='cpu', dtype=torch.float32)
        I = I.to(device='cpu', dtype=torch.float32)
        G = G.to(device='cpu', dtype=torch.float32)
        
        if pred_descriptors_k is None:
            pred_descriptors_k = pred_descriptors_i.detach().clone()
        else:
            pred_descriptors_k = torch.cat((pred_descriptors_k, pred_descriptors_i), dim=0)
    
    descriptor_size = pred_descriptors_k.shape[1]
    repeat_pred = pred_descriptors_k.repeat(1,k).view(-1,descriptor_size)
    d_L1 = torch.sum(torch.abs(repeat_pred - pred_descriptors_k.repeat(k,1)), dim=1).view(k,-1)
    d_L1[torch.tril(torch.ones(d_L1.shape),diagonal=-1)==0] = torch.finfo(torch.float32).max
    
    min_d = torch.min(d_L1, dim=0)
    top_p = torch.topk(-min_d.values, p)
    
    i_idxs = min_d.indices[top_p.indices]
    j_idxs = top_p.indices
    labels = dataset_k.calc_matches_bool(i_idxs, j_idxs)

    return i_idxs, j_idxs, labels

def train_net(model, dataset_dict, batch_size, epochs, optimizer, loss_func, device, n, k, p):
    
    for epoch in range(epochs):
        print(f"Epoch: {epoch}/{epochs}")

        model.train()
        epoch_loss = 0
        
        n_idxs_groups_size_k = utils.split_data_to_n_groups_size_k(dataset_dict['train'], n, k)
        
        for groups_idxs in n_idxs_groups_size_k:
            optimizer.zero_grad()
            
            dataset_k = RobotCarDataset.subset_of_dataset(dataset_dict['train'], groups_idxs)
            i_idxs, j_idxs, labels = hard_mine(model, dataset_k, batch_size, k)
            
            get_index_by_batch = lambda i: i*int(batch_size/2)
            get_idxs_list = lambda from_list: [(from_list[get_index_by_batch(i):] if i==int(p/batch_size) 
                                                else from_list[get_index_by_batch(i):get_index_by_batch(i+1)]) 
                                               for i in range(int(p/batch_size)+1)]
            i_batch_idxs_list = get_idxs_list(i_idxs)
            j_batch_idxs_list = get_idxs_list(j_idxs)
            
            pred_descriptors_p = None
            for i_batch_idxs, j_batch_idxs in zip(i_batch_idxs_list, j_batch_idxs_list):
                dataset_i = RobotCarDataset.subset_of_dataset(dataset_k, i_batch_idxs)
                dataset_j = RobotCarDataset.subset_of_dataset(dataset_k, j_batch_idxs)
                
                Ii, Gi = dataset_i.get_items()
                Ij, Gj = dataset_j.get_items()
                
                I = torch.cat((Ii, Ij), dim=0).to(device=device, dtype=torch.float32)
                G = torch.cat((Gi, Gj), dim=0).to(device=device, dtype=torch.float32)
                
                pred_descriptors = model(I, G).to(device='cpu', dtype=torch.float32)
                I = I.to(device='cpu', dtype=torch.float32)
                G = I.to(device='cpu', dtype=torch.float32)
                
                if pred_descriptors_p is None:
                    pred_descriptors_p = pred_descriptors.detach().clone()
                else:
                    pred_descriptors_p = torch.cat((pred_descriptors_p, pred_descriptors), dim=0)
            
            loss = loss_func(pred_descriptors_p[:int(pred_descriptors_p.shape[0]/2)], 
                             pred_descriptors_p[int(pred_descriptors_p.shape[0]/2):], 
                             labels)
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
    structure_time_span=2
    match_threshold=5
    batch_size = 10
    val_percent=10
    epochs = 20
    n = 100
    k = 50
    p = 12
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_model = CompoundNet()
    net_model = net_model.to(device=device)
    loss_func = nn.MarginRankingLoss()
    optimizer = optim.SGD(net_model.parameters(), lr=0.01, weight_decay=1e-8)

    dataset_dict = init_data(data_dir, dataset_path, val_percent, structure_time_span, match_threshold)
    net_model = train_net(net_model, dataset_dict, batch_size, epochs, optimizer, loss_func, device, n, k, p)

    # weights_path = f"weights_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pt"
    # if not os.path.isdir(net_weights_dir):
    #     os.makedirs(net_weights_dir)

    # torch.save(net_model.state_dict(), weights_path)

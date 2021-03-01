import os
from datetime import datetime
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from net_models import *
from dataset import RobotCarDataset
import time


def init_data(data_dir, dataset_csv, val_percent, structure_time_span, match_threshold):
    dataset = create_dataset(data_dir, structure_time_span, dataset_csv)
    cameras = load_cameras(data_dir)
    
    train_idxs, val_idxs = utils.split_idxs_to_train_val_idxs(dataset, val_percent/100)
    select_dataset = lambda set_idxs: dataset.iloc[set_idxs, :].reset_index()
    
    return {'val': RobotCarDataset(select_dataset(val_idxs), cameras, match_threshold),
            'train': RobotCarDataset(select_dataset(train_idxs), cameras, match_threshold)}


def train_net(model, dataset_dict, batch_size, epochs, optimizer, loss_func, device, n, k):
    # global_step = 0
    for epoch in range(epochs):
        print(f"Epoch: {epoch}/{epochs}")

        model.train()
        epoch_loss = 0
        
        dataset_k = utils.select_samples(dataset_dict['train'], k)
        
        pred_descriptors_k = None
        
        for i in range(floor(k/batch_size)):
            curr_samples = samples_idxs[i*batch_size:(i+1)*batch_size] if i!=floor(k/batch_size)-1 else samples_idxs[i*batch_size:]
            I_curr = I_k[curr_samples].to(device=device, dtype=torch.float32)
            G_curr = G_k[curr_samples].to(device=device, dtype=torch.float32)
            
            with torch.no_grad():
                pred_descriptors_curr = model(I_curr, G_curr)
            pred_descriptors_curr = pred_descriptors_curr.to(device='cpu', dtype=torch.float32)
            
            if pred_descriptors_k is None:
                pred_descriptors_k = pred_descriptors_curr.detach().clone()
            else:
                pred_descriptors_k = torch.cat((pred_descriptors_k, pred_descriptors_curr), dim=0)
            
        descriptor_size = pred_descriptors_k.shape[0]
        repeat_pred = pred_descriptors_k.view(-1,1).repeat(1,k).view(descriptor_size,-1)
        d_L1 = torch.sum(torch.abs(repeat_pred - pred_descriptors_k.repeat(1,k)), dim=0).view(k,-1)
        d_L1[torch.tril(torch.ones(d_L1.shape),diagonal=-1)==0] = torch.finfo(torch.float32).max
        
        min_d_indices = torch.min(d_L1, dim=0)
        
        Ij, Gj = Ii[min_d_indices], Gj[min_d_indices]
        
            
            
            
        
        # start=time.time()
        # for batch in data['train']:
        #     end=time.time()
            
        #     optimizer.zero_grad()

        #     Ii, Gi = batch['Ii'], batch['Gi']
        #     Ij, Gj = batch['Ij'], batch['Gj']
        #     labels = batch['is_match']
            
        #     Ii, Gi = Ii.to(device=device, dtype=torch.float32), Gi.to(device=device, dtype=torch.float32)
        #     with torch.no_grad():
        #         Xi_predicted_descriptor = model(Ii, Gi)
            
        #     Ii_pos, Gi_pos = Ii[labels[:,0]==1], Gi[labels[:,0]==1]
        #     Ij_pos, Gj_pos = Ij[labels[:,0]==1], Gj[labels[:,0]==1]
            
        #     Ii_neg, Gi_neg = Ii[labels[:,0]==-1], Gi[labels[:,0]==-1]
        #     Ij_neg, Gj_neg = Ij[labels[:,0]==-1], Gj[labels[:,0]==-1]
            
        #     Ij_neg = Ij_neg.reshape(-1, *Ij_neg.shape[2:])
        #     Gj_neg = Gj_neg.reshape(-1, *Gj_neg.shape[2:])
            
        #     Ij_neg, Gj_neg = Ij_neg.to(device=device, dtype=torch.float32), Gj_neg.to(device=device, dtype=torch.float32)
        #     with torch.no_grad():
        #         Xj_neg_pred_desc = model(Ij_neg, Gj_neg)
                
        #     Ii_neg, Gi_neg = Ii_neg.to(device=device, dtype=torch.float32), Gi_neg.to(device=device, dtype=torch.float32)
        #     with torch.no_grad():
        #         Xi_neg_pred_desc = model(Ii_neg, Gi_neg)
                
        #         super_batch_size = Xj_neg_pred_desc.shape[0] / Xi_neg_pred_desc.shape[0]
        #         descriptor_size = Xi_neg_pred_desc.shape[1]
                
        #         print("test")
                
        #     # Ij = Ij.reshape(-1, *Ij.shape[2:])
        #     # Gj = Gj.reshape(-1, *Gj.shape[2:])
            
        #     # Ij = Ij.permute(2, 3, 4, 0, 1)
        #     # Ij = Ij.contiguous().view(Ij.shape[0], Ij.shape[1], Ij.shape[2], -1)
        #     # Ij = Ij.permute(3, 0, 1, 2)
            
        #     # Gj = Gj.permute(2, 3, 4, 5, 0, 1)
        #     # Gj = Gj.contiguous().view(Gj.shape[0], Gj.shape[1], Gj.shape[2], Gj.shape[3], -1)
        #     # Gj = Gj.permute(4, 0, 1, 2, 3)

        #     Ii, Gi = Ii.to(device=device, dtype=torch.float32), Gi.to(device=device, dtype=torch.float32)
        #     Ij, Gj = Ij.to(device=device, dtype=torch.float32), Gj.to(device=device, dtype=torch.float32)
        #     labels = labels.to(device=device, dtype=torch.float32)
            
        #     Xi_predicted_descriptor = model(Ii, Gi)
        #     with torch.no_grad(): # TODO: fix hard-mine
        #         Xi_neg_predicted_descriptor = model(Ii_neg, Gi_neg)
        #         Xj_neg_predicted_descriptor = model(Ij_neg, Gj_neg)
                
        #         super_batch_size = Xj_neg_predicted_descriptor.shape[0] / Xi_neg_predicted_descriptor.shape[0]
        #         descriptor_size = Xi_neg_predicted_descriptor.shape[1]
        #         repeat_Xi = Xi_neg_predicted_descriptor.view(-1,1).repeat(1,super_batch_size).view(descriptor_size,-1)
                
        #         assert(Xj_neg_predicted_descriptor.shape == repeat_Xi.shape), "Tensors shapes doesn't match"
        #         L1_distance = abs(Xj_neg_predicted_descriptor - repeat_Xi).sum(axis=0)
        #         min_Xj_idxs = L1_distance.view(super_batch_size, -1).argmin(dim=1)
                
        #     # Xj_predicted_descriptor = model(Ij, Gj)
            
        #     loss = loss_func(Xi_predicted_descriptor, Xj_predicted_descriptor, labels)
        #     epoch_loss += loss.item()

        #     loss.backward()
        #     optimizer.step()

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
    dataset_path = os.path.join(data_dir,'dataset.csv')
    structure_time_span=2
    match_threshold=5
    batch_size = 10
    val_percent=10
    epochs = 20
    k=50
    n=5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_model = CompoundNet()
    net_model = net_model.to(device=device)
    loss_func = nn.MarginRankingLoss()
    optimizer = optim.SGD(net_model.parameters(), lr=0.01, weight_decay=1e-8)

    dataset_dict = init_data(data_dir, dataset_path, val_percent, structure_time_span, match_threshold)
    net_model = train_net(net_model, dataset_dict, batch_size, epochs, optimizer, loss_func, device)

    # weights_path = f"weights_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pt"
    # if not os.path.isdir(net_weights_dir):
    #     os.makedirs(net_weights_dir)

    # torch.save(net_model.state_dict(), weights_path)

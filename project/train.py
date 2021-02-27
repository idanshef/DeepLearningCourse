import os
from datetime import datetime
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from net_models import *
from dataset import RobotCarDataset


def split_dataset(dataset, val_percent):
    assert 0. <= val_percent <= 1., f"Validation percent must be between [0,1]. Got {val_percent}"

    num_val = int(len(dataset) * val_percent)
    num_train = len(dataset) - num_val

    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])
    val_dataset.dataset.color_jitter = None ############

    return train_dataset, val_dataset


def load_data(data_dir, batch_size, super_batch_size, val_percent=10, structure_time_span=10, match_threshold=5, dataset_csv=None):
    
    dataset = RobotCarDataset(data_dir, structure_time_span, match_threshold, super_batch_size, dataset_csv)
    train_dataset, val_dataset = split_dataset(dataset, val_percent / 100)

    data_loaders = dict()
    data_loaders['train'] = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4,
                                       pin_memory=True)
    data_loaders['val'] = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    return data_loaders


def train_net(model, data, epochs, optimizer, loss_func, device):
    global_step = 0

    for epoch in range(epochs):
        print(f"Epoch: {epoch}/{epochs}")

        model.train()
        epoch_loss = 0
        
        for batch in data['train']:
            optimizer.zero_grad()

            Ii, Gi = batch['Ii'], batch['Gi']
            Ij, Gj = batch['Ij'], batch['Gj']
            labels = batch['is_match']
            
            pos_Ii, pos_Gi = Ii[labels == 1], Gi[labels == 1]
            pos_Ij, pos_Gj = Ij[labels == 1], Gj[labels == 1]
            
            Ij = Ij.reshape(-1, *Ij.shape[2:])
            Gj = Gj.reshape(-1, *Gj.shape[2:])
            
            # Ij = Ij.permute(2, 3, 4, 0, 1)
            # Ij = Ij.contiguous().view(Ij.shape[0], Ij.shape[1], Ij.shape[2], -1)
            # Ij = Ij.permute(3, 0, 1, 2)
            
            # Gj = Gj.permute(2, 3, 4, 5, 0, 1)
            # Gj = Gj.contiguous().view(Gj.shape[0], Gj.shape[1], Gj.shape[2], Gj.shape[3], -1)
            # Gj = Gj.permute(4, 0, 1, 2, 3)

            Ii, Gi = Ii.to(device=device, dtype=torch.float32), Gi.to(device=device, dtype=torch.float32)
            Ij, Gj = Ij.to(device=device, dtype=torch.float32), Gj.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.int8)
            
            Xi_predicted_descriptor = model(Ii, Gi)
            with torch.no_grad():
                Xi_neg_predicted_descriptor = model(Ii_neg, Gi_neg)
                Xj_neg_predicted_descriptor = model(Ij_neg, Gj_neg)
                
                super_batch_size = Xj_neg_predicted_descriptor.shape[0] / Xi_neg_predicted_descriptor.shape[0]
                descriptor_size = Xi_neg_predicted_descriptor.shape[1]
                repeat_Xi = Xi_neg_predicted_descriptor.view(-1,1).repeat(1,super_batch_size).view(descriptor_size,-1)
                
                assert(Xj_neg_predicted_descriptor.shape == repeat_Xi.shape), "Tensors shapes doesn't match"
                L1_distance = abs(Xj_neg_predicted_descriptor - repeat_Xi).sum(axis=0)
                min_Xj_idxs = L1_distance.view(super_batch_size, -1).argmin(dim=1)
                
            # Xj_predicted_descriptor = model(Ij, Gj)
            
            loss = loss_func(Xi_predicted_descriptor, Xj_predicted_descriptor, labels)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            global_step += 1
            print("Loss: {0}".format(loss.item()))
            print('done batch {0}'.format(global_step))

        writer.add_scalar("Loss-train", epoch_loss / len(data_loaded['train']), global_step)

    writer.close()
    return model


if __name__ == "__main__":
    
    data_dir = r"/media/idansheffer/multi_view_hd/DeepLearning/data"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    super_batch_size = 50
    batch_size = 10
    data_loaders = load_data(data_dir, batch_size, super_batch_size, dataset_csv=os.path.join(data_dir,'dataset.csv'))

    net_model = CompoundNet()
    net_model = net_model.to(device=device)

    loss_func = nn.MarginRankingLoss()
    optimizer = optim.SGD(net_model.parameters(), lr=0.01, weight_decay=1e-8)

    net_model = train_net(net_model, data_loaders, epochs=20, optimizer=optimizer, loss_func=loss_func, device=device)

    # weights_path = f"weights_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pt"
    # if not os.path.isdir(net_weights_dir):
    #     os.makedirs(net_weights_dir)

    # torch.save(net_model.state_dict(), weights_path)

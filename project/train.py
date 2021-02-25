import os
from datetime import datetime
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from net_models import *
from dataset import RobotCarDataset
from robotcar_dataset_sdk.camera_model import CameraModel
from robotcar_dataset_sdk.image import load_image
import pandas as pd


def split_dataset(dataset, val_percent):
    assert 0. <= val_percent <= 1., f"Validation percent must be between [0,1]. Got {val_percent}"

    num_val = int(len(dataset) * val_percent)
    num_train = len(dataset) - num_val

    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])
    val_dataset.dataset.color_jitter = None ############

    return train_dataset, val_dataset


def load_data(data_dir, batch_size, val_percent=10, structure_time_span=10, match_threshold=5):
    
    dataset = RobotCarDataset(data_dir, structure_time_span, match_threshold)
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

            Ii, Gi = Ii.to(device=device, dtype=torch.float32), Gi.to(device=device, dtype=torch.float32)
            Ij, Gj = Ij.to(device=device, dtype=torch.float32), Gj.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.int8)
            
            Xi_predicted_descriptor = model(Ii, Gi)
            Xj_predicted_descriptor = model(Ij, Gj)
            
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
    
    data_dir = r"/media/idansheffer/multi_view_hd/DeepLearning/data1"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_loaders = load_data(data_dir, 10, 10)

    net_model = CompoundNet()
    net_model = net_model.to(device=device)

    loss_func = nn.MarginRankingLoss()
    optimizer = optim.SGD(net_model.parameters(), lr=0.01, weight_decay=1e-8)

    net_model = train_net(net_model, data_loaders, epochs=20, optimizer=optimizer, loss_func=loss_func, device=device)

    # weights_path = f"weights_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pt"
    # if not os.path.isdir(net_weights_dir):
    #     os.makedirs(net_weights_dir)

    # torch.save(net_model.state_dict(), weights_path)

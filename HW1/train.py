import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from utils import *
from dataset import FashionMNISTDataSet
from lenet5 import LeNet5


batch_size = 10


def split_dataset(dataset, val_percent):
    assert 0. <= val_percent <= 1., "Validation percent must be between [0,1]. Got {}".format(val_percent)

    num_val = int(len(dataset) * val_percent)
    num_train = len(dataset) - num_val

    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])
    # val_dataset.dataset.color_jitter = None

    return train_dataset, val_dataset


def load_train_data(train_images_path, train_labels_path, val_percent=10):
    dataset = FashionMNISTDataSet(train_images_path, train_labels_path)

    train_dataset, val_dataset = split_dataset(dataset, val_percent / 100)

    data_loaders = dict()
    data_loaders['train'] = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4,
                                       pin_memory=True)
    data_loaders['val'] = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=4,
                                     pin_memory=True)

    return data_loaders


def train_net(model, data_loaded, epochs, optimizer, loss_func, device):
    global_step = 0

    writer = SummaryWriter()
    epoch_idx = 0
    for epoch in range(epochs):
        print("Epoch: {0}/{1}".format(epoch_idx, epochs))
        epoch_idx += 1

        model.train()
        epoch_loss = 0
        for batch in data_loaded['train']:
            optimizer.zero_grad()

            images, labels = batch['image'], batch['label']

            images = images.to(device=device, dtype=torch.float)
            labels = labels.to(device=device, dtype=torch.long)

            predicted_labels = model(images)
            loss = loss_func(predicted_labels, labels)
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = LeNet5()
    net = net.to(device=device)

    images_path, labels_path = download_train_data(fashion_data_dir)
    data = load_train_data(images_path, labels_path)

    optimizer = optim.SGD(net.parameters(), weight_decay=1e-8, lr=0.05)
    loss_func = torch.nn.CrossEntropyLoss()

    net = train_net(net, data, epochs=10, optimizer=optimizer, loss_func=loss_func, device=device)

    weights_path = weight_decay_net_path
    if not os.path.isdir(net_weights_dir):
        os.makedirs(net_weights_dir)

    torch.save(net.state_dict(), weights_path)

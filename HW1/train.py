import os
import torch
from torch.utils.data import DataLoader, random_split
# from torch.utils.tensorboard import SummaryWriter
from torch import optim
from dataset import FashionMNISTDataSet
from lenet5 import LeNet5

fashion_data_dir = r"C:\Users\isheffer\OneDrive - Intel Corporation\Desktop\university\DeepLearning\DeepLearningCourse\HW1\data"
batch_size = 16


def load_train_data(data_dir):
    images_path, labels_path = os.path.join(data_dir, "train-images-idx3-ubyte"), \
                               os.path.join(data_dir, "train-labels-idx1-ubyte")
    dataset = FashionMNISTDataSet(images_path, labels_path)
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)

    # train_dataset, val_dataset = split_dataset(dataset, val_percent / 100)
    #
    # data_loaders = dict()
    # data_loaders['train'] = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4,
    #                                    pin_memory=True)
    # data_loaders['val'] = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=4,
    #                                  pin_memory=True)

    return data_loader


def train_net(model, data_loaded, epochs, optimizer, loss_func, device):
    global_step = 0

    # writer = SummaryWriter()
    epoch_idx = 0
    for epoch in range(epochs):
        print("Epoch: {0}/{1}".format(epoch_idx, epochs))
        epoch_idx += 1

        model.train()
        epoch_loss = 0
        for batch in data_loaded:
            optimizer.zero_grad()

            images = batch['image']
            labels = batch['label']

            images = images.to(device=device, dtype=torch.float)
            labels = labels.to(device=device, dtype=torch.float)

            predicted_labels = model(images)
            loss = loss_func(predicted_labels, labels)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            global_step += 1
            print("Loss: {0}".format(loss.item()))
            print('done batch {0}'.format(global_step))

        # writer.add_scalar("Loss-train", epoch_loss / len(data_loaded['train']), global_step)

    # writer.close()
    return model


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = LeNet5()
    net = net.to(device=device)

    data = load_train_data(fashion_data_dir)

    optimizer = optim.SGD(net.parameters(), weight_decay=1e-8, lr=0.05)
    loss_func = torch.nn.CrossEntropyLoss()

    net = train_net(net, data, epochs=10, optimizer=optimizer, loss_func=loss_func, device=device)

    weights_dir = r"C:\Users\isheffer\OneDrive - Intel Corporation\Desktop\university\DeepLearning\HW1\net_weights"
    weights_file_name = "NormalLeNet5.pt"
    weights_path = os.path.join(weights_dir, weights_file_name)
    torch.save(net.state_dict(), weights_path)

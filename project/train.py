import os
from datetime import datetime
import torch
import torch.nn as nn
from torch import optim
from net_models import *
from robotcar_dataset_sdk.camera_model import CameraModel
from robotcar_dataset_sdk.image import load_image
import cv2 as cv

net_weights_dir = "/path/to/weights/dir"


def load_data(train_images_path, train_labels_path, val_percent=10):
    # dataset = FashionMNISTDataSet(train_images_path, train_labels_path)

    # train_dataset, val_dataset = split_dataset(dataset, val_percent / 100)

    # data_loaders = dict()
    # data_loaders['train'] = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4,
    #                                    pin_memory=True)
    # data_loaders['val'] = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=4,
    #                                  pin_memory=True)

    # return data_loaders
    return


def train_net(model, data_loaded, epochs, optimizer, loss_func, device):
    # global_step = 0

    # epoch_idx = 0
    # for epoch in range(epochs):
    #     print(f"Epoch: {epoch_idx}/{epochs}")
    #     epoch_idx += 1

    #     model.train()
    #     epoch_loss = 0
    #     for batch in data_loaded['train']:
    #         optimizer.zero_grad()

    #         images, labels = batch['image'], batch['label']

    #         images = images.to(device=device, dtype=torch.float)
    #         labels = labels.to(device=device, dtype=torch.long)

    #         predicted_labels = model(images)
    #         loss = loss_func(predicted_labels, labels)
    #         epoch_loss += loss.item()

    #         loss.backward()
    #         optimizer.step()

    #         global_step += 1
    #         print("Loss: {0}".format(loss.item()))
    #         print('done batch {0}'.format(global_step))

    #     writer.add_scalar("Loss-train", epoch_loss / len(data_loaded['train']), global_step)

    # writer.close()
    # return model
    return


if __name__ == "__main__":
    


    # img_path = r"C:\Users\isheffer\OneDrive - Intel Corporation\Desktop\university\DeepLearning\Project\sample\mono_rear\1418381801212730.png"
    # img_dir = os.path.dirname(img_path)
    # models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    # camera_model = CameraModel(models_dir, img_dir)
    # cv.imshow('img', load_image(img_path, camera_model))
    # cv.waitKey(0)


    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data = load_data()

    # net_model = CompoundNet()
    # net_model = net_model.to(device=device)

    # loss_func = nn.MarginRankingLoss()
    # optimizer = optim.SGD(net_model.parameters(), lr=0.01, weight_decay=1e-8)

    # net_model = train_net(net_model, data, epochs=20, optimizer=optimizer, loss_func=loss_func, device=device)

    # weights_path = f"weights_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pt"
    # if not os.path.isdir(net_weights_dir):
    #     os.makedirs(net_weights_dir)

    # torch.save(net_model.state_dict(), weights_path)

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, n_channels=1):
        super().__init__()
        self.n_channels = n_channels

        self.conv1 = ConvBlock(in_channels=n_channels, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.pooling1 = PoolingBlock(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.pooling2 = PoolingBlock(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.activation = nn.Tanh()
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=10)
        self.dropout = nn.Dropout(p=0.5, inplace=True)
        self.softmax = nn.Softmax()
        self.batch_norm = nn.BatchNorm1d()
        self.batch_norm = nn.BatchNorm2d(num_features=)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pooling1(x)
        x = self.conv2(x)
        x = self.pooling2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.softmax(x)  # numerically it is better to output raw logits and use the BCEwithLogitsLoss


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv_block(x)


class PoolingBlock(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.pooling_block = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.pooling_block(x)

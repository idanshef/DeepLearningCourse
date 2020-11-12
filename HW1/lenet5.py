import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, n_channels=3):
        super().__init__()
        self.n_channels = n_channels

        self.conv1 = ConvBlock(in_channels=n_channels, out_channels=6, kernel_size=5, stride=1)
        self.pooling1 = PoolingBlock(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.pooling2 = PoolingBlock(kernel_size=2, stride=2)
        self.fc1 = FCBlock(in_channels=400, out_channels=120)
        self.fc2 = FCBlock(in_channels=120, out_channels=84)
        self.out = FCBlock(in_channels=84, out_channels=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pooling1(x)
        x = self.conv2(x)
        x = self.pooling2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)


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
        self.pooling_block = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, stride=stride),
            nn.Tanh()
        )


class FCBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Softmax()
        )

    def forward(self, x):
        return self.conv(x)

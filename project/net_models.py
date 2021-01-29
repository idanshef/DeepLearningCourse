import torch
import torch.nn as nn


class VisualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_section = nn.Sequential(
            ConvBlock2D(in_channels=1, out_channels=64),
            ConvBlock2D(in_channels=64, out_channels=64),
            ConvBlock2D(in_channels=64, out_channels=64),
            ConvBlock2D(in_channels=64, out_channels=64),
            ConvBlock2D(in_channels=64, out_channels=64),
            ConvBlock2D(in_channels=64, out_channels=64),
            ConvBlock2D(in_channels=64, out_channels=128),
            ConvBlock2D(in_channels=128, out_channels=128),
            ConvBlock2D(in_channels=128, out_channels=128),
            ConvBlock2D(in_channels=128, out_channels=128),
            ConvBlock2D(in_channels=128, out_channels=128),
            ConvBlock2D(in_channels=128, out_channels=128)
        )

    def forward(self, x):
        return self.conv_section(x)


class StructuralNet(nn.Module):
    def __init__(self):
        super(StructuralNet, self).__init__()
        self.conv_section = nn.Sequential(
            ConvBlock3D(in_channels=1, out_channels=32),
            ConvBlock3D(in_channels=32, out_channels=32),
            ConvBlock3D(in_channels=32, out_channels=64),
            ConvBlock3D(in_channels=64, out_channels=64),
            ConvBlock3D(in_channels=64, out_channels=64),
            ConvBlock3D(in_channels=64, out_channels=64),
            ConvBlock3D(in_channels=64, out_channels=128),
            ConvBlock3D(in_channels=128, out_channels=128),
            ConvBlock3D(in_channels=128, out_channels=128)
        )

    def forward(self, x):
        return self.conv_section(x)


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(ConvBlock2D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv_block(x)


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(ConvBlock3D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv_block(x)

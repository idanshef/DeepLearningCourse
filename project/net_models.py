import torch
import torch.nn as nn


class CompoundNet(nn.Module):
    def __init__(self):
        super(CompoundNet, self).__init__()
        self.appearance_net = AppearanceNet()
        self.structural_net = StructuralNet()
        self.global_avg_pooling_2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg_pooling_3d = nn.AdaptiveAvgPool3d(1)
        self.fusion_net = FusionNet()

    def forward(self, I, G):
        I = self.global_avg_pooling_2d(self.appearance_net(I)).squeeze()
        G = self.global_avg_pooling_3d(self.structural_net(G)).squeeze()
        return self.fusion_net(I, G)


class AppearanceNet(nn.Module):
    def __init__(self):
        super(AppearanceNet, self).__init__()
        self.conv_section = nn.Sequential(
            ConvBlock2D(in_channels=3, out_channels=64),
            ConvBlock2D(in_channels=64, out_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock2D(in_channels=64, out_channels=64),
            ConvBlock2D(in_channels=64, out_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock2D(in_channels=64, out_channels=64),
            ConvBlock2D(in_channels=64, out_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock2D(in_channels=64, out_channels=128),
            ConvBlock2D(in_channels=128, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock2D(in_channels=128, out_channels=128),
            ConvBlock2D(in_channels=128, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock2D(in_channels=128, out_channels=128),
            ConvBlock2D(in_channels=128, out_channels=128)
            
            # ConvBlock2D(in_channels=3, out_channels=64),
            # ConvBlock2D(in_channels=64, out_channels=64),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # ConvBlock2D(in_channels=64, out_channels=64),
            # ConvBlock2D(in_channels=64, out_channels=64),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # ConvBlock2D(in_channels=64, out_channels=64),
            # ConvBlock2D(in_channels=64, out_channels=128),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # ConvBlock2D(in_channels=64, out_channels=128),
            # ConvBlock2D(in_channels=128, out_channels=128),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # ConvBlock2D(in_channels=128, out_channels=128),
            # ConvBlock2D(in_channels=128, out_channels=128),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # ConvBlock2D(in_channels=128, out_channels=256),
            # ConvBlock2D(in_channels=128, out_channels=256)
        )

    def forward(self, x):
        return self.conv_section(x)


class StructuralNet(nn.Module):
    def __init__(self):
        super(StructuralNet, self).__init__()
        self.conv_section = nn.Sequential(
            ConvBlock3D(in_channels=1, out_channels=32),
            ConvBlock3D(in_channels=32, out_channels=32),
            nn.AvgPool3d(kernel_size=2, stride=2),
            ConvBlock3D(in_channels=32, out_channels=64),
            ConvBlock3D(in_channels=64, out_channels=64),
            nn.AvgPool3d(kernel_size=2, stride=2),
            ConvBlock3D(in_channels=64, out_channels=64),
            ConvBlock3D(in_channels=64, out_channels=64),
            nn.AvgPool3d(kernel_size=2, stride=2),
            ConvBlock3D(in_channels=64, out_channels=128),
            ConvBlock3D(in_channels=128, out_channels=128),
            nn.AvgPool3d(kernel_size=2, stride=2),
            ConvBlock3D(in_channels=128, out_channels=128)
            
            # ConvBlock3D(in_channels=1, out_channels=32),
            # ConvBlock3D(in_channels=32, out_channels=32),
            # nn.AvgPool3d(kernel_size=2, stride=2),
            # ConvBlock3D(in_channels=32, out_channels=64),
            # ConvBlock3D(in_channels=64, out_channels=64),
            # nn.AvgPool3d(kernel_size=2, stride=2),
            # ConvBlock3D(in_channels=64, out_channels=64),
            # ConvBlock3D(in_channels=64, out_channels=128),
            # nn.AvgPool3d(kernel_size=2, stride=2),
            # ConvBlock3D(in_channels=64, out_channels=128),
            # ConvBlock3D(in_channels=128, out_channels=256),
            # nn.AvgPool3d(kernel_size=2, stride=2),
            # ConvBlock3D(in_channels=128, out_channels=256)
        )

    def forward(self, x):
        return self.conv_section(x)


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock2D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_block(x)


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock3D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_block(x)

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
    
    def forward(self, I, G):
        return torch.cat((I, G), 1)

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
        self.fc1 = nn.Linear(in_features=128, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=128)
    
    def forward(self, I, G):
        I, G = self.fc1(I), self.fc1(G)
        I_G = self.fc2(I + G)
        I_G = self.fc3(I_G)
        return self.fc4(I_G)
        # return torch.cat((I, G), 1)

class MarginBasedLoss(nn.Module):
    def __init__(self, alpha, m, norm_type=1, reduction='mean'):
        super(MarginBasedLoss, self).__init__()
        self.alpha = alpha
        self.m = m
        self.norm_type = norm_type
        assert reduction is None or reduction in ['mean', 'sum'], "Invalid reduction value!"
        self.reduction = reduction
    
    def forward(self, f1, f2, labels):
        assert len(f1.shape) == len(f2.shape) == 2, "f1 and f2 must be 2d matrices"
        assert f1.shape[0] == f2.shape[0], "f1 and f2 batch dimension must be equal"
        assert f1.shape[0] == labels.shape[0], "input features batch dimension and number of labels must be equal"
        assert len(labels.shape) == 1, "labels must be a one-dim vector"
        
        loss = torch.max(self.alpha + labels.float() * (torch.linalg.norm(f1 - f2, dim=-1, ord=self.norm_type) - self.m),
                         torch.tensor([0.], device=f1.device))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
            
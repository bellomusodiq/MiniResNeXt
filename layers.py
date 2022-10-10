import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ResNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, C=16, downsample=False):
        super(ResNeXtBlock, self).__init__()
        self.downsample = downsample
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.transformations = []
        self.relu = nn.ReLU()
        for i in range(C):
            self.transformations.append(self._make_transformation())
        self.res_downsample_layer = conv2d(
            in_channels, out_channels, kernel_size=1, stride=2
        )

    def _make_transformation(self):
        stride = 1
        if self.downsample:
            stride = 2
        conv1 = nn.Conv2d(self.in_channels, 4, kernel_size=1, stride=1)
        bn1 = nn.BatchNorm2d(4)
        conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=stride, padding=1)
        bn2 = nn.BatchNorm2d(4)
        conv3 = nn.Conv2d(4, self.out_channels, kernel_size=1, stride=1)
        bn3 = nn.BatchNorm2d(self.out_channels)
        return nn.Sequential(
            conv1, bn1, self.relu, conv2, bn2, self.relu, conv3, bn3, self.relu
        ).to(device)

    def forward(self, x):
        residual = x
        if self.downsample:
            residual = self.res_downsample_layer(x)
        transform_result = torch.zeros(
            residual.size(0), self.out_channels, residual.size(2), residual.size(3)
        ).to(device)
        for transformation in self.transformations:
            transform_result.add_(transformation(x))

        result = self.relu(residual + transform_result)

        return result

import torch
import torch.nn as nn

from .layers import ResNeXtBlock


class ResNeXt(nn.Module):
    def __init__(self):
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(2)
        self.resnext1 = ResNeXtBlock(64, 64, C=8)
        self.resnext2 = ResNeXtBlock(64, 128, C=8, downsample=True)
        self.resnext3 = ResNeXtBlock(128, 128, C=8)
        self.resnext4 = ResNeXtBlock(128, 256, C=8, downsample=True)
        self.resnext5 = ResNeXtBlock(256, 256, C=8)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.resnext1(x)
        x = self.resnext2(x)
        x = self.resnext3(x)
        x = self.resnext4(x)
        x = self.resnext5(x)
        x = self.avgpool(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

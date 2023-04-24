import math

import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.bn(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ResNetBlock(3, 32)
        self.block2 = ResNetBlock(32, 32)
        self.block3 = ResNetBlock(32, 32)
        self.block4 = ResNetBlock(32, 32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 8)
        self.init_layers()

    def init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.block1(x)
        x1_out = x1
        x2 = self.block2(x1_out)
        x2_out = x1_out + x2
        x3 = self.block3(x2_out)
        x3_out = x1_out + x2_out + x3
        x4 = self.block4(x3_out)
        x4_out = x1_out + x2_out + x3_out + x4
        x = self.pool(x4_out)
        # out = torch.flatten(out, start_dim=1)
        # change with average pooling
        x = nn.AdaptiveAvgPool2d((1, 1))(x).squeeze()
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    model = ResNet()
    print(model)

    x = torch.randn(1, 3, 32, 32)
    print(model(x).shape)

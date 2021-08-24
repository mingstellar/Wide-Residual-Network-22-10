import torch
from torch import Tensor
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride : int = 1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        if in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                    stride=stride, padding=0, bias=False)
        else:
            self.conv3 = None
    
    def forward(self, x):
        
        shortcut = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.conv3:
            shortcut = self.conv3(shortcut)
        
        out = out + shortcut
        return out

class WideResNet(nn.Module):
    def __init__(self, blocks, widen_factor, block, num_classes=10):
        super(WideResNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                                stride=1, padding=1, bias=True)
        
        self.layer1 = self._make_layer(block, 16, 16*widen_factor, 1, blocks)
        self.layer2 = self._make_layer(block, 16*widen_factor, 32*widen_factor, 2, blocks)
        self.layer3 = self._make_layer(block, 32*widen_factor, 64*widen_factor, 2, blocks)
        self.bn = nn.BatchNorm2d(64*widen_factor)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64*widen_factor, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    

    def _make_layer(self, block, in_channels, out_channels, stride, blocks):
        layers = []

        layers.append(block(in_channels, out_channels, stride))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
    
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def wide_resnet(depth, widen_factor, block, num_classes):
    blocks = (depth - 4) // 6
    return WideResNet(blocks, widen_factor, block, num_classes)

def wrn_22_10():
    return wide_resnet(22, 10, BasicBlock, 10)
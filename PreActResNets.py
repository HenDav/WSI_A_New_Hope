"""preactresnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/abs/1603.05027
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PreActBasic(nn.Module):

    expansion = 1
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * PreActBasic.expansion, kernel_size=3, padding=1)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * PreActBasic.expansion:
            self.shortcut = nn.Conv2d(in_channels, out_channels * PreActBasic.expansion, 1, stride=stride)

    def forward(self, x):

        res = self.residual(x)
        shortcut = self.shortcut(x)

        return res + shortcut


class PreActBottleNeck(nn.Module):

    expansion = 4
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, stride=stride),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * PreActBottleNeck.expansion, 1)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * PreActBottleNeck.expansion:
            self.shortcut = nn.Conv2d(in_channels, out_channels * PreActBottleNeck.expansion, 1, stride=stride)

    def forward(self, x):

        res = self.residual(x)
        shortcut = self.shortcut(x)

        return res + shortcut

class PreActResNet(nn.Module):

    def __init__(self, block, num_block, class_num=2):
        super().__init__()
        self.input_channels = 64

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_layers(block, num_block[0], 64,  1)
        self.stage2 = self._make_layers(block, num_block[1], 128, 2)
        self.stage3 = self._make_layers(block, num_block[2], 256, 2)
        self.stage4 = self._make_layers(block, num_block[3], 512, 2)

        self.linear = nn.Linear(self.input_channels, class_num)

    def _make_layers(self, block, block_num, out_channels, stride):
        layers = []

        layers.append(block(self.input_channels, out_channels, stride))
        self.input_channels = out_channels * block.expansion

        while block_num - 1:
            layers.append(block(self.input_channels, out_channels, 1))
            self.input_channels = out_channels * block.expansion
            block_num -= 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


class PreActResNet_Omer(nn.Module):

    def __init__(self, block, num_block, class_num=2):
        super().__init__()
        self.input_channels = 64

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage1 = self._make_layers(block, num_block[0], 64,  1)
        self.stage2 = self._make_layers(block, num_block[1], 128, 2)
        self.stage3 = self._make_layers(block, num_block[2], 256, 2)
        self.stage4 = self._make_layers(block, num_block[3], 512, 2)

        self.linear = nn.Linear(self.input_channels, class_num)

    def _make_layers(self, block, block_num, out_channels, stride):
        layers = []

        layers.append(block(self.input_channels, out_channels, stride))
        self.input_channels = out_channels * block.expansion

        while block_num - 1:
            layers.append(block(self.input_channels, out_channels, 1))
            self.input_channels = out_channels * block.expansion
            block_num -= 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

def preactresnet18():
    model = PreActResNet(PreActBasic, [2, 2, 2, 2])
    model.model_name = 'preactresnet_18()'
    print('Using model {}'.format(model.model_name))
    return model

def preactresnet34():
    model = PreActResNet(PreActBasic, [3, 4, 6, 3])
    model.model_name = 'preactresnet_34()'
    print('Using model {}'.format(model.model_name))
    return model

def preactresnet50():
    model = PreActResNet(PreActBottleNeck, [3, 4, 6, 3])
    model.model_name = 'preactresnet_50()'
    print('Using model {}'.format(model.model_name))
    return model


def preactresnet50_Omer():
    model = PreActResNet_Omer(PreActBottleNeck, [3, 4, 6, 3])
    model.model_name = 'preactresnet_50_Omer()'
    print('Using model {}'.format(model.model_name))
    return model

def preactresnet101():
    return PreActResNet(PreActBottleNeck, [3, 4, 23, 3])
    model.model_name = 'preactresnet_101()'
    print('Using model {}'.format(model.model_name))
    return model

def preactresnet152():
    return PreActResNet(PreActBottleNeck, [3, 8, 36, 3])
    model.model_name = 'preactresnet_152()'
    print('Using model {}'.format(model.model_name))
    return model


####################################################################################################
# Ron's nets

class PreActBottleneck_Ron(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck_Ron, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet_Ron(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(PreActResNet_Ron, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(128*block.expansion, num_classes)
        self.model_name = ''

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.shape[3])
        out = out.view(out.size(0), -1)
        feat = out
        # out = self.dropout(self.linear(out))
        out = self.linear(out)
        return out


def PreActResNet50_Ron():
    model = PreActResNet_Ron(PreActBottleneck_Ron, [3, 4, 6, 3])
    model.model_name = 'PreActResNet50_Ron()'
    return model

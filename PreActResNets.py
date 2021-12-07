"""preactresnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/abs/1603.05027
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

#THIS_FILE = os.path.realpath(__file__).split('/')[-1].split('.')[0] + '.'
THIS_FILE = os.path.basename(os.path.realpath(__file__)).split('.')[0] + '.'

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
        x = torch.nn.functional.softmax(x, dim=1)

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
        x = torch.nn.functional.softmax(x, dim=1)

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
    model.model_name = THIS_FILE + 'preactresnet50()'
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

        # is_HeatMap is used when we want to create a heatmap and we need to fkip the order of the last two layers
        self.is_HeatMap = False  # Omer 26/7/2021

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def change_num_classes(self, num_classes):
        self.linear = nn.Linear(self.linear.in_features, num_classes)


    def forward(self, x):
        if len(x.shape) == 5:
            num_of_bags, tiles_amount, _, tiles_size, _ = x.shape
            x = torch.reshape(x, (num_of_bags * tiles_amount, 3, tiles_size, tiles_size))


        if self.is_HeatMap:
            if self.training is True:
                raise Exception('Pay Attention that the model in not in eval mode')

            #print('Input size to Conv-Net is of size {}'.format(x.shape))
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)

            #print('OutPut size from Conv-Net is of size {}'.format(out.shape))
            #if x.shape[2] <= 256 or x.shape[2] == 1024 or x.shape[2] == 2048:  # If input is in the original tile size dimensions (using <= and not == for the case tile is 128 pixels)
            if x.shape[2] <= 256 or x.shape[2] == 1024 or x.shape[2] >= 2048:
                image_to_compute_MilWeights = F.avg_pool2d(out, kernel_size=32, stride=1, padding=16, count_include_pad=False)
                initial_image_size = out.shape[2]
                out_for_dict = out
                #vectored_image = out.view(out.size(2) * out.size(3), -1)
                vectored_image = torch.transpose(torch.reshape(out.squeeze(0), (out.size(1), out.size(2) * out.size(3))), 1, 0)
                vectored_heat_image_2_channels = self.linear(vectored_image)
                vectored_heat_image = vectored_heat_image_2_channels[:, 1] - vectored_heat_image_2_channels[:, 0]
                small_heat_map = vectored_heat_image.view(1, 1, initial_image_size, initial_image_size)

                # Upsampling the heat map:
                large_heat_map = F.interpolate(small_heat_map, size=x.shape[2], mode='bilinear')

                # Computing the scores:
                out = F.avg_pool2d(out, out.shape[3])
                features = out.view(out.size(0), -1)
                out = self.linear(features)

                data_dict_4_gil = {'linear_weights': self.linear.weight.cpu().numpy(),
                                   'linear_bias': self.linear.bias.cpu().numpy(),
                                   'heat_map': small_heat_map.squeeze().cpu().numpy(),
                                   'feature_map': out_for_dict.squeeze(0).cpu().numpy(),
                                   }
                out_data_dict = {'Large Heat Map': large_heat_map,
                                 'Small Heat Map': small_heat_map,
                                 'Scores': out,
                                 'Data 4 Gil': data_dict_4_gil,
                                 'Features': features,
                                 'Large Image for MIL Weights': image_to_compute_MilWeights,
                                 'Large Image for MIL Weights Without Averaging Sliding Window': out_for_dict}
                return out_data_dict

            else:
                raise Exception('Need to correct the code')
                '''initial_image_size = out.shape[2]
                out = F.avg_pool2d(out, kernel_size=32, stride=1)  # using a kernel of size 32 since this is the size of last image before changing to features
                features = out.view(out.size(2) * out.size(3), -1)
                vectored_tile_heat_map_2_channels = self.linear(features)
                vectored_tile_heat_map = vectored_tile_heat_map_2_channels[:, 1] - vectored_tile_heat_map_2_channels[:, 0]
                image_tile_heatmap = vectored_tile_heat_map.view(1, 1, out.shape[2], out.shape[2])
                large_heat_map = F.upsample(image_tile_heatmap, size=initial_image_size)

                return large_heat_map'''

        else:
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            # The following lines (commented) are needed when trying to plot the model graph using summarywriter
            '''print(type(out.shape[3]), out.shape[3])
            print(type(out.shape))
            if type(out.shape[3]) == torch.Tensor:
                out = F.avg_pool2d(out, int(out.shape[3]))
            else:'''
            out = F.avg_pool2d(out, out.shape[3])
            out = out.view(out.size(0), -1)
            features = out
            #out = self.linear(self.dropout(out))
            out = self.linear(out)

            #return out
            return out, features #RanS 1.7.21


#def PreActResNet50_Ron():
def PreActResNet50_Ron(train_classifier_only=False):
    model = PreActResNet_Ron(PreActBottleneck_Ron, [3, 4, 6, 3])
    model.model_name = THIS_FILE + 'PreActResNet50_Ron()'
    print(model.model_name)

    if train_classifier_only:
        model.model_name = THIS_FILE + 'PreActResNet50_Ron(train_classifier_only=True)'
        for param in model.parameters():
            param.requires_grad = False
        for param in model.linear.parameters():
            param.requires_grad = True

    return model

def MIL_PreActResNet50_Ron():
    model = PreActResNet_Ron(PreActBottleneck_Ron, [3, 4, 6, 3], num_classes=500)
    model.model_name = THIS_FILE + 'PreActResNet50_Ron()'
    print(model.model_name)
    return model



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import resnet, resnet50, resnet34
from torchvision.models.resnet import ResNet
import os
from torchvision.models.utils import load_state_dict_from_url

THIS_FILE = os.path.basename(os.path.realpath(__file__)).split('.')[0] + '.'


class MyGroupNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyGroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=num_channels,
                                 eps=1e-5, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return x


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = THIS_FILE + 'ResNet18()'
        print('Using model {}'.format(self.model_name))
        self.basic_resnet = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=2)

    def forward(self, x):
        x = x.squeeze()
        x = self.basic_resnet(x)
        #x = torch.nn.functional.softmax(x, dim=1) #cancelled RanS 11.4.21
        return x


class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = THIS_FILE + 'ResNet34()'
        print('Using model {}'.format(self.model_name))
        self.basic_resnet = resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3], num_classes=2)

    def forward(self, x):
        x = x.squeeze()
        x = self.basic_resnet(x)
        #x = torch.nn.functional.softmax(x, dim=1) #cancelled RanS 11.4.21
        return x


#RanS 17.11.21, resnet34 pretrained on Imagenet, with binary classifier
class MyResNet34(nn.Module):
    def __init__(self, train_classifier_only=False):
        super().__init__()
        self.model = resnet34(pretrained=True)
        #model.fc.in_features = 2
        N_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.model.my_fc = nn.Linear(N_features, 2)
        self.model_name = THIS_FILE + 'MyResNet34()'
        print('Using model {}'.format(self.model_name))

        if train_classifier_only:
            self.model_name = THIS_FILE + 'MyResNet34(train_classifier_only=True)'
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.my_fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = x.squeeze()
        features = self.model(x)
        out = self.model.my_fc(features)
        # x = torch.nn.functional.softmax(x, dim=1) #cancelled RanS 11.4.21
        return out, features

'''def MyResNet34(train_classifier_only=False):
    model = resnet34(pretrained=True)
    #model.fc.in_features = 2
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.model_name = THIS_FILE + 'MyResNet34()'
    print(model.model_name)

    if train_classifier_only:
        model.model_name = THIS_FILE + 'MyResNet34(train_classifier_only=True)'
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    return model'''


class ResNet50(nn.Module):
    def __init__(self,
                 pretrained=False,
                 num_classes: int = 2):
        super().__init__()
        self.model_name = THIS_FILE + 'ResNet50()'
        print('Using model {}'.format(self.model_name))

        self.basic_resnet = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3])

        if pretrained:
            state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
            self.basic_resnet.load_state_dict(state_dict)

        self.linear_layer = nn.Linear(1000, num_classes)


    def forward(self, x):
        if len(x.shape) == 5:
            num_of_bags, tiles_amount, _, tiles_size, _ = x.shape
            x = torch.reshape(x, (num_of_bags * tiles_amount, 3, tiles_size, tiles_size))

        x = self.linear_layer(self.basic_resnet(x))
        #x = torch.nn.functional.softmax(x, dim=1) #cancelled RanS 11.4.21
        return x



'''
def ResNet_18():
    print('Using model ResNet_18')
    model = models.resnet18(pretrained=False)
    model.fc.out_features = 2
    model.model_name = 'ResNet_18()'
    return model


def ResNet_34():
    print('Using model ResNet_34')
    model = models.resnet34(pretrained=False)
    model.fc.out_features = 2
    model.model_name = 'ResNet_34()'
    return model


def ResNet_50():
    print('Using model ResNet_50')
    model = models.resnet50(pretrained=False)
    model.fc.out_features = 2
    model.model_name = 'ResNet_50()'
    return model
'''


class ResNet34_GN(nn.Module):
    def __init__(self):
        super(ResNet34_GN, self).__init__()
        self.model_name = THIS_FILE + 'ResNet34_GN()'
        print('Using model {}'.format(self.model_name))
        # Replace all BatchNorm layers with GroupNorm:

        self.con_layers = resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3], num_classes=1000, zero_init_residual=False,
                                        groups=1, width_per_group=64, replace_stride_with_dilation=None,
                                        norm_layer=MyGroupNorm)

        self.linear_layer = nn.Linear(in_features=1000, out_features=2)

    def forward(self, x):
        x = x.squeeze()
        x = self.linear_layer(self.con_layers(x))
        #x = torch.nn.functional.softmax(x, dim=1) #cancelled RanS 11.4.21
        return x


class ResNet50_GN(nn.Module):
    def __init__(self):
        super(ResNet50_GN, self).__init__()
        self.model_name = THIS_FILE + 'ResNet50_GN()'
        print('Using model {}'.format(self.model_name))
        # Replace all BatchNorm layers with GroupNorm:

        self.con_layers = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1000, zero_init_residual=False,
                                        groups=1, width_per_group=64, replace_stride_with_dilation=None,
                                        norm_layer=MyGroupNorm)

        self.linear_layer = nn.Linear(in_features=1000, out_features=2)

    def forward(self, x):
        x = x.squeeze()
        x = self.linear_layer(self.con_layers(x))
        #x = torch.nn.functional.softmax(x, dim=1) #cancelled RanS 11.4.21
        return x


# RanS 14.12.20
class net_with_3FC(nn.Module):
    def __init__(self, pretrained_model, reinit_last_layer=True):
        super(net_with_3FC, self).__init__()
        self.model_name = THIS_FILE + 'net_with_3FC()'
        self.pretrained = pretrained_model
        num_ftrs = self.pretrained.fc.in_features
        # RanS 18.11.20, change momentum to 0.5
        # for layer in self.pretrained.modules():
        # if layer._get_name() == 'BatchNorm2d':
        # layer.momentum = 0.5
        # print(layer)
        # re-init last bottleneck.layer! RanS 17.11.20
        # nn.init.kaiming_normal_(self.pretrained.layer4[0].downsample[0].weight, mode='fan_in', nonlinearity='relu')
        # nn.init.constant_(self.pretrained.layer4[0].downsample[1].weight, 0)
        # nn.init.constant_(self.pretrained.layer4[0].downsample[1].bias, 0)
        # for ii in range(3):
        # for ii in range(2,3):
        if reinit_last_layer:
            for ii in range(1, 3):
                nn.init.kaiming_normal_(self.pretrained.layer4[ii].conv1.weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(self.pretrained.layer4[ii].conv2.weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(self.pretrained.layer4[ii].conv3.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(self.pretrained.layer4[ii].bn1.weight, 0)
                nn.init.constant_(self.pretrained.layer4[ii].bn2.weight, 0)
                nn.init.constant_(self.pretrained.layer4[ii].bn3.weight, 0)
                nn.init.constant_(self.pretrained.layer4[ii].bn1.bias, 0)
                nn.init.constant_(self.pretrained.layer4[ii].bn2.bias, 0)
                nn.init.constant_(self.pretrained.layer4[ii].bn3.bias, 0)
                # RanS 18.11.20, reset all running bn stats
                # self.pretrained.layer4[ii].bn1.reset_running_stats()
                # self.pretrained.layer4[ii].bn2.reset_running_stats()
                # self.pretrained.layer4[ii].bn3.reset_running_stats()
            # nn.init.kaiming_normal_(self.pretrained.layer4[-1].conv1.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.kaiming_normal_(self.pretrained.layer4[-1].conv2.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.kaiming_normal_(self.pretrained.layer4[-1].conv3.weight, mode='fan_in', nonlinearity='relu')
        self.pretrained.fc = nn.Identity()
        self.dropout = nn.Dropout(p=0.5)
        # self.fc = nn.Linear(num_ftrs, 2)
        self.fc1 = nn.Linear(num_ftrs, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)
        # nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')  # RanS 17.11.20, try He init
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')  # RanS 17.11.20, try He init
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')  # RanS 17.11.20, try He init
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')  # RanS 17.11.20, try He init

    def forward(self, x):
        x = self.pretrained(x)
        # x = self.fc(x)
        # x = self.fc(self.dropout(x))
        x = F.relu(self.fc1(self.dropout(x)))
        x = F.relu(self.fc2(self.dropout(x)))
        x = self.fc3(x)
        #x = torch.nn.functional.softmax(x, dim=1) #cancelled RanS 11.4.21
        return x


# RanS 17.12.20
class resnet50_with_3FC(nn.Module):
    def __init__(self, pretrained=True):
        super(resnet50_with_3FC, self).__init__()
        self.model_name = THIS_FILE + 'resnet50_with_3FC()'
        pretrained_model = models.resnet50(pretrained=pretrained)
        self.model = net_with_3FC(pretrained_model=pretrained_model, reinit_last_layer=False)

    def forward(self, x):
        x = self.model(x)
        return x


# RanS 21.12.20
class net_with_2FC(nn.Module):
    def __init__(self, pretrained_model):
        super(net_with_2FC, self).__init__()
        self.model_name = THIS_FILE + 'net_with_2FC()'
        self.pretrained = pretrained_model
        num_ftrs = self.pretrained.fc.in_features
        self.pretrained.fc = nn.Identity()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(num_ftrs, 512)
        self.fc2 = nn.Linear(512, 512)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')  # RanS 17.11.20, try He init
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')  # RanS 17.11.20, try He init

    def forward(self, x):
        # old version:
        '''x = self.pretrained(x)
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(x)'''

        # RanS 14.1.21, Nikhil's version
        x = self.pretrained(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        #x = torch.nn.functional.softmax(x, dim=1) #cancelled RanS 11.4.21
        return x


# RanS 21.12.20
class ReceptorNet_feature_extractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ReceptorNet_feature_extractor, self).__init__()
        self.model_name = THIS_FILE + 'ReceptorNet_feature_extractor()'
        pretrained_model = models.resnet50(pretrained=pretrained)
        self.model = net_with_2FC(pretrained_model=pretrained_model)

    def forward(self, x):
        x = self.model(x)
        return x


##########################################################

class ResNet_NO_downsample(nn.Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_NO_downsample, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        '''self.inplanes = 64'''
        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        '''self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)'''
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        '''self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))'''

        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) == 5:
            num_of_bags, tiles_amount, _, tiles_size, _ = x.shape
            x = torch.reshape(x, (num_of_bags * tiles_amount, 3, tiles_size, tiles_size))
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        #x = torch.nn.functional.softmax(x, dim=1) #cancelled RanS 11.4.21
        return x


# def _resnet(arch, block, layers, pretrained, progress, **kwargs):
def _resnet_NO_DOWNSAMPLE(arch, block, layers, progress, **kwargs):
    model = ResNet_NO_downsample(block, layers, **kwargs)
    return model


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def resnet50_pretrained():
    model = resnet50(pretrained=True, progress=True, )
    model.model_name = THIS_FILE + 'resnet50_pretrained()'
    model.fc.out_features = 2
    print('Using model: ', model.model_name)
    return model

def ResNet_50_NO_downsample():
    """
    This model is based on ResNet50 but without the downsample layer at the beginning of the net.
    The other layers are modified so that the model will fit (memory-wise) in the GPU.
    """
    model = ResNet_NO_downsample(Bottleneck, [3, 4, 6, 3], num_classes=2)
    model.model_name = THIS_FILE + 'ResNet_50_NO_downsample()'
    print('Using model {}'.format(model.model_name))

    return model


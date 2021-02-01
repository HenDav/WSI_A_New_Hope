import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import resnet
import os

#THIS_FILE = os.path.realpath(__file__).split('/')[-1].split('.')[0] + '.'
THIS_FILE = os.path.basename(os.path.realpath(__file__)).split('.')[0] + '.'


'''
class Flatten(nn.Module):
    """
    This class flattens an array to a vector
    """
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
class ResNet50_GatedAttention(nn.Module):
    def __init__(self):
        super(ResNet50_GatedAttention, self).__init__()
        self.M = 500
        self.L = 128
        self.K = 1    # in the paper referred a 1.

        self.infer = False
        self.part_1 = False
        self.part_2 = False

        self._feature_extractor_ResNet50_part_1 = models.resnet50()

        self._feature_extractor_fc = nn.Sequential(
            #nn.Dropout(p=0.5),
            nn.Linear(in_features=1000, out_features=self.M)
        )
        
        self.feature_extractor = nn.Sequential(
            self._feature_extractor_ResNet50_part_1,
            self._feature_extractor_fc
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.L, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x, H = None, A = None):
        if not self.infer:
            x = x.squeeze(0)
            # In case we are training the model we'll use bags that contains only part of the tiles.
            H = self.feature_extractor(x)

            """
            H = H.view(-1, 50 * 4 * 4) 
            H = self.feature_extractor_part2(H)  # NxL
            """

            A_V = self.attention_V(H)  # NxL
            A_U = self.attention_U(H)  # NxL
            A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N

            M = torch.mm(A, H)  # KxM

            # Because this is a binary classifier, the output of it is one single number which can be interpreted as the
            # probability that the input belong to class 1/TRUE (and not 0/FALSE)
            Y_prob = self.classifier(M)

            # The following line just turns probability to class.
            Y_class = torch.ge(Y_prob, 0.5).float()

            return Y_prob, Y_class, A

        # In case we want an inference of a whole slide we need ALL the tiles from that slide:
        else:
            if not self.part_1 ^ self.part_2:
                raise Exception('Inference Mode should include feature extraction (part 1) OR classification (part 2)')
            if self.part_1:
                x = x.squeeze(0)
                H = self.feature_extractor(x)
                A_V = self.attention_V(H)  # NxL
                A_U = self.attention_U(H)  # NxL
                A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
                A = torch.transpose(A, 1, 0)  # KxN
                return H, A

            elif self.part_2:
                A = F.softmax(A, dim=1)  # softmax over N

                M = torch.mm(A, H)  # KxM

                Y_prob = self.classifier(M)

                Y_class = torch.ge(Y_prob, 0.5).float()
                return Y_prob, Y_class, A


    """
    # AUXILIARY METHODS
    def calculate_classification_accuracy(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
    """
'''
"""
class ResNet50_2(nn.Module):
    def __init__(self):
        super(ResNet50_2, self).__init__()

        self.num_classes = 2
        self.part_1 = models.resnet50(pretrained=True)
        self.dropout_1 = nn.Dropout(p=0.5)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.linear_1 = nn.Linear(in_features=1000, out_features=700)
        self.linear_2 = nn.Linear(in_features=700, out_features=self.num_classes)


        self.net = nn.Sequential( self.part_1,
                                  self.linear_1,
                                  self.dropout_1,
                                  self.linear_2,
                                  self.dropout_2
                                  )

    def forward(self, x):
        x = x.squeeze(0)
        out = self.net(x)
        '''
        x = self.part_1(x)
        out = self.dropout_2(self.linear_2(self.dropout_1(self.linear_1(x))))
        '''
        return out

class ResNext_50(nn.Module):
    def __init__(self):
        super(ResNext_50, self).__init__()

        self.num_classes = 2
        self.part_1 = models.resnext50_32x4d(pretrained=False)
        self.dropout_1 = nn.Dropout(p=0.5)

        self.net = nn.Sequential( models.resnext50_32x4d(pretrained=False),
                                  nn.Dropout(p=0.5),
                                  nn.Linear(in_features=1000, out_features=self.num_classes))

    def forward(self, x):
        x = x.squeeze(0)
        out = self.net(x)
        return out
"""


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
        x = torch.nn.functional.softmax(x, dim=1)
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
        x = torch.nn.functional.softmax(x, dim=1)
        return x


class ResNet50(nn.Module):
    def __init__(self,
                 num_classes: int = 2):
        super().__init__()
        self.model_name = THIS_FILE + 'ResNet50()'
        print('Using model {}'.format(self.model_name))
        self.basic_resnet = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

    def forward(self, x):
        if len(x.shape) == 5:
            num_of_bags, tiles_amount, _, tiles_size, _ = x.shape
            x = torch.reshape(x, (num_of_bags * tiles_amount, 3, tiles_size, tiles_size))

        #x = x.squeeze()
        x = self.basic_resnet(x)
        x = torch.nn.functional.softmax(x, dim=1)
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
        x = torch.nn.functional.softmax(x, dim=1)
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
        x = torch.nn.functional.softmax(x, dim=1)
        return x



#RanS 14.12.20
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
        #self.fc = nn.Linear(num_ftrs, 2)
        self.fc1 = nn.Linear(num_ftrs, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)
        #nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')  # RanS 17.11.20, try He init
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')  # RanS 17.11.20, try He init
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')  # RanS 17.11.20, try He init
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')  # RanS 17.11.20, try He init
    def forward(self, x):
        x = self.pretrained(x)
        # x = self.fc(x)
        #x = self.fc(self.dropout(x))
        x = F.relu(self.fc1(self.dropout(x)))
        x = F.relu(self.fc2(self.dropout(x)))
        x = self.fc3(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


#RanS 17.12.20
class resnet50_with_3FC(nn.Module):
    def __init__(self, pretrained=True):
        super(resnet50_with_3FC, self).__init__()
        self.model_name = THIS_FILE + 'resnet50_with_3FC()'
        pretrained_model = models.resnet50(pretrained=pretrained)
        self.model = net_with_3FC(pretrained_model=pretrained_model, reinit_last_layer=False)

    def forward(self, x):
        x = self.model(x)
        return x


#RanS 21.12.20
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
        #old version:
        '''x = self.pretrained(x)
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(x)'''

        #RanS 14.1.21, Nikhil's version
        x = self.pretrained(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = torch.nn.functional.softmax(x, dim=1)
        return x


#RanS 21.12.20
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
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


# def _resnet(arch, block, layers, pretrained, progress, **kwargs):
def _resnet(arch, block, layers, progress, **kwargs):
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


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def ResNet_50_NO_downsample():
    """
    This model is based on ResNet50 but without the downsample layer at the beginning of the net.
    The other layers are modified so that the model will fit (memory-wise) in the GPU.
    """
    model = ResNet_NO_downsample(Bottleneck, [3, 4, 6, 3], num_classes=2)
    model.model_name = THIS_FILE + 'ResNet_50_NO_downsample()'
    print('Using model {}'.format(model.model_name))

    return model


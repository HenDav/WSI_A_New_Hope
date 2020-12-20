import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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


class MyGroupNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyGroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=num_channels,
                                 eps=1e-5, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return x


def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3])
    #return PreActResNet(PreActBottleneck, [3, 4, 4, 2])

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(PreActResNet, self).__init__()
        self.model_name = 'preact_resnet'
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(128*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.squeeze(0)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.shape[3])
        out = out.view(out.size(0), -1)
        feat = out
        #out = self.dropout(self.linear(out))
        out = self.linear(self.dropout(out))
        return out#, feat

class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
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

def ResNet_50():
    print('Using model ResNet_50')
    model = models.resnet50(pretrained=False)
    model.fc.out_features = 2
    return model


class ResNet34_GN(nn.Module):
    def __init__(self):
        super(ResNet34_GN, self).__init__()
        self.model_name = 'ResNet34_GN'
        print('Using model {}'.format(self.model_name))
        # Replace all BatchNorm layers with GroupNorm:

        self.con_layers = resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3], num_classes=1000, zero_init_residual=False,
                         groups=1, width_per_group=64, replace_stride_with_dilation=None,
                         norm_layer=MyGroupNorm)

        self.linear_layer = nn.Linear(in_features=1000, out_features=2)

    def forward(self, x):
        x = x.squeeze()
        x = self.linear_layer(self.con_layers(x))

        return x


class ResNet50_GN(nn.Module):
    def __init__(self):
        super(ResNet50_GN, self).__init__()
        self.model_name = 'ResNet50_GN'
        print('Using model {}'.format(self.model_name))
        # Replace all BatchNorm layers with GroupNorm:

        self.con_layers = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], num_classes=1000, zero_init_residual=False,
                                        groups=1, width_per_group=64, replace_stride_with_dilation=None,
                                        norm_layer=MyGroupNorm)

        self.linear_layer = nn.Linear(in_features=1000, out_features=2)

    def forward(self, x):
        x = x.squeeze()
        x = self.linear_layer(self.con_layers(x))

        return x



#RanS 14.12.20
class net_with_3FC(nn.Module):
    def __init__(self, pretrained_model, reinit_last_layer=True):
        super(net_with_3FC, self).__init__()
        self.model_name = 'net_with_3FC'
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
        return x


#RanS 17.12.20
class resnet50_with_3FC(nn.Module):
    def __init__(self, pretrained=True):
        super(resnet50_with_3FC, self).__init__()
        self.model_name = 'resnet50_with_3FC'
        pretrained_model = models.resnet50(pretrained=pretrained)
        self.model = net_with_3FC(pretrained_model=pretrained_model, reinit_last_layer=False)

    def forward(self, x):
        x = self.model(x)
        return x
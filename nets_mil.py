import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.models as models
from torchvision.models import resnet
from nets import ResNet50_GN, ResNet34_GN, ReceptorNet_feature_extractor


class Flatten(nn.Module):
    """
    This class flattens an array to a vector
    """

    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class ResNet18_GatedAttention(nn.Module):
    def __init__(self):
        super(ResNet18_GatedAttention, self).__init__()
        print('Using model ResNet18_GatedAttention')
        self.M = 500
        self.L = 128
        self.K = 1  # in the paper referred a 1.

        self.infer = False
        self.part_1 = False
        self.part_2 = False

        self.feat_ext_part_1 = models.resnet18(pretrained=False)
        self.dropout_1 = nn.Dropout(p=0.5)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.linear_1 = nn.Linear(in_features=1000, out_features=700)
        self.linear_2 = nn.Linear(in_features=700, out_features=self.M)

        self._feature_extractor_fc = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=1000, out_features=self.M)
        )

        self.feature_extractor = nn.Sequential(
            self.feat_ext_part_1,
            self._feature_extractor_fc
        )

        self.att_V_1 = nn.Linear(self.M, self.L)
        self.att_V_2 = nn.Tanh()
        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh()
        )
        self.att_U_1 = nn.Linear(self.M, self.L)
        self.att_U_2 = nn.Sigmoid()
        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Sigmoid()
        )

        self.weig = nn.Linear(self.L, self.K)
        self.attention_weights = nn.Linear(self.L, self.K)


        self.class_1 = nn.Linear(self.M * self.K, 1)
        self.class_2 = nn.Sigmoid()
        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x, H=None, A=None):
        if not self.infer:
            x = x.squeeze(0)
            H = self.linear_2(self.linear_1(self.feat_ext_part_1(x)))
            # H = self.linear_2(self.dropout_2(self.linear_1(self.dropout_1(self.feat_ext_part_1(x)))))
            # In case we are training the model we'll use bags that contains only part of the tiles.
            ### H = self.feature_extractor(x)

            """
            H = H.view(-1, 50 * 4 * 4) 
            H = self.feature_extractor_part2(H)  # NxL
            """
            A_V = self.att_V_2(self.att_V_1(H))
            ### A_V = self.attention_V(H)  # NxL
            A_U = self.att_U_2(self.att_U_1(H))
            ### A_U = self.attention_U(H)  # NxL
            A = self.weig(A_V * A_U)
            ### A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N

            M = torch.mm(A, H)  # KxM

            # Because this is a binary classifier, the output of it is one single number which can be interpreted as the
            # probability that the input belong to class 1/TRUE (and not 0/FALSE)
            Y_prob = self.class_2(self.class_1(M))
            ### Y_prob = self.classifier(M)

            # The following line just turns probability to class.
            Y_class = torch.ge(Y_prob, 0.5).float()

            return Y_prob, Y_class, A

        # In case we want an inference of a whole slide we need ALL the tiles from that slide:
        else:
            if not self.part_1 ^ self.part_2:
                raise Exception('Inference Mode should include feature extraction (part 1) OR classification (part 2)')
            if self.part_1:
                '''
                H = self.feature_extractor(x)
                A_V = self.attention_V(H)  # NxL
                A_U = self.attention_U(H)  # NxL
                A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
                A = torch.transpose(A, 1, 0)  # KxN
                '''

                x = x.squeeze(0)
                # H = self.linear_2(self.dropout_2(self.linear_1(self.dropout_1(self.feat_ext_part_1(x)))))
                H = self.linear_2(self.dropout_2(self.linear_1(self.dropout_1(self.feat_ext_part_1(x)))))
                # TODO : try to remove dropout layers in inference mode
                A_V = self.att_V_2(self.att_V_1(H))
                A_U = self.att_U_2(self.att_U_1(H))
                A = self.weig(A_V * A_U)
                A = torch.transpose(A, 1, 0)  # KxN
                return H, A

            elif self.part_2:
                A = F.softmax(A, dim=1)  # softmax over N
                M = torch.mm(A, H)  # KxM
                # Y_prob = self.classifier(M)
                Y_prob = self.class_2(self.class_1(M))

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


class ResNet34_GatedAttention(nn.Module):
    def __init__(self):
        super(ResNet34_GatedAttention, self).__init__()
        print('Using model ResNet34_GatedAttention')
        self.M = 500
        self.L = 128
        self.K = 1  # in the paper referred a 1.

        self.infer = False
        self.part_1 = False
        self.part_2 = False

        self.feat_ext_part_1 = models.resnet34(pretrained=False)
        self.dropout_1 = nn.Dropout(p=0.5)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.linear_1 = nn.Linear(in_features=1000, out_features=700)
        self.linear_2 = nn.Linear(in_features=700, out_features=self.M)
        self.att_V_1 = nn.Linear(self.M, self.L)
        self.att_V_2 = nn.Tanh()
        self.att_U_1 = nn.Linear(self.M, self.L)
        self.att_U_2 = nn.Sigmoid()
        self.class_1 = nn.Linear(self.M * self.K, 1)
        self.class_2 = nn.Sigmoid()
        self.weig = nn.Linear(self.L, self.K)

        '''
        self._feature_extractor_fc = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=1000, out_features=self.M)
        )

        self.feature_extractor = nn.Sequential(
            self.feat_ext_part_1,
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
        '''

    def forward(self, x, H=None, A=None):
        if not self.infer:
            x = x.squeeze(0)
            # H = self.linear_2(self.dropout_2(self.linear_1(self.dropout_1(self.feat_ext_part_1(x)))))
            H = self.linear_2(self.linear_1(self.feat_ext_part_1(x)))
            # In case we are training the model we'll use bags that contains only part of the tiles.
            ### H = self.feature_extractor(x)

            """
            H = H.view(-1, 50 * 4 * 4) 
            H = self.feature_extractor_part2(H)  # NxL
            """
            A_V = self.att_V_2(self.att_V_1(H))
            ### A_V = self.attention_V(H)  # NxL
            A_U = self.att_U_2(self.att_U_1(H))
            ### A_U = self.attention_U(H)  # NxL
            A = self.weig(A_V * A_U)
            ### A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N

            M = torch.mm(A, H)  # KxM

            # Because this is a binary classifier, the output of it is one single number which can be interpreted as the
            # probability that the input belong to class 1/TRUE (and not 0/FALSE)
            Y_prob = self.class_2(self.class_1(M))
            ### Y_prob = self.classifier(M)

            # The following line just turns probability to class.
            Y_class = torch.ge(Y_prob, 0.5).float()

            return Y_prob, Y_class, A

        # In case we want an inference of a whole slide we need ALL the tiles from that slide:
        else:
            if not self.part_1 ^ self.part_2:
                raise Exception('Inference Mode should include feature extraction (part 1) OR classification (part 2)')
            if self.part_1:
                '''
                H = self.feature_extractor(x)
                A_V = self.attention_V(H)  # NxL
                A_U = self.attention_U(H)  # NxL
                A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
                A = torch.transpose(A, 1, 0)  # KxN
                '''

                x = x.squeeze(0)
                # H = self.linear_2(self.dropout_2(self.linear_1(self.dropout_1(self.feat_ext_part_1(x)))))
                H = self.linear_2(self.dropout_2(self.linear_1(self.dropout_1(self.feat_ext_part_1(x)))))
                # TODO : try to remove dropout layers in inference mode
                A_V = self.att_V_2(self.att_V_1(H))
                A_U = self.att_U_2(self.att_U_1(H))
                A = self.weig(A_V * A_U)
                A = torch.transpose(A, 1, 0)  # KxN
                return H, A

            elif self.part_2:
                A = F.softmax(A, dim=1)  # softmax over N
                M = torch.mm(A, H)  # KxM
                # Y_prob = self.classifier(M)
                Y_prob = self.class_2(self.class_1(M))

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


class ResNet50_GatedAttention(nn.Module):
    def __init__(self):
        super(ResNet50_GatedAttention, self).__init__()
        print('Using model ResNet50_GatedAttention')
        self.M = 500
        self.L = 128
        self.K = 1  # in the paper referred a 1.

        self.infer = False
        self.part_1 = False
        self.part_2 = False

        self.feat_ext_part_1 = models.resnet50(pretrained=False)
        self.dropout_1 = nn.Dropout(p=0.5)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.linear_1 = nn.Linear(in_features=1000, out_features=700)
        self.linear_2 = nn.Linear(in_features=700, out_features=self.M)
        self.att_V_1 = nn.Linear(self.M, self.L)
        self.att_V_2 = nn.Tanh()
        self.att_U_1 = nn.Linear(self.M, self.L)
        self.att_U_2 = nn.Sigmoid()
        self.class_1 = nn.Linear(self.M * self.K, 1)
        self.class_2 = nn.Sigmoid()
        self.weig = nn.Linear(self.L, self.K)

        '''
        self._feature_extractor_fc = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=1000, out_features=self.M)
        )

        self.feature_extractor = nn.Sequential(
            self.feat_ext_part_1,
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
        '''

    def forward(self, x, H=None, A=None):
        if not self.infer:
            x = x.squeeze(0)
            # H = self.linear_2(self.dropout_2(self.linear_1(self.dropout_1(self.feat_ext_part_1(x)))))
            H = self.linear_2(self.linear_1(self.feat_ext_part_1(x)))
            # In case we are training the model we'll use bags that contains only part of the tiles.
            ### H = self.feature_extractor(x)

            """
            H = H.view(-1, 50 * 4 * 4) 
            H = self.feature_extractor_part2(H)  # NxL
            """
            A_V = self.att_V_2(self.att_V_1(H))
            ### A_V = self.attention_V(H)  # NxL
            A_U = self.att_U_2(self.att_U_1(H))
            ### A_U = self.attention_U(H)  # NxL
            A = self.weig(A_V * A_U)
            ### A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N

            M = torch.mm(A, H)  # KxM

            # Because this is a binary classifier, the output of it is one single number which can be interpreted as the
            # probability that the input belong to class 1/TRUE (and not 0/FALSE)
            Y_prob = self.class_2(self.class_1(M))
            ### Y_prob = self.classifier(M)

            # The following line just turns probability to class.
            Y_class = torch.ge(Y_prob, 0.5).float()

            return Y_prob, Y_class, A

        # In case we want an inference of a whole slide we need ALL the tiles from that slide:
        else:
            if not self.part_1 ^ self.part_2:
                raise Exception('Inference Mode should include feature extraction (part 1) OR classification (part 2)')
            if self.part_1:
                '''
                H = self.feature_extractor(x)
                A_V = self.attention_V(H)  # NxL
                A_U = self.attention_U(H)  # NxL
                A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
                A = torch.transpose(A, 1, 0)  # KxN
                '''

                x = x.squeeze(0)
                # H = self.linear_2(self.dropout_2(self.linear_1(self.dropout_1(self.feat_ext_part_1(x)))))
                H = self.linear_2(self.dropout_2(self.linear_1(self.dropout_1(self.feat_ext_part_1(x)))))
                # TODO : try to remove dropout layers in inference mode
                A_V = self.att_V_2(self.att_V_1(H))
                A_U = self.att_U_2(self.att_U_1(H))
                A = self.weig(A_V * A_U)
                A = torch.transpose(A, 1, 0)  # KxN
                return H, A

            elif self.part_2:
                A = F.softmax(A, dim=1)  # softmax over N
                M = torch.mm(A, H)  # KxM
                # Y_prob = self.classifier(M)
                Y_prob = self.class_2(self.class_1(M))

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


def PreActResNet50():
    # return PreActResNet(PreActBottleneck, [3, 4, 6, 3])
    return PreActResNet(PreActBottleneck, [3, 4, 6, 2])


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=500):
        super(PreActResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(128 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
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
        #feat = out
        out = self.dropout(self.linear(out))
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
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet50_GatedAttention_Ron(nn.Module):
    def __init__(self):
        super(ResNet50_GatedAttention_Ron, self).__init__()
        self.M = 500
        self.L = 128
        self.K = 1  # in the paper referred a 1.

        self.infer = False
        self.part_1 = False
        self.part_2 = False

        self.feature_extractor_ResNet50 = PreActResNet50()

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

    def forward(self, x, H=None, A=None):
        if not self.infer:
            x = x.squeeze(0)
            # In case we are training the model we'll use bags that contains only part of the tiles.
            H = self.feature_extractor_ResNet50(x)

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

class ResNet50_2(nn.Module):
    def __init__(self):
        super(ResNet50_2, self).__init__()

        self.num_classes = 2
        self.part_1 = models.resnet50(pretrained=True)
        self.dropout_1 = nn.Dropout(p=0.5)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.linear_1 = nn.Linear(in_features=1000, out_features=700)
        self.linear_2 = nn.Linear(in_features=700, out_features=self.num_classes)

    def forward(self, x):
        x = x.squeeze(0)
        x = self.part_1(x)
        out = self.dropout_2(self.linear_2(self.dropout_1(self.linear_1(x))))
        return out


class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 29 * 29, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 29 * 29)
        H = self.feature_extractor_part2(H)  # NxL

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
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


####################################################################################################################


class MyGroupNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyGroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=num_channels,
                                 eps=1e-5, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return x

'''
def ResNet34_GN():
    return resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3], num_classes=1000, zero_init_residual=False,
                      groups=1, width_per_group=64, replace_stride_with_dilation=None,
                      norm_layer=MyGroupNorm)
'''

class ResNet34_GN_GatedAttention(nn.Module):
    def __init__(self):
        super(ResNet34_GN_GatedAttention, self).__init__()
        self.model_name = 'ResNet34_GN_GatedAttention'
        print('Using model {}'.format(self.model_name))
        self.M = 500
        self.L = 128
        self.K = 1  # in the paper referred a 1.

        self.infer = False
        self.part_1 = False
        self.part_2 = False

        self.feat_ext_part_1 = ResNet34_GN()
        self.linear_1 = nn.Linear(in_features=1000, out_features=700)
        self.linear_2 = nn.Linear(in_features=700, out_features=self.M)
        self.att_V_1 = nn.Linear(self.M, self.L)
        self.att_V_2 = nn.Tanh()
        self.att_U_1 = nn.Linear(self.M, self.L)
        self.att_U_2 = nn.Sigmoid()
        self.class_1 = nn.Linear(self.M * self.K, 1)
        self.class_2 = nn.Sigmoid()
        self.weig = nn.Linear(self.L, self.K)


    def forward(self, x, H=None, A=None):
        if not self.infer:
            x = x.squeeze(0)
            H = self.linear_2(self.linear_1(self.feat_ext_part_1(x)))
            # In case we are training the model we'll use bags that contains only part of the tiles.
            ### H = self.feature_extractor(x)

            A_V = self.att_V_2(self.att_V_1(H))
            ### A_V = self.attention_V(H)  # NxL
            A_U = self.att_U_2(self.att_U_1(H))
            ### A_U = self.attention_U(H)  # NxL
            A = self.weig(A_V * A_U)
            ### A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N

            M = torch.mm(A, H)  # KxM

            # Because this is a binary classifier, the output of it is one single number which can be interpreted as the
            # probability that the input belong to class 1/TRUE (and not 0/FALSE)
            Y_prob = self.class_2(self.class_1(M))
            ### Y_prob = self.classifier(M)

            # The following line just turns probability to class.
            Y_class = torch.ge(Y_prob, 0.5).float()

            return Y_prob, Y_class, A

        # In case we want an inference of a whole slide we need ALL the tiles from that slide:
        else:
            if not self.part_1 ^ self.part_2:
                raise Exception('Inference Mode should include feature extraction (part 1) OR classification (part 2)')
            if self.part_1:
                x = x.squeeze(0)
                H = self.linear_2(self.linear_1(self.feat_ext_part_1(x)))

                # TODO : try to remove dropout layers in inference mode
                A_V = self.att_V_2(self.att_V_1(H))
                A_U = self.att_U_2(self.att_U_1(H))
                A = self.weig(A_V * A_U)
                A = torch.transpose(A, 1, 0)  # KxN
                return H, A

            elif self.part_2:
                A = F.softmax(A, dim=1)  # softmax over N
                M = torch.mm(A, H)  # KxM
                # Y_prob = self.classifier(M)
                Y_prob = self.class_2(self.class_1(M))

                Y_class = torch.ge(Y_prob, 0.5).float()
                return Y_prob, Y_class, A


class ResNet50_GN_GatedAttention(nn.Module):
    def __init__(self):
        super(ResNet50_GN_GatedAttention, self).__init__()
        self.model_name = 'ResNet50_GN_GatedAttention'
        print('Using model {}'.format(self.model_name))
        print('As Feature Extractor, the model will be ', end='')

        self.M = 500
        self.L = 128
        self.K = 1  # in the paper referred a 1.

        self.infer = False
        self.part_1 = False
        self.part_2 = False

        self.feat_ext_part_1 = ResNet50_GN().con_layers
        self.linear_1 = nn.Linear(in_features=1000, out_features=700)
        self.linear_2 = nn.Linear(in_features=700, out_features=self.M)
        self.att_V_1 = nn.Linear(self.M, self.L)
        self.att_V_2 = nn.Tanh()
        self.att_U_1 = nn.Linear(self.M, self.L)
        self.att_U_2 = nn.Sigmoid()
        self.class_1 = nn.Linear(self.M * self.K, 1)
        self.class_2 = nn.Sigmoid()
        self.weig = nn.Linear(self.L, self.K)


    def forward(self, x, H=None, A=None):
        if not self.infer:
            x = x.squeeze(0)
            H = self.linear_2(self.linear_1(self.feat_ext_part_1(x)))
            # In case we are training the model we'll use bags that contains only part of the tiles.
            ### H = self.feature_extractor(x)

            A_V = self.att_V_2(self.att_V_1(H))
            ### A_V = self.attention_V(H)  # NxL
            A_U = self.att_U_2(self.att_U_1(H))
            ### A_U = self.attention_U(H)  # NxL
            A = self.weig(A_V * A_U)
            ### A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N

            M = torch.mm(A, H)  # KxM

            # Because this is a binary classifier, the output of it is one single number which can be interpreted as the
            # probability that the input belong to class 1/TRUE (and not 0/FALSE)
            Y_prob = self.class_2(self.class_1(M))
            ### Y_prob = self.classifier(M)

            # The following line just turns probability to class.
            Y_class = torch.ge(Y_prob, 0.5).float()

            return Y_prob, Y_class, A

        # In case we want an inference of a whole slide we need ALL the tiles from that slide:
        else:
            if not self.part_1 ^ self.part_2:
                raise Exception('Inference Mode should include feature extraction (part 1) OR classification (part 2)')
            if self.part_1:
                x = x.squeeze(0)
                H = self.linear_2(self.linear_1(self.feat_ext_part_1(x)))

                # TODO : try to remove dropout layers in inference mode
                A_V = self.att_V_2(self.att_V_1(H))
                A_U = self.att_U_2(self.att_U_1(H))
                A = self.weig(A_V * A_U)
                A = torch.transpose(A, 1, 0)  # KxN
                return H, A

            elif self.part_2:
                A = F.softmax(A, dim=1)  # softmax over N
                M = torch.mm(A, H)  # KxM
                # Y_prob = self.classifier(M)
                Y_prob = self.class_2(self.class_1(M))

                Y_class = torch.ge(Y_prob, 0.5).float()
                return Y_prob, Y_class, A


class ResNet50_GN_GatedAttention_MultiBag(nn.Module):
    def __init__(self,
                 num_bags: int = 1,
                 tiles: int = 50):
        super(ResNet50_GN_GatedAttention_MultiBag, self).__init__()
        self.model_name = 'ResNet50_GN_GatedAttention_MultiBag'
        print('Using model {}'.format(self.model_name))
        print('As Feature Extractor, the model will be ', end='')

        self.num_bags = num_bags
        self.tiles = tiles
        self.M = 500
        self.L = 128
        self.K = 1  # in the paper referred a 1.

        self.feat_ext_part1 = ResNet50_GN().con_layers
        self.linear_1 = nn.Linear(in_features=1000, out_features=self.M)
        self.att_V_1 = nn.Linear(self.M, self.L)
        self.att_V_2 = nn.Tanh()
        self.att_U_1 = nn.Linear(self.M, self.L)
        self.att_U_2 = nn.Sigmoid()
        self.class_1 = nn.Linear(self.M * self.K, 1)
        self.class_2 = nn.Sigmoid()
        self.class_10 = nn.Linear(self.M * self.K, 2)
        self.weig = nn.Linear(self.L, self.K)


    def forward(self, x):
        bag_size, tiles_amount, _, tiles_size, _ = x.shape
        print(x.shape, end='')
        x = torch.reshape(x, (bag_size * tiles_amount, 3, tiles_size, tiles_size))
        print(x.shape)

        H = self.linear_1(self.feat_ext_part1(x))  # After this, H will contain all tiles for all bags as feature vectors

        A_V = self.att_V_2(self.att_V_1(H))

        A_U = self.att_U_2(self.att_U_1(H))

        A = self.weig(A_V * A_U)

        A = torch.transpose(A, 1, 0)  # KxN
        M = torch.zeros(bag_size, self.M)
        '''
        if not self.training:
            A = F.softmax(A, dim=1)
            M = torch.mm(A, H)
            Y_prob = self.class_10(M)

            Y_class = torch.ge(Y_prob, 0.5).float()  # This line just turns probability to class.
            Y_class_1D = Y_class[:, 0]

            return Y_prob#, Y_class_1D, A
        '''

        for i in range(bag_size):
            first_tile_idx = i * self.tiles
            a = A[:, first_tile_idx : first_tile_idx + self.tiles]
            a = F.softmax(a, dim=1)

            h = H[first_tile_idx : first_tile_idx + self.tiles, :]
            m = torch.mm(a, h)

            M[i, :] = m

        # Because this is a binary classifier, the output of it is one single number which can be interpreted as the
        # probability that the input belong to class 1/TRUE (and not 0/FALSE)
        '''Y_prob = self.class_2(self.class_1(M))
        Y_class = torch.ge(Y_prob, 0.5).float()'''

        Y_prob = self.class_10(M)

        Y_class = torch.ge(Y_prob, 0.5).float()  # This line just turns probability to class.
        Y_class_1D = Y_class[:, 0]

        return Y_prob# , Y_class_1D, A


class ResNet50_GN_GatedAttention_MultiBag_2(nn.Module):
    def __init__(self,
                 num_bags: int = 2,
                 tiles: int = 50):
        super(ResNet50_GN_GatedAttention_MultiBag_2, self).__init__()
        self.model_name = 'ResNet50_GN_GatedAttention_MultiBag'
        print('Using model {}'.format(self.model_name))
        print('As Feature Extractor, the model will be ', end='')

        self.num_bags = 2
        self.tiles = tiles
        self.M = 500
        self.L = 128
        self.K = 1  # in the paper referred a 1.

        self.feat_net_2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=9, stride=4, padding=0)
        self.linear_2 = nn.Linear(in_features=900, out_features=500)

        '''
        self.feat_ext_part1 = ResNet50_GN().con_layers
        self.linear_1 = nn.Linear(in_features=1000, out_features=self.M)
        '''
        self.att_V_1 = nn.Linear(self.M, self.L)
        self.att_V_2 = nn.Tanh()
        self.att_U_1 = nn.Linear(self.M, self.L)
        self.att_U_2 = nn.Sigmoid()
        self.class_1 = nn.Linear(self.M * self.K, 1)
        self.class_2 = nn.Sigmoid()
        self.class_10 = nn.Linear(self.M * self.K, 2)
        self.weig = nn.Linear(self.L, self.K)


    def forward(self, x):
        bag_size, tiles_amount, _, tiles_size, _ = x.shape

        x = torch.reshape(x, (bag_size * tiles_amount, 3, tiles_size, tiles_size))
        print('Before conv:', x.shape)
        x = self.feat_net_2(x)
        print('after conv:', x.shape)
        x = torch.flatten(x, 1)
        print('after flatten:', x.shape)
        H = self.linear_2(x)
        print('after linear:', x.shape)

        #H = self.linear_1(self.feat_ext_part1(x))  # After this, H will contain all tiles for all bags as feature vectors

        A_V = self.att_V_2(self.att_V_1(H))

        A_U = self.att_U_2(self.att_U_1(H))

        A = self.weig(A_V * A_U)

        A = torch.transpose(A, 1, 0)  # KxN
        M = torch.zeros(bag_size, self.M)

        if not self.training:
            A = F.softmax(A, dim=1)
            M = torch.mm(A, H)
            Y_prob = self.class_10(M)

            Y_class = torch.ge(Y_prob, 0.5).float()  # This line just turns probability to class.
            Y_class_1D = Y_class[:, 0]

            return Y_prob  # , Y_class_1D, A

        a1 = A[:, : self.tiles]
        a2 = A[:, self.tiles :]

        a1 = F.softmax(a1, dim=1)
        a2 = F.softmax(a2, dim=1)

        h1 = H[: self.tiles, :]
        h2 = H[self.tiles :, :]

        m1 = torch.mm(a1, h1)
        m2 = torch.mm(a2, h2)

        M[0, :] = m1
        M[1, :] = m2

        # Because this is a binary classifier, the output of it is one single number which can be interpreted as the
        # probability that the input belong to class 1/TRUE (and not 0/FALSE)
        '''Y_prob = self.class_2(self.class_1(M))
        Y_class = torch.ge(Y_prob, 0.5).float()'''

        Y_prob = self.class_10(M)

        Y_class = torch.ge(Y_prob, 0.5).float()  # This line just turns probability to class.
        Y_class_1D = Y_class[:, 0]

        return Y_prob# , Y_class_1D, A


class ResNet50_GN_GatedAttention_MultiBag_1(nn.Module):
    def __init__(self,
                 num_bags: int = 2,
                 tiles: int = 50):
        super(ResNet50_GN_GatedAttention_MultiBag_1, self).__init__()
        self.model_name = 'ResNet50_GN_GatedAttention_MultiBag'
        print('Using model {}'.format(self.model_name))
        print('As Feature Extractor, the model will be ', end='')

        self.num_bags = 1
        self.tiles = tiles
        self.M = 500
        self.L = 128
        self.K = 1  # in the paper referred a 1.

        self.feat_ext_part1 = ResNet50_GN().con_layers
        self.linear_1 = nn.Linear(in_features=1000, out_features=self.M)
        self.att_V_1 = nn.Linear(self.M, self.L)
        self.att_V_2 = nn.Tanh()
        self.att_U_1 = nn.Linear(self.M, self.L)
        self.att_U_2 = nn.Sigmoid()
        self.class_1 = nn.Linear(self.M * self.K, 1)
        self.class_2 = nn.Sigmoid()
        self.class_10 = nn.Linear(self.M * self.K, 2)
        self.weig = nn.Linear(self.L, self.K)

    def forward(self, x):
        bag_size, tiles_amount, _, tiles_size, _ = x.shape

        x = torch.reshape(x, (bag_size * tiles_amount, 3, tiles_size, tiles_size))

        H = self.linear_1(
            self.feat_ext_part1(x))  # After this, H will contain all tiles for all bags as feature vectors

        A_V = self.att_V_2(self.att_V_1(H))

        A_U = self.att_U_2(self.att_U_1(H))

        A = self.weig(A_V * A_U)

        A = torch.transpose(A, 1, 0)  # KxN

        A = F.softmax(A, dim=1)
        M = torch.mm(A, H)
        Y_prob = self.class_10(M)

        Y_class = torch.ge(Y_prob, 0.5).float()  # This line just turns probability to class.
        Y_class_1D = Y_class[:, 0]

        return Y_prob  # , Y_class_1D, A



        # Because this is a binary classifier, the output of it is one single number which can be interpreted as the
        # probability that the input belong to class 1/TRUE (and not 0/FALSE)
        '''Y_prob = self.class_2(self.class_1(M))
        Y_class = torch.ge(Y_prob, 0.5).float()'''

        Y_prob = self.class_10(M)

        Y_class = torch.ge(Y_prob, 0.5).float()  # This line just turns probability to class.
        Y_class_1D = Y_class[:, 0]

        return Y_prob  # , Y_class_1D, A


#RanS 21.12.20, based on ReceptorNet from the review paper
#https://www.nature.com/articles/s41467-020-19334-3

class ReceptorNet(nn.Module):
    def __init__(self):
        super(ReceptorNet, self).__init__()
        self.model_name = 'ReceptorNet'
        print('Using model {}'.format(self.model_name))
        print('As Feature Extractor, the model will be ', end='')

        self.M = 512
        self.L = 128
        self.K = 1  # in the paper referred a 1.

        self.infer = False
        self.infer_part = 0

        self.feat_ext_part_1 = ReceptorNet_feature_extractor()
        self.att_V_1 = nn.Linear(self.M, self.L)
        self.att_V_2 = nn.Tanh()
        self.class_1 = nn.Linear(self.M * self.K, 1)
        self.class_2 = nn.Sigmoid()
        self.weig = nn.Linear(self.L, self.K)

    def part1(self, x):
        x = x.squeeze(0)
        H = self.feat_ext_part_1(x)
        A_V = self.att_V_2(self.att_V_1(H))
        A = self.weig(A_V)
        A = torch.transpose(A, 1, 0)  # KxN
        return H, A

    def part2(self, A, H):
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, H)  # KxM
        Y_prob = self.class_2(self.class_1(M))

        Y_class = torch.ge(Y_prob, 0.5).float()
        return Y_prob, Y_class, A


    def forward(self, x, H=None, A=None):
        if not self.infer:
            H, A = self.part1(x)
            Y_prob, Y_class, A = self.part2(A, H)
            return Y_prob, Y_class, A

        # In case we want an inference of a whole slide we need ALL the tiles from that slide:
        else:
            if self.infer_part>2 or self.infer_part<1:
                raise Exception('Inference Mode should include feature extraction (part 1) or classification (part 2)')
            elif self.infer_part == 1:
                H, A = self.part1(x)
                return H, A

            elif self.infer_part == 2:
                Y_prob, Y_class, A = self.part2(A, H)
                return Y_prob, Y_class, A
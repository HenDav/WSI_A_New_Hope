import torch
import torch.nn as nn
import torch.nn.functional as F
import nets
import torchvision.models as models
import os
import PreActResNets

THIS_FILE = os.path.basename(os.path.realpath(__file__)).split('.')[0] + '.'


class MyGroupNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyGroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=num_channels,
                                 eps=1e-5, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return x


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


class MIL_PreActResNet50_GatedAttention_Ron(nn.Module):
    def __init__(self):
        super(MIL_PreActResNet50_GatedAttention_Ron, self).__init__()
        self.M = 500
        self.L = 128
        self.K = 1  # in the paper referred a 1.

        self.infer = False
        self.part_1 = False
        self.part_2 = False

        self.model_name = THIS_FILE + 'MIL_PreActResNet50_GatedAttention_Ron()'

        #self.feature_extractor_ResNet50 = PreActResNet50()
        self.feature_extractor = PreActResNets.MIL_PreActResNet50_Ron()

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
            #nn.Linear(self.M * self.K, 1),
            nn.Linear(self.M * self.K, 2),
            #nn.Sigmoid()
        )

    def forward(self, x, H=None, A=None):
        if not self.infer:
            num_of_bags, tiles_amount, _, tiles_size, _ = x.shape
            x = x.squeeze(0)
            # In case we are training the model we'll use bags that contains only part of the tiles.
            #H = self.feature_extractor_ResNet50(x)
            H = self.feature_extractor(x)

            A_V = self.attention_V(H)  # NxL
            A_U = self.attention_U(H)  # NxL
            A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N

            M = torch.mm(A, H)  # KxM

            '''
            # Because this is a binary classifier, the output of it is one single number which can be interpreted as the
            # probability that the input belong to class 1/TRUE (and not 0/FALSE)
            Y_prob = self.classifier(M)

            # The following line just turns probability to class.
            Y_class = torch.ge(Y_prob, 0.5).float()
            return Y_prob, Y_class, A
            '''

            # The output of this net is a 2 score vector (one for each class)
            out = self.classifier(M)

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


'''
def ResNet34_GN():
    return resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3], num_classes=1000, zero_init_residual=False,
                      groups=1, width_per_group=64, replace_stride_with_dilation=None,
                      norm_layer=MyGroupNorm)
'''

class ResNet34_GN_GatedAttention(nn.Module):
    def __init__(self):
        super(ResNet34_GN_GatedAttention, self).__init__()
        self.model_name = THIS_FILE + 'ResNet34_GN_GatedAttention()'
        print('Using model {}'.format(self.model_name))
        self.M = 500
        self.L = 128
        self.K = 1  # in the paper referred a 1.

        self.infer = False
        self.part_1 = False
        self.part_2 = False

        self.feat_ext_part_1 = nets.ResNet34_GN()
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
        self.model_name = THIS_FILE + 'ResNet50_GN_GatedAttention()'
        print('Using model {}'.format(self.model_name))
        print('As Feature Extractor, the model will be ', end='')

        self.M = 500
        self.L = 128
        self.K = 1  # in the paper referred a 1.

        self.infer = False
        self.part_1 = False
        self.part_2 = False

        self.feat_ext_part_1 = nets.ResNet50_GN().con_layers
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



class MIL_PreActResNet50_Ron_MultiBag(nn.Module):
    def __init__(self,
                 tiles_per_bag: int = 10):
        super(MIL_PreActResNet50_Ron_MultiBag, self).__init__()

        self.model_name = THIS_FILE + 'MIL_PreActResNet50_Ron_MultiBag()'
        print('Using model {}'.format(self.model_name))

        self.infer = False
        self.features_part = False

        self.tiles_per_bag = tiles_per_bag
        self.M = 500
        self.L = 128
        self.K = 1  # in the paper referred a 1.

        # is_HeatMap is used when we want to create a heatmap and we need to fkip the order of the last two layers
        self.is_HeatMap = False  # Omer 27/7/2021

        self.feat_ext_part1 = PreActResNets.MIL_PreActResNet50_Ron()

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
            # nn.Linear(self.M * self.K, 1),
            # nn.Sigmoid()
            nn.Linear(self.M * self.K, 2)
        )


    def forward(self, x, H = None, A = None):
        if not self.infer:  # Training mode
            num_of_bags, tiles_amount, _, tiles_size, _ = x.shape

            x = torch.reshape(x, (num_of_bags * tiles_amount, 3, tiles_size, tiles_size))
            H = self.feat_ext_part1(x)
            print(H.shape, type(H), H.dtype)
            A_V = self.attention_V(H)  # NxL
            A_U = self.attention_U(H)  # NxL
            A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            #A = F.softmax(A, dim=1)  # softmax over N

            if torch.cuda.is_available():
                M = torch.zeros(0).cuda()
                A_after_sftmx = torch.zeros(0).cuda()
            else:
                M = torch.zeros(0)
                A_after_sftmx = torch.zeros(0)

            # The following if statement is needed in cases where the accuracy checking (testing phase) is done in
            # a 1 bag per mini-batch mode
            if num_of_bags == 1 and not self.training:
                A_after_sftmx = F.softmax(A, dim=1)
                M = torch.mm(A_after_sftmx, H)

            else:
                for i in range(num_of_bags):
                    first_tile_idx = i * self.tiles_per_bag
                    a = A[:, first_tile_idx : first_tile_idx + self.tiles_per_bag]
                    a = F.softmax(a, dim=1)

                    h = H[first_tile_idx : first_tile_idx + self.tiles_per_bag, :]
                    m = torch.mm(a, h)
                    M = torch.cat((M, m))

                    A_after_sftmx = torch.cat((A_after_sftmx, a))

            out = self.classifier(M)
            return out, A_after_sftmx

        # In inference mode, there is no need to use more than one bag in each mini-batch because the data variability
        # is needed only for the training process.
        else:  # Inference mode
            if self.features_part:
                num_of_bags, tiles_amount, _, tiles_size, _ = x.shape

                if num_of_bags != 1:
                    raise Exception('Inference mode supports only 1 bag(Slide) per Mini Batch')

                x = torch.reshape(x, (num_of_bags * tiles_amount, 3, tiles_size, tiles_size))
                H = self.feat_ext_part1(x)

                A_V = self.attention_V(H)  # NxL
                A_U = self.attention_U(H)  # NxL
                A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
                A = torch.transpose(A, 1, 0)  # KxN

                return H, A

            else:
                A_after_sftmx = F.softmax(A, dim=1)  # softmax over N
                M = torch.mm(A_after_sftmx, H)
                out = self.classifier(M)

                return out, A_after_sftmx


class MIL_Feature_Attention_MultiBag(nn.Module):
    def __init__(self,
                 tiles_per_bag: int = 500):
        super(MIL_Feature_Attention_MultiBag, self).__init__()

        self.model_name = THIS_FILE + 'MIL_Feature_Attention_MultiBag()'
        print('Using model {}'.format(self.model_name))

        self.features_only = True
        self.infer = False
        self.features_part = False

        self.tiles_per_bag = tiles_per_bag
        self.M = 512
        self.L = 128
        self.K = 1  # in the paper referred a 1.

        if not self.features_only:  # if we're working only on features than we don't need the conv net
            self.feat_ext_part1 = PreActResNets.MIL_PreActResNet50_Ron()

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.L, self.K)

        self.classifier = nn.Linear(self.M * self.K, 2)
        '''self.classifier = nn.Sequential(
            # nn.Linear(self.M * self.K, 1),
            # nn.Sigmoid()
            nn.Linear(self.M * self.K, 2)
        )'''

    def forward(self, x, H=None, A=None):
        if True: #not self.infer:  # Training mode
            if not self.features_only:
                num_of_bags, tiles_amount, _, tiles_size, _ = x.shape

                x = torch.reshape(x, (num_of_bags * tiles_amount, 3, tiles_size, tiles_size))
                H = self.feat_ext_part1(x)
            else:
                H_shape = H.shape
                if len(H_shape) == 2:
                    num_of_bags = 1
                    tiles_amount = H_shape[0]
                    DividedSlides_Flag = False
                elif len(H_shape) == 3:
                    num_of_bags = H_shape[0]
                    tiles_amount = H_shape[1]
                    DividedSlides_Flag = True

                if tiles_amount != self.tiles_per_bag and self.infer == False:
                    raise Exception('Declared tiles per bag is different than the input (tiles_amount')

                if x != None:
                    raise Exception('Model in features only mode expects to get x=None and H=features')

            A_V = self.attention_V(H)  # NxL
            A_U = self.attention_U(H)  # NxL
            A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK

            if DividedSlides_Flag:  # DividedSlides_Flag tells if all the feature from all slides are gathered together in the same dimension or divided between dimensions
                A = torch.transpose(A, 2, 1)
            else:
                A = torch.transpose(A, 1, 0)  # KxN

            # A = F.softmax(A, dim=1)  # softmax over N

            if torch.cuda.is_available():
                M = torch.zeros(0).cuda()
                A_after_sftmx = torch.zeros(0).cuda()
            else:
                M = torch.zeros(0)
                A_after_sftmx = torch.zeros(0)

            '''# The following if statement is needed in cases where the accuracy checking (testing phase) is done in
            # a 1 bag per mini-batch mode
            if num_of_bags == 1 and not self.training:
                A_after_sftmx = F.softmax(A, dim=1)
                M = torch.mm(A_after_sftmx, H)

            else:'''
            if DividedSlides_Flag:
                for i in range(num_of_bags):
                    a = A[i, :, :]
                    a = F.softmax(a, dim=1)

                    h = H[i, :, :]
                    m = torch.mm(a, h)
                    M = torch.cat((M, m))

                    A_after_sftmx = torch.cat((A_after_sftmx, a))
            else:
                for i in range(num_of_bags):
                    first_tile_idx = i * self.tiles_per_bag
                    a = A[:, first_tile_idx: first_tile_idx + self.tiles_per_bag]
                    a = F.softmax(a, dim=1)

                    h = H[first_tile_idx: first_tile_idx + self.tiles_per_bag, :]
                    m = torch.mm(a, h)
                    M = torch.cat((M, m))

                    A_after_sftmx = torch.cat((A_after_sftmx, a))

            out = self.classifier(M)

            return out, A_after_sftmx, A.squeeze(0)

        # In inference mode, there is no need to use more than one bag in each mini-batch because the data variability
        # is needed only for the training process.
        else:  # Inference mode
            if self.features_part:
                raise Exception('wrong part')
                num_of_bags, tiles_amount, _, tiles_size, _ = x.shape

                if num_of_bags != 1:
                    raise Exception('Inference mode supports only 1 bag(Slide) per Mini Batch')

                x = torch.reshape(x, (num_of_bags * tiles_amount, 3, tiles_size, tiles_size))
                H = self.feat_ext_part1(x)

                A_V = self.attention_V(H)  # NxL
                A_U = self.attention_U(H)  # NxL
                A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
                A = torch.transpose(A, 1, 0)  # KxN

                return H, A

            else:
                A_after_sftmx = F.softmax(A, dim=1)  # softmax over N
                M = torch.mm(A_after_sftmx, H)
                out = self.classifier(M)

                return out, A_after_sftmx


class Combined_MIL_Feature_Attention_MultiBag(nn.Module):
    def __init__(self,
                 tiles_per_bag: int = 500):
        super(Combined_MIL_Feature_Attention_MultiBag, self).__init__()

        self.model_name = THIS_FILE + 'Combined_MIL_Feature_Attention_MultiBag()'
        print('Using model {}'.format(self.model_name))

        self.zero_carmel_weights = False

        self.infer = False
        self.features_part = False

        self.tiles_per_bag = tiles_per_bag
        self.free_bias = False
        self.M = 512
        self.L = 128
        self.K = 1  # in the paper referred a 1.

        # Defining layers for the 1st MIL model:
        self.attention_V = nn.ModuleDict({
            'CAT': nn.Sequential(nn.Linear(self.M, self.L),
                                 nn.Tanh()
                                 ),
            'CARMEL': nn.Sequential(nn.Linear(self.M, self.L),
                                    nn.Tanh()
                                    )
        })
        self.attention_U = nn.ModuleDict({
            'CAT': nn.Sequential(nn.Linear(self.M, self.L),
                                 nn.Sigmoid()
                                 ),
            'CARMEL': nn.Sequential(nn.Linear(self.M, self.L),
                                    nn.Sigmoid()
                                    )
        })
        self.attention_weights = nn.ModuleDict({
            'CAT': nn.Linear(self.L, self.K),
            'CARMEL': nn.Linear(self.L, self.K)
        })
        self.classifier = nn.ModuleDict({
            'CAT': nn.Linear(self.M * self.K, 2),
            'CARMEL': nn.Linear(self.M * self.K, 2)
        })

        self.key_list = list(self.attention_U.keys())


    def zeroize_carmel_weights(self):
        self.attention_weights['CARMEL'].bias.data = torch.zeros_like(self.attention_weights['CARMEL'].bias.data)
        self.attention_weights['CARMEL'].weight.data = torch.zeros_like(self.attention_weights['CARMEL'].weight.data)
        self.attention_weights['CARMEL'].bias.requires_grad = False
        self.attention_weights['CARMEL'].weight.requires_grad = False
        self.zero_carmel_weights = True

    def create_free_bias(self):
        self.free_bias_layer = nn.Linear(2, 2)
        self.free_bias_layer.weight.data = torch.eye(2)
        self.free_bias_layer.weight.requires_grad = False
        self.free_bias_layer.bias.requires_grad = True
        self.free_bias = True

    def forward(self, x, H=None, A=None):

        H_shape = H[self.key_list[0]].shape
        if len(H_shape) == 2:
            num_of_bags = 1
            tiles_amount = H_shape[0]
            DividedSlides_Flag = False
        elif len(H_shape) == 3:
            num_of_bags = H_shape[0]
            tiles_amount = H_shape[1]
            DividedSlides_Flag = True

        if tiles_amount != self.tiles_per_bag and self.infer is False:
            raise Exception('Declared tiles per bag is different than the input (tiles_amount')

        if x != None:
            raise Exception('Model in features only mode expects to get x=None and H=features')


        A_V, A_U, A, M, A_after_sftmx = {}, {}, {}, {}, {}
        bias_relative_part = {}
        a, h = {}, {}

        for key in self.key_list:
            if key == 'CARMEL':  # DEBUGGING
                continue

            A_V[key] = self.attention_V[key](H[key])  # NxL
            A_U[key] = self.attention_U[key](H[key])  # NxL
            A[key] = self.attention_weights[key](A_V[key] * A_U[key])  # element wise multiplication # NxK

            if DividedSlides_Flag:  # DividedSlides_Flag tells if all the feature from all slides are gathered together in the same dimension or divided between dimensions
                A[key] = torch.transpose(A[key], 2, 1)
            else:
                A[key] = torch.transpose(A[key], 1, 0)  # KxN


            # A = F.softmax(A, dim=1)  # softmax over N

            if torch.cuda.is_available():
                M[key] = torch.zeros(0).cuda()
                A_after_sftmx[key] = torch.zeros(0).cuda()
                bias_relative_part[key] = torch.zeros(0).cuda()
            else:
                M[key] = torch.zeros(0)
                A_after_sftmx[key] = torch.zeros(0)
                bias_relative_part[key] = torch.zeros(0)

        '''# The following if statement is needed in cases where the accuracy checking (testing phase) is done in
        # a 1 bag per mini-batch mode
        if num_of_bags == 1 and not self.training:
            A_after_sftmx = F.softmax(A, dim=1)
            M = torch.mm(A_after_sftmx, H)

        else:'''
        if DividedSlides_Flag:
            for i in range(num_of_bags):

                '''if self.zero_carmel_weights:                    
                    if A[self.key_list[0]][i, :, :].sum().item() == 0:
                        A[self.key_list[0]][i, :, :] = torch.ones_like(A[self.key_list[0]][i, :, :]) * -1e9
                    elif A[self.key_list[1]][i, :, :].sum().item() == 0:
                        A[self.key_list[1]][i, :, :] = torch.ones_like(A[self.key_list[1]][i, :, :]) * -1e9'''

                #aa = torch.cat([A[self.key_list[0]][i, :, :], A[self.key_list[1]][i, :, :]], dim=1)
                aa = A['CAT'][i, :, :]  # DEBUGGING
                aa = F.softmax(aa, dim=1)
                a['CAT'] = aa[:, :tiles_amount]  # DEBUGGING
                #a['CARMEL'] = aa[:, tiles_amount:]  # DEBUGGING
                '''a[self.key_list[0]] = aa[:, :tiles_amount]
                a[self.key_list[1]] = aa[:, tiles_amount:]'''

                for key in self.key_list:
                    if key == 'CARMEL':  # DEBUGGING
                        continue
                    bias_relative_part[key] = torch.cat((bias_relative_part[key], torch.reshape(a[key].sum().detach(), (1,))))

                    h = H[key][i, :, :]
                    try:
                        m = torch.mm(a[key], h)
                    except RuntimeError as e:
                        raise e

                    M[key] = torch.cat((M[key], m))

                    A_after_sftmx[key] = torch.cat((A_after_sftmx[key], a[key]))
        else:
            raise Exception('Not Implemented')
            '''
            for i in range(num_of_bags):
                first_tile_idx = i * self.tiles_per_bag
                a = A[:, first_tile_idx: first_tile_idx + self.tiles_per_bag]
                a = F.softmax(a, dim=1)

                h = H[first_tile_idx: first_tile_idx + self.tiles_per_bag, :]
                m = torch.mm(a, h)
                M = torch.cat((M, m))

                A_after_sftmx = torch.cat((A_after_sftmx, a))
            '''

        # Before using the classifier we'll change the bias to 0 and than add it manually after using the weights for the bias
        DEVICE = self.classifier['CAT'].bias.device
        out = {}
        if self.free_bias is False:
            bias = {}
            for key in self.key_list:
                if key == 'CARMEL':  # DEBUGGING
                    continue
                bias[key] = torch.tensor([self.classifier[key].bias[0].item(), self.classifier[key].bias[1].item()])
                self.classifier[key].bias.data = torch.tensor([0, 0], dtype=torch.float32, device=DEVICE)
                out[key] = self.classifier[key](M[key])

                # Computing and adding the weighted bias:
                new_bias = torch.zeros_like(out[key])
                new_bias[:, 0] = bias[key][0] * bias_relative_part[key]
                new_bias[:, 1] = bias[key][1] * bias_relative_part[key]

                out[key] += new_bias
        else:
            for key in self.key_list:
                if key == 'CARMEL':  # DEBUGGING
                    continue
                self.classifier[key].bias.data = torch.tensor([0, 0], dtype=torch.float32, device=DEVICE)
                out[key] = self.classifier[key](M[key])
                # Adding the free bias:
                out[key] = self.free_bias_layer(out[key])

        #out_total = out[self.key_list[0]] + out[self.key_list[1]]
        out_total = out['CAT']  # DEBUGGING

        A_after_sftmx['CARMEL'] = torch.zeros_like(A_after_sftmx['CAT'])  # DEBUGGING
        A['CARMEL'] = torch.zeros_like(A['CAT'])  # DEBUGGING

        return out_total, A_after_sftmx, A


class Combined_MIL_Feature_Attention_MultiBag_DEBUG(nn.Module):
    def __init__(self,
                 tiles_per_bag: int = 500):
        super(Combined_MIL_Feature_Attention_MultiBag_DEBUG, self).__init__()

        self.model_name = THIS_FILE + 'Combined_MIL_Feature_Attention_MultiBag_DEBUG()'
        print('Using model {}'.format(self.model_name))

        self.infer = False
        self.Model_1_only = False

        self.tiles_per_bag = tiles_per_bag
        self.free_bias = False
        self.M = 512
        self.L = 128
        self.K = 1  # in the paper referred a 1.

        # Defining layers for the 1st MIL model:
        # Modules for CAT:
        self.attention_V_Model_1 = nn.Sequential(nn.Linear(self.M, self.L),
                                                 nn.Tanh()
                                                 )
        self.attention_U_Model_1 = nn.Sequential(nn.Linear(self.M, self.L),
                                                 nn.Sigmoid()
                                                 )
        self.attention_weights_Model_1 = nn.Linear(self.L, self.K)
        self.classifier_Model_1 = nn.Linear(self.M * self.K, 2)
        self.bias_Model_1 = None

        # Modules for CARMEL:
        self.attention_V_Model_2 = nn.Sequential(nn.Linear(self.M, self.L),
                                                 nn.Tanh()
                                                 )
        self.attention_U_Model_2 = nn.Sequential(nn.Linear(self.M, self.L),
                                                 nn.Sigmoid()
                                                 )
        self.attention_weights_Model_2 = nn.Linear(self.L, self.K)
        self.classifier_Model_2 = nn.Linear(self.M * self.K, 2)
        self.bias_Model_2 = None


    def forward(self, x, H=None, A=None):

        dataset_list = list(H.keys())
        H_shape = H[dataset_list[0]].shape
        if len(H_shape) == 2:
            num_of_bags = 1
            tiles_amount = H_shape[0]
            DividedSlides_Flag = False
        elif len(H_shape) == 3:
            num_of_bags = H_shape[0]
            tiles_amount = H_shape[1]
            DividedSlides_Flag = True

        if tiles_amount != self.tiles_per_bag and self.infer is False:
            raise Exception('Declared tiles per bag is different than the input (tiles_amount')

        if x != None:
            raise Exception('Model in features only mode expects to get x=None and H=features')

        A_V_Model_1 = self.attention_V_Model_1(H[dataset_list[0]])  # NxL
        A_U_Model_1 = self.attention_U_Model_1(H[dataset_list[0]])  # NxL
        A_Model_1 = self.attention_weights_Model_1(A_V_Model_1 * A_U_Model_1)  # element wise multiplication # NxK

        if not self.Model_1_only:
            A_V_Model_2 = self.attention_V_Model_2(H[dataset_list[1]])  # NxL
            A_U_Model_2 = self.attention_U_Model_2(H[dataset_list[1]])  # NxL
            A_Model_2 = self.attention_weights_Model_2(A_V_Model_2 * A_U_Model_2)  # element wise multiplication # NxK

        if DividedSlides_Flag:  # DividedSlides_Flag tells if all the feature from all slides are gathered together in the same dimension or divided between dimensions
            A_Model_1 = torch.transpose(A_Model_1, 2, 1)
            if not self.Model_1_only:
                A_Model_2 = torch.transpose(A_Model_2, 2, 1) # Zeroizing CARMEL
        else:
            A_Model_1 = torch.transpose(A_Model_1, 1, 0)  # KxN
            if not self.Model_1_only:
                A_Model_2 = torch.transpose(A_Model_2, 1, 0)  # Zeroizing CARMEL

        '''if torch.cuda.is_available():
            M_CAT = torch.zeros(0).cuda()
            A_after_sftmx_CAT = torch.zeros(0).cuda()
            bias_relative_part_CAT = torch.zeros(0).cuda()

            if not self.CAT_only:
                M_CARMEL = torch.zeros(0).cuda()
                A_after_sftmx_CARMEL = torch.zeros(0).cuda()
                bias_relative_part_CARMEL = torch.zeros(0).cuda()
        else:
            M_CAT = torch.zeros(0)
            A_after_sftmx_CAT = torch.zeros(0)
            bias_relative_part_CAT = torch.zeros(0)

            if not self.CAT_only:
                M_CARMEL = torch.zeros(0)
                A_after_sftmx_CARMEL = torch.zeros(0)
                bias_relative_part_CARMEL = torch.zeros(0)'''

        if DividedSlides_Flag:
            if self.Model_1_only:
                A_total = A_Model_1
            else:
                A_total = torch.cat([A_Model_1, A_Model_2], dim=2)

            A_after_sftmx = F.softmax(A_total, dim=2).squeeze(1)

            A_after_sftmx_Model_1 = A_after_sftmx[:, :tiles_amount]
            bias_relative_part_Model_1 = A_after_sftmx_Model_1.sum(dim=1)
            M_Model_1 = torch.matmul(A_after_sftmx_Model_1.unsqueeze(1), H[dataset_list[0]]).squeeze(1)
            if not self.Model_1_only:
                A_after_sftmx_Model_2 = A_after_sftmx[:, tiles_amount:]
                bias_relative_part_Model_2 = A_after_sftmx_Model_2.sum(dim=1)
                M_Model_2 = torch.matmul(A_after_sftmx_Model_2.unsqueeze(1), H[dataset_list[1]]).squeeze(1)

            '''for i in range(num_of_bags):
                aa = A_CAT[i, :, :]  # Zeroizing CARMEL   torch.cat([A_CAT[i, :, :], A_CARMEL[i, :, :]], dim=1)
                aa = F.softmax(aa, dim=1)
                a_CAT = aa[:, :tiles_amount]
                if not self.CAT_only:
                    a_CARMEL = aa[:, tiles_amount:]

                bias_relative_part_CAT = torch.cat((bias_relative_part_CAT, torch.reshape(a_CAT.sum().detach(), (1,))))
                h_CAT = H['CAT'][i, :, :]
                m_CAT = torch.mm(a_CAT, h_CAT)
                M_CAT = torch.cat((M_CAT, m_CAT))
                A_after_sftmx_CAT = torch.cat((A_after_sftmx_CAT, a_CAT))

                if not self.CAT_only:
                    bias_relative_part_CARMEL = torch.cat((bias_relative_part_CARMEL, torch.reshape(a_CARMEL.sum().detach(), (1,))))
                    h_CARMEL = H['CARMEL'][i, :, :]
                    m_CARMEL = torch.mm(a_CARMEL, h_CARMEL)
                    M_CARMEL = torch.cat((M_CARMEL, m_CARMEL))
                    A_after_sftmx_CARMEL = torch.cat((A_after_sftmx_CARMEL, a_CARMEL))'''
        else:
            raise Exception('Not Implemented')

        # Before using the classifier we'll change the bias to 0 and than add it manually after using the weights for the bias
        DEVICE = self.classifier_Model_1.bias.device

        if self.free_bias is False:
            if self.bias_Model_1 is None:
                self.bias_Model_1 = torch.tensor([self.classifier_Model_1.bias[0].item(), self.classifier_Model_1.bias[1].item()])
                self.classifier_Model_1.bias.data = torch.tensor([0, 0], dtype=torch.float32, device=DEVICE)
                self.bias_Model_2 = torch.tensor([self.classifier_Model_2.bias[0].item(), self.classifier_Model_2.bias[1].item()])
                self.classifier_Model_2.bias.data = torch.tensor([0, 0], dtype=torch.float32, device=DEVICE)

            out_Model_1 = self.classifier_Model_1(M_Model_1)

            if not self.Model_1_only:
                out_Model_2 = self.classifier_Model_2(M_Model_2)

            # Computing and adding the weighted bias:
            new_bias_Model_1 = torch.zeros_like(out_Model_1)
            new_bias_Model_1[:, 0] = self.bias_Model_1[0] * bias_relative_part_Model_1
            new_bias_Model_1[:, 1] = self.bias_Model_1[1] * bias_relative_part_Model_1

            if not self.Model_1_only:
                new_bias_Model_2 = torch.zeros_like(out_Model_2)
                new_bias_Model_2[:, 0] = self.bias_Model_2[0] * bias_relative_part_Model_2
                new_bias_Model_2[:, 1] = self.bias_Model_2[1] * bias_relative_part_Model_2

            out_Model_1 += new_bias_Model_1
            if not self.Model_1_only:
                out_Model_2 += new_bias_Model_2  # Zeroizing CARMEL
        else:
            print('NOT IMPLEMENTED !')
            '''for key in self.key_list:
                self.classifier[key].bias.data = torch.tensor([0, 0], dtype=torch.float32, device=DEVICE)
                out[key] = self.classifier[key](M[key])
                # Adding the free bias:
                out[key] = self.free_bias_layer(out[key])'''

        if self.Model_1_only:
            out_total = out_Model_1 #+ out_CARMEL # Zeroizing CARMEL
            return out_total, [A_after_sftmx_Model_1], [A_Model_1]
        else:
            out_total = out_Model_1 + out_Model_2
            return out_total, [A_after_sftmx_Model_1, A_after_sftmx_Model_2], [A_Model_1, A_Model_2]


class Combined_MIL_Feature_Attention_MultiBag_Class_Relation_FIxation(nn.Module):
    def __init__(self,
                 tiles_per_bag: int = 500,
                 relation: float = None):
        super(Combined_MIL_Feature_Attention_MultiBag_Class_Relation_FIxation, self).__init__()

        self.model_name = THIS_FILE + 'Combined_MIL_Feature_Attention_MultiBag_Class_Relation_FIxation()'
        print('Using model {}'.format(self.model_name))

        self.infer = False
        self.Model_1_only = False

        self.tiles_per_bag = tiles_per_bag
        self.relation = relation
        self.free_bias = False
        self.M = 512
        self.L = 128
        self.K = 1  # in the paper referred a 1.

        # Defining layers for the 1st MIL model:
        # Modules for CAT:
        self.attention_V_Model_1 = nn.Sequential(nn.Linear(self.M, self.L),
                                                 nn.Tanh()
                                                 )
        self.attention_U_Model_1 = nn.Sequential(nn.Linear(self.M, self.L),
                                                 nn.Sigmoid()
                                                 )
        self.attention_weights_Model_1 = nn.Linear(self.L, self.K)
        self.classifier_Model_1 = nn.Linear(self.M * self.K, 2)
        self.bias_Model_1 = None

        # Modules for CARMEL:
        self.attention_V_Model_2 = nn.Sequential(nn.Linear(self.M, self.L),
                                                 nn.Tanh()
                                                 )
        self.attention_U_Model_2 = nn.Sequential(nn.Linear(self.M, self.L),
                                                 nn.Sigmoid()
                                                 )
        self.attention_weights_Model_2 = nn.Linear(self.L, self.K)
        self.classifier_Model_2 = nn.Linear(self.M * self.K, 2)
        self.bias_Model_2 = None


    def forward(self, x, H=None, A=None):

        dataset_list = list(H.keys())
        H_shape = H[dataset_list[0]].shape
        if len(H_shape) == 2:
            num_of_bags = 1
            tiles_amount = H_shape[0]
            DividedSlides_Flag = False
        elif len(H_shape) == 3:
            num_of_bags = H_shape[0]
            tiles_amount = H_shape[1]
            DividedSlides_Flag = True

        if tiles_amount != self.tiles_per_bag and self.infer is False:
            raise Exception('Declared tiles per bag is different than the input (tiles_amount')

        if x != None:
            raise Exception('Model in features only mode expects to get x=None and H=features')

        A_V_Model_1 = self.attention_V_Model_1(H[dataset_list[0]])  # NxL
        A_U_Model_1 = self.attention_U_Model_1(H[dataset_list[0]])  # NxL
        A_Model_1 = self.attention_weights_Model_1(A_V_Model_1 * A_U_Model_1)  # element wise multiplication # NxK

        if not self.Model_1_only:
            A_V_Model_2 = self.attention_V_Model_2(H[dataset_list[1]])  # NxL
            A_U_Model_2 = self.attention_U_Model_2(H[dataset_list[1]])  # NxL
            A_Model_2 = self.attention_weights_Model_2(A_V_Model_2 * A_U_Model_2)  # element wise multiplication # NxK

        if DividedSlides_Flag:  # DividedSlides_Flag tells if all the feature from all slides are gathered together in the same dimension or divided between dimensions
            A_Model_1 = torch.transpose(A_Model_1, 2, 1)
            if not self.Model_1_only:
                A_Model_2 = torch.transpose(A_Model_2, 2, 1) # Zeroizing CARMEL
        else:
            A_Model_1 = torch.transpose(A_Model_1, 1, 0)  # KxN
            if not self.Model_1_only:
                A_Model_2 = torch.transpose(A_Model_2, 1, 0)  # Zeroizing CARMEL

        '''if torch.cuda.is_available():
            M_CAT = torch.zeros(0).cuda()
            A_after_sftmx_CAT = torch.zeros(0).cuda()
            bias_relative_part_CAT = torch.zeros(0).cuda()

            if not self.CAT_only:
                M_CARMEL = torch.zeros(0).cuda()
                A_after_sftmx_CARMEL = torch.zeros(0).cuda()
                bias_relative_part_CARMEL = torch.zeros(0).cuda()
        else:
            M_CAT = torch.zeros(0)
            A_after_sftmx_CAT = torch.zeros(0)
            bias_relative_part_CAT = torch.zeros(0)

            if not self.CAT_only:
                M_CARMEL = torch.zeros(0)
                A_after_sftmx_CARMEL = torch.zeros(0)
                bias_relative_part_CARMEL = torch.zeros(0)'''

        if DividedSlides_Flag:
            if self.relation is None:
                if self.Model_1_only:
                    A_total = A_Model_1
                else:
                    A_total = torch.cat([A_Model_1, A_Model_2], dim=2)

                A_after_sftmx = F.softmax(A_total, dim=2).squeeze(1)
                A_after_sftmx_Model_1 = A_after_sftmx[:, :tiles_amount]
                if not self.Model_1_only:
                    A_after_sftmx_Model_2 = A_after_sftmx[:, tiles_amount:]

            else:
                A_after_sftmx_Model_1 = F.softmax(A_Model_1, dim=2).squeeze(1)
                A_after_sftmx_Model_2 = F.softmax(A_Model_2, dim=2).squeeze(1)

            bias_relative_part_Model_1 = A_after_sftmx_Model_1.sum(dim=1)
            M_Model_1 = torch.matmul(A_after_sftmx_Model_1.unsqueeze(1), H[dataset_list[0]]).squeeze(1)
            if not self.Model_1_only:
                bias_relative_part_Model_2 = A_after_sftmx_Model_2.sum(dim=1)
                M_Model_2 = torch.matmul(A_after_sftmx_Model_2.unsqueeze(1), H[dataset_list[1]]).squeeze(1)

            '''for i in range(num_of_bags):
                aa = A_CAT[i, :, :]  # Zeroizing CARMEL   torch.cat([A_CAT[i, :, :], A_CARMEL[i, :, :]], dim=1)
                aa = F.softmax(aa, dim=1)
                a_CAT = aa[:, :tiles_amount]
                if not self.CAT_only:
                    a_CARMEL = aa[:, tiles_amount:]

                bias_relative_part_CAT = torch.cat((bias_relative_part_CAT, torch.reshape(a_CAT.sum().detach(), (1,))))
                h_CAT = H['CAT'][i, :, :]
                m_CAT = torch.mm(a_CAT, h_CAT)
                M_CAT = torch.cat((M_CAT, m_CAT))
                A_after_sftmx_CAT = torch.cat((A_after_sftmx_CAT, a_CAT))

                if not self.CAT_only:
                    bias_relative_part_CARMEL = torch.cat((bias_relative_part_CARMEL, torch.reshape(a_CARMEL.sum().detach(), (1,))))
                    h_CARMEL = H['CARMEL'][i, :, :]
                    m_CARMEL = torch.mm(a_CARMEL, h_CARMEL)
                    M_CARMEL = torch.cat((M_CARMEL, m_CARMEL))
                    A_after_sftmx_CARMEL = torch.cat((A_after_sftmx_CARMEL, a_CARMEL))'''
        else:
            raise Exception('Not Implemented')

        # Before using the classifier we'll change the bias to 0 and than add it manually after using the weights for the bias
        DEVICE = self.classifier_Model_1.bias.device

        if self.free_bias is False:
            if self.bias_Model_1 is None:
                self.bias_Model_1 = torch.tensor([self.classifier_Model_1.bias[0].item(), self.classifier_Model_1.bias[1].item()])
                self.classifier_Model_1.bias.data = torch.tensor([0, 0], dtype=torch.float32, device=DEVICE)
                self.bias_Model_2 = torch.tensor([self.classifier_Model_2.bias[0].item(), self.classifier_Model_2.bias[1].item()])
                self.classifier_Model_2.bias.data = torch.tensor([0, 0], dtype=torch.float32, device=DEVICE)

            out_Model_1 = self.classifier_Model_1(M_Model_1)

            if not self.Model_1_only:
                out_Model_2 = self.classifier_Model_2(M_Model_2)

            # Computing and adding the weighted bias:
            new_bias_Model_1 = torch.zeros_like(out_Model_1)
            new_bias_Model_1[:, 0] = self.bias_Model_1[0] * bias_relative_part_Model_1
            new_bias_Model_1[:, 1] = self.bias_Model_1[1] * bias_relative_part_Model_1

            if not self.Model_1_only:
                new_bias_Model_2 = torch.zeros_like(out_Model_2)
                new_bias_Model_2[:, 0] = self.bias_Model_2[0] * bias_relative_part_Model_2
                new_bias_Model_2[:, 1] = self.bias_Model_2[1] * bias_relative_part_Model_2

            out_Model_1 += new_bias_Model_1
            if not self.Model_1_only:
                out_Model_2 += new_bias_Model_2  # Zeroizing CARMEL
        else:
            print('NOT IMPLEMENTED !')
            '''for key in self.key_list:
                self.classifier[key].bias.data = torch.tensor([0, 0], dtype=torch.float32, device=DEVICE)
                out[key] = self.classifier[key](M[key])
                # Adding the free bias:
                out[key] = self.free_bias_layer(out[key])'''

        if self.Model_1_only:
            out_total = out_Model_1 #+ out_CARMEL # Zeroizing CARMEL
            return out_total, [A_after_sftmx_Model_1], [A_Model_1]
        elif self.relation is None:
            out_total = out_Model_1 + out_Model_2
        else:
            out_total = self.relation * out_Model_1 + (1 - self.relation) * out_Model_2

        return out_total, [A_after_sftmx_Model_1, A_after_sftmx_Model_2], [A_Model_1, A_Model_2]



class MIL_Feature_2_Attention_MultiBag(nn.Module):
    def __init__(self,
                 tiles_per_bag: int = 500):
        super(MIL_Feature_2_Attention_MultiBag, self).__init__()

        self.model_name = THIS_FILE + 'MIL_Feature_2_Attention_MultiBag()'
        print('Using model {}'.format(self.model_name))

        self.infer = False

        self.tiles_per_bag = tiles_per_bag
        self.M = 512
        self.L = 128
        self.K = 1  # in the paper referred a 1.


        self.pre_layers = nn.Sequential(
            nn.Linear(self.M, self.M),
            nn.ReLU()
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
            # nn.Linear(self.M * self.K, 1),
            # nn.Sigmoid()
            nn.Linear(self.M * self.K, 2)
        )

    def forward(self, x, H=None, A=None):

        H_shape = H.shape
        if len(H_shape) == 2:
            num_of_bags = 1
            tiles_amount = H_shape[0]
            DividedSlides_Flag = False
        elif len(H_shape) == 3:
            num_of_bags = H_shape[0]
            tiles_amount = H_shape[1]
            DividedSlides_Flag = True

        if tiles_amount != self.tiles_per_bag and self.infer is False:
            print('tiles_amount is {} and self.tiles_per_bag is {}'.format(tiles_amount, self.tiles_per_bag))
            raise Exception('Declared tiles per bag is different than the input (tiles_amount')

        if x != None:
            raise Exception('Feature Model expects to get x=None and H=features')

        H = self.pre_layers(H)
        A_V = self.attention_V(H)  # NxL
        A_U = self.attention_U(H)  # NxL
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        if DividedSlides_Flag:  # DividedSlides_Flag tells if all the feature from all slides are gathered together in the same dimension or divided between dimensions
            A = torch.transpose(A, 2, 1)
        else:
            A = torch.transpose(A, 1, 0)  # KxN

        # A = F.softmax(A, dim=1)  # softmax over N

        if torch.cuda.is_available():
            M = torch.zeros(0).cuda()
            A_after_sftmx = torch.zeros(0).cuda()
        else:
            M = torch.zeros(0)
            A_after_sftmx = torch.zeros(0)

        '''# The following if statement is needed in cases where the accuracy checking (testing phase) is done in
        # a 1 bag per mini-batch mode
        if num_of_bags == 1 and not self.training:
            A_after_sftmx = F.softmax(A, dim=1)
            M = torch.mm(A_after_sftmx, H)

        else:'''
        if DividedSlides_Flag:
            for i in range(num_of_bags):
                a = A[i, :, :]
                a = F.softmax(a, dim=1)

                h = H[i, :, :]
                m = torch.mm(a, h)
                M = torch.cat((M, m))

                A_after_sftmx = torch.cat((A_after_sftmx, a))
        else:
            for i in range(num_of_bags):
                first_tile_idx = i * self.tiles_per_bag
                a = A[:, first_tile_idx: first_tile_idx + self.tiles_per_bag]
                a = F.softmax(a, dim=1)

                h = H[first_tile_idx: first_tile_idx + self.tiles_per_bag, :]
                m = torch.mm(a, h)
                M = torch.cat((M, m))

                A_after_sftmx = torch.cat((A_after_sftmx, a))

        out = self.classifier(M)
        return out, A_after_sftmx


class MIL_Feature_3_Attention_MultiBag(nn.Module):
    def __init__(self,
                 tiles_per_bag: int = 500):
        super(MIL_Feature_3_Attention_MultiBag, self).__init__()

        self.model_name = THIS_FILE + 'MIL_Feature_3_Attention_MultiBag()'
        print('Using model {}'.format(self.model_name))

        self.infer = False

        self.tiles_per_bag = tiles_per_bag
        self.M = 512
        self.L = 128
        self.K = 1  # in the paper referred a 1.


        self.pre_layers = nn.Sequential(
            nn.Linear(self.M, self.M),
            nn.ReLU(),
            nn.Linear(self.M, self.M),
            nn.ReLU()
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
            # nn.Linear(self.M * self.K, 1),
            # nn.Sigmoid()
            nn.Linear(self.M * self.K, 2)
        )

    def forward(self, x, H=None, A=None):

        H_shape = H.shape
        if len(H_shape) == 2:
            num_of_bags = 1
            tiles_amount = H_shape[0]
            DividedSlides_Flag = False
        elif len(H_shape) == 3:
            num_of_bags = H_shape[0]
            tiles_amount = H_shape[1]
            DividedSlides_Flag = True

        if tiles_amount != self.tiles_per_bag and self.infer is False:
            print('tiles_amount is {} and self.tiles_per_bag is {}'.format(tiles_amount, self.tiles_per_bag))
            raise Exception('Declared tiles per bag is different than the input (tiles_amount')

        if x != None:
            raise Exception('Feature Model expects to get x=None and H=features')

        H = self.pre_layers(H)
        A_V = self.attention_V(H)  # NxL
        A_U = self.attention_U(H)  # NxL
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        if DividedSlides_Flag:  # DividedSlides_Flag tells if all the feature from all slides are gathered together in the same dimension or divided between dimensions
            A = torch.transpose(A, 2, 1)
        else:
            A = torch.transpose(A, 1, 0)  # KxN

        # A = F.softmax(A, dim=1)  # softmax over N

        if torch.cuda.is_available():
            M = torch.zeros(0).cuda()
            A_after_sftmx = torch.zeros(0).cuda()
        else:
            M = torch.zeros(0)
            A_after_sftmx = torch.zeros(0)

        '''# The following if statement is needed in cases where the accuracy checking (testing phase) is done in
        # a 1 bag per mini-batch mode
        if num_of_bags == 1 and not self.training:
            A_after_sftmx = F.softmax(A, dim=1)
            M = torch.mm(A_after_sftmx, H)

        else:'''
        if DividedSlides_Flag:
            for i in range(num_of_bags):
                a = A[i, :, :]
                a = F.softmax(a, dim=1)

                h = H[i, :, :]
                m = torch.mm(a, h)
                M = torch.cat((M, m))

                A_after_sftmx = torch.cat((A_after_sftmx, a))
        else:
            for i in range(num_of_bags):
                first_tile_idx = i * self.tiles_per_bag
                a = A[:, first_tile_idx: first_tile_idx + self.tiles_per_bag]
                a = F.softmax(a, dim=1)

                h = H[first_tile_idx: first_tile_idx + self.tiles_per_bag, :]
                m = torch.mm(a, h)
                M = torch.cat((M, m))

                A_after_sftmx = torch.cat((A_after_sftmx, a))

        out = self.classifier(M)
        return out, A_after_sftmx


class ResNet50_GatedAttention_MultiBag(nn.Module):
    def __init__(self,
                 num_bags: int = 2,
                 tiles: int = 50):
        super(ResNet50_GatedAttention_MultiBag, self).__init__()

        self.model_name = THIS_FILE + 'ResNet50_GatedAttention_MultiBag()'
        print('Using model {}'.format(self.model_name))

        self.num_bags = num_bags
        self.tiles = tiles
        self.M = 500
        self.L = 128
        self.K = 1  # in the paper referred a 1.

        #self.feat_ext_part1 = nets.ResNet50_GN().con_layers
        self.feat_ext_part1 = nets.ResNet50(num_classes=self.M)

        #self.linear_1 = nn.Linear(in_features=1000, out_features=self.M)
        self.att_V_1 = nn.Linear(self.M, self.L)
        self.att_V_2 = nn.Tanh()
        self.att_U_1 = nn.Linear(self.M, self.L)
        self.att_U_2 = nn.Sigmoid()
        self.class_1 = nn.Linear(self.M * self.K, 1)
        self.class_2 = nn.Sigmoid()
        #self.class_10 = nn.Linear(self.M * self.K, 2)
        self.weig = nn.Linear(self.L, self.K)


    def forward(self, x):
        #print('Before: ', x.shape, end=' ')
        num_of_bags, tiles_amount, _, tiles_size, _ = x.shape

        x = torch.reshape(x, (num_of_bags * tiles_amount, 3, tiles_size, tiles_size))
        #print('After: ', x.shape)

        #H = self.linear_1(self.feat_ext_part1(x))  # After this, H will contain all tiles for all bags as feature vectors
        H = self.feat_ext_part1(x)
        A_V = self.att_V_2(self.att_V_1(H))
        A_U = self.att_U_2(self.att_U_1(H))
        A = self.weig(A_V * A_U)
        A = torch.transpose(A, 1, 0)  # KxN

        if torch.cuda.is_available():
            M = torch.zeros(0).cuda()
            A_after_sftmx = torch.zeros(0).cuda()
        else:
            M = torch.zeros(0)
            A_after_sftmx = torch.zeros(0)

        # The following if statement is needed in cases where the accuracy checking is done in a 1 bag per minibatch mode
        if num_of_bags == 1 and not self.training:
            A = F.softmax(A, dim=1)
            M = torch.mm(A, H)
            Y_prob = self.class_2(self.class_1(M))  #self.class_10(M)

            Y_class = torch.ge(Y_prob, 0.5).float()  # This line just turns probability to class.
            Y_class_1D = Y_class[:, 0]

            return Y_prob, Y_class_1D, A


        for i in range(num_of_bags):
            first_tile_idx = i * self.tiles
            a = A[:, first_tile_idx : first_tile_idx + self.tiles]
            a = F.softmax(a, dim=1)

            h = H[first_tile_idx : first_tile_idx + self.tiles, :]
            m = torch.mm(a, h)

            M = torch.cat((M, m))
            A_after_sftmx = torch.cat((A_after_sftmx, a))

        # Because this is a binary classifier, the output of it is one single number which can be interpreted as the
        # probability that the input belong to class 1/TRUE (and not 0/FALSE)
        '''Y_prob = self.class_2(self.class_1(M))
        Y_class = torch.ge(Y_prob, 0.5).float()'''

        Y_prob = self.class_2(self.class_1(M))  # self.class_10(M)

        Y_class = torch.ge(Y_prob, 0.5).float()  # This line just turns probability to class.
        Y_class_1D = Y_class[:, 0]

        return Y_prob, Y_class_1D, A_after_sftmx


class ResNet50_GatedAttention_MultiBag_Other_Loss(nn.Module):
    def __init__(self,
                 num_bags: int = 2,
                 tiles: int = 50):
        super(ResNet50_GatedAttention_MultiBag_Other_Loss, self).__init__()

        self.model_name = THIS_FILE + 'ResNet50_GatedAttention_MultiBag_Other_Loss()'
        print('Using model {}'.format(self.model_name))

        self.num_bags = num_bags
        self.tiles = tiles
        self.M = 2
        self.L = 128
        self.K = 1  # in the paper referred a 1.

        #self.feat_ext_part1 = nets.ResNet50_GN().con_layers
        self.feat_ext_part1 = nets.ResNet50(num_classes=self.M)

        #self.linear_1 = nn.Linear(in_features=1000, out_features=self.M)
        self.att_V_1 = nn.Linear(self.M, self.L)
        self.att_V_2 = nn.Tanh()
        self.att_U_1 = nn.Linear(self.M, self.L)
        self.att_U_2 = nn.Sigmoid()
        self.class_1 = nn.Linear(self.M * self.K, 2)
        self.class_2 = nn.Sigmoid()
        self.weig = nn.Linear(self.L, self.K)


    def forward(self, x):
        #print('Before: ', x.shape, end=' ')
        num_of_bags, tiles_amount, _, tiles_size, _ = x.shape
        x = torch.reshape(x, (num_of_bags * tiles_amount, 3, tiles_size, tiles_size))
        #print('After: ', x.shape)

        #H = self.linear_1(self.feat_ext_part1(x))  # After this, H will contain all tiles for all bags as feature vectors
        H = self.feat_ext_part1(x)
        A_V = self.att_V_2(self.att_V_1(H))
        A_U = self.att_U_2(self.att_U_1(H))
        A = self.weig(A_V * A_U)
        A = torch.transpose(A, 1, 0)  # KxN

        if torch.cuda.is_available():
            M = torch.zeros(0).cuda()
            A_after_sftmx = torch.zeros(0).cuda()
        else:
            M = torch.zeros(0)
            A_after_sftmx = torch.zeros(0)

        # The following if statement is needed in cases where the accuracy checking is done in a 1 bag per minibatch mode
        if num_of_bags == 1 and not self.training:
            A = F.softmax(A, dim=1)
            M = torch.mm(A, H)
            scores = M  #self.class_2(self.class_1(M))  #self.class_10(M)

            #Y_class = torch.ge(Y_prob, 0.5).float()  # This line just turns probability to class.
            #Y_class_1D = Y_class[:, 0]

            return scores  #Y_prob, Y_class_1D, A


        for i in range(num_of_bags):
            first_tile_idx = i * self.tiles
            a = A[:, first_tile_idx : first_tile_idx + self.tiles]
            a = F.softmax(a, dim=1)

            h = H[first_tile_idx : first_tile_idx + self.tiles, :]
            m = torch.mm(a, h)

            M = torch.cat((M, m))
            A_after_sftmx = torch.cat((A_after_sftmx, a))

        # Because this is a binary classifier, the output of it is one single number which can be interpreted as the
        # probability that the input belong to class 1/TRUE (and not 0/FALSE)
        '''Y_prob = self.class_2(self.class_1(M))
        Y_class = torch.ge(Y_prob, 0.5).float()'''

        scores = M  #self.class_2(self.class_1(M))  # self.class_10(M)

        #Y_class = torch.ge(Y_prob, 0.5).float()  # This line just turns probability to class.
        #Y_class_1D = Y_class[:, 0]

        return scores  #Y_prob, Y_class_1D, A_after_sftmx


class ResNet50_GN_GatedAttention_MultiBag_2(nn.Module):
    def __init__(self,
                 tiles: int = 50):
        super(ResNet50_GN_GatedAttention_MultiBag_2, self).__init__()

        self.model_name = THIS_FILE + 'ResNet50_GN_GatedAttention_MultiBag_2()'
        print('Using model {}'.format(self.model_name))
        print('As Feature Extractor, the model will be ', end='')

        self.num_bags = 2
        self.tiles = tiles
        self.M = 500
        self.L = 128
        self.K = 1  # in the paper referred as 1.

        self.feat_ext_part1 = ResNet50_GN().con_layers
        self.linear_1 = nn.Linear(in_features=1000, out_features=self.M)

        self.att_V_1 = nn.Linear(self.M, self.L)
        self.att_V_2 = nn.Tanh()
        self.att_U_1 = nn.Linear(self.M, self.L)
        self.att_U_2 = nn.Sigmoid()
        self.class_1 = nn.Linear(self.M * self.K, 1)
        self.class_2 = nn.Sigmoid()
        #self.class_10 = nn.Linear(self.M * self.K, 2)
        self.weig = nn.Linear(self.L, self.K)


    def forward(self, x):
        bag_size, tiles_amount, _, tiles_size, _ = x.shape

        x = torch.reshape(x, (bag_size * tiles_amount, 3, tiles_size, tiles_size))

        H = self.linear_1(self.feat_ext_part1(x))  # After this, H will contain all tiles for all bags as feature vectors

        A_V = self.att_V_2(self.att_V_1(H))

        A_U = self.att_U_2(self.att_U_1(H))

        A = self.weig(A_V * A_U)

        A = torch.transpose(A, 1, 0)  # KxN

        if torch.cuda.is_available():
            M = torch.zeros(0).cuda()
        else:
            M = torch.zeros(0)

        if not self.training:
            A = F.softmax(A, dim=1)
            M = torch.mm(A, H)
            Y_prob = self.class_2(self.class_1(M))  # self.class_10(M)

            Y_class = torch.ge(Y_prob, 0.5).float()  # This line just turns probability to class.
            Y_class_1D = Y_class[:, 0]

            return Y_prob, Y_class_1D  #, A

        a1 = A[:, : self.tiles]
        a2 = A[:, self.tiles :]

        a1 = F.softmax(a1, dim=1)
        a2 = F.softmax(a2, dim=1)

        h1 = H[: self.tiles, :]
        h2 = H[self.tiles :, :]

        m1 = torch.mm(a1, h1)
        m2 = torch.mm(a2, h2)

        M = torch.cat((M, m1, m2))

        #M[0, :] = m1
        #M[1, :] = m2

        # Because this is a binary classifier, the output of it is one single number which can be interpreted as the
        # probability that the input belong to class 1/TRUE (and not 0/FALSE)
        '''Y_prob = self.class_2(self.class_1(M))
        Y_class = torch.ge(Y_prob, 0.5).float()'''

        Y_prob = self.class_2(self.class_1(M))#self.class_10(M)

        Y_class = torch.ge(Y_prob, 0.5).float()  # This line just turns probability to class.
        Y_class_1D = Y_class[:, 0]

        return Y_prob, Y_class_1D#, A


class ResNet50_GN_GatedAttention_MultiBag_1(nn.Module):
    def __init__(self,
                 tiles: int = 50):
        super(ResNet50_GN_GatedAttention_MultiBag_1, self).__init__()
        self.model_name = THIS_FILE + 'ResNet50_GN_GatedAttention_MultiBag_1()'
        print('Using model {}'.format(self.model_name))
        print('As Feature Extractor, the model will be ', end='')

        self.num_bags = 1
        self.tiles = tiles
        self.M = 128#500
        self.L = 64#128
        self.K = 1  # in the paper referred as 1.

        self.feat_net_2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=9, stride=4, padding=0)
        self.linear_2 = nn.Linear(in_features=3844, out_features=self.M)

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

        #print('Before conv:', x.shape)
        x = self.feat_net_2(x)
        #print('after conv:', x.shape)
        x = torch.flatten(x, 1)
        #print('after flatten:', x.shape)
        H = self.linear_2(x)
        #print('after linear:', H.shape)
        '''
        print('Before conv:', x.shape)
        x = self.feat_ext_part1(x)
        print('after conv:', x.shape)
        H = self.linear_1(x)
        print('after linear:', x.shape)
        '''
        #H = self.linear_1(self.feat_ext_part1(x))  # After this, H will contain all tiles for all bags as feature vectors

        A_V = self.att_V_2(self.att_V_1(H))

        A_U = self.att_U_2(self.att_U_1(H))

        A = self.weig(A_V * A_U)

        A = torch.transpose(A, 1, 0)  # KxN
        #M = torch.zeros(bag_size, self.M)

        if torch.cuda.is_available():
            M = torch.zeros(0).cuda()
        else:
            M = torch.zeros(0)

        if not self.training:
            A = F.softmax(A, dim=1)
            M = torch.mm(A, H)
            Y_prob = self.class_2(self.class_1(M))  # self.class_10(M)

            Y_class = torch.ge(Y_prob, 0.5).float()  # This line just turns probability to class.
            Y_class_1D = Y_class[:, 0]

            return Y_prob, Y_class_1D #, A

        a1 = A[:, : self.tiles]


        a1 = F.softmax(a1, dim=1)


        h1 = H[: self.tiles, :]

        #print('before: ', M.shape)
        #M = torch.mm(a1, h1)
        m1 = torch.mm(a1, h1)
        print('m1 GPU:, ', m1.is_cuda)
        print('M GPU: ', M.is_cuda)
        M = torch.cat((M, m1))

        #print('after: ', M.shape)

        #print(m1.shape)


        #M[0, :] = m1


        # Because this is a binary classifier, the output of it is one single number which can be interpreted as the
        # probability that the input belong to class 1/TRUE (and not 0/FALSE)
        '''Y_prob = self.class_2(self.class_1(M))
        Y_class = torch.ge(Y_prob, 0.5).float()'''

        Y_prob = self.class_2(self.class_1(M))#self.class_10(M)

        Y_class = torch.ge(Y_prob, 0.5).float()  # This line just turns probability to class.
        Y_class_1D = Y_class[:, 0]

        return Y_prob, Y_class_1D#, A



#RanS 21.12.20, based on ReceptorNet from the review paper
#https://www.nature.com/articles/s41467-020-19334-3

class ReceptorNet(nn.Module):
    #def __init__(self):
    def __init__(self, feature_extractor, saved_model_path='none'): #RanS 6.1.21
        super(ReceptorNet, self).__init__()
        self.model_name = THIS_FILE + "ReceptorNet(feature_extractor='" + feature_extractor+ "',savel_model_path=" + saved_model_path + ')'
        #self.model_name = 'ReceptorNet_' + feature_extractor
        print('Using model {}'.format(self.model_name))
        print('As Feature Extractor, the model will be ', end='')

        self.M = 512
        self.L = 128
        self.K = 1  # in the paper referred a 1.

        self.infer = False
        self.infer_part = 0

        if feature_extractor == 'resnet50_2FC':
            self.feat_ext_part_1 = nets.ReceptorNet_feature_extractor()
        elif feature_extractor == 'preact_resnet50':
            self.M = 500
            self.feat_ext_part_1 = PreActResNets.PreActResNet50_Ron()

        if saved_model_path != 'none':
            model_data_loaded = torch.load(saved_model_path, map_location='cpu')
            saved_model = PreActResNets.PreActResNet50_Ron()
            saved_model.load_state_dict(model_data_loaded['model_state_dict'])
            saved_model.linear = nn.Linear(saved_model.linear.in_features, self.M)
            nn.init.kaiming_normal_(saved_model.linear.weight, mode='fan_in', nonlinearity='relu')
            self.feat_ext_part_1 = saved_model

            print('loaded saved model from: ' + saved_model_path)

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
            if self.infer_part > 2 or self.infer_part < 1:
                raise Exception('Inference Mode should include feature extraction (part 1) or classification (part 2)')
            elif self.infer_part == 1:
                H, A = self.part1(x)
                return H, A

            elif self.infer_part == 2:
                Y_prob, Y_class, A = self.part2(A, H)
                return Y_prob, Y_class, A




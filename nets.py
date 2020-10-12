import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Flatten(nn.Module):
    """
    This class flattens an array to a vector
    """
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class ResNet50_GatedAttention(nn.Module):
    def __init__(self):
    #def __init__(self, tile_size: int = 256):
        super(ResNet50_GatedAttention, self).__init__()
        #self.tile_size = tile_size
        self.M = 500
        self.L = 128
        self.K = 1    # in the paper referred a 1.

        self.infer = False
        self.get_features = False
        self.get_classification = False

        self._feature_extractor_ResNet50_part_1 = models.resnet50()

        self._feature_extractor_fc = nn.Sequential(
            nn.Dropout(),
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

    def forward(self, x):
        x = x.squeeze(0)
        # In case we are training the model we'll use bags that contains only part of the tiles.
        if not self.infer:
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
            if not self.get_features ^ self.get_classification:
                raise Exception('Inference Mode should include feature extraction OR classification')
            if self.get_features:
                H = self.feature_extractor(x)
                return H

            elif self.get_classification:
                A_V = self.attention_V(H)  # NxL
                A_U = self.attention_U(H)  # NxL
                A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
                A = torch.transpose(A, 1, 0)  # KxN
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
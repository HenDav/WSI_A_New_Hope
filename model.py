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


class GatedAttention(nn.Module):
    def __init__(self, tile_size: int = 7):
        super(GatedAttention, self).__init__()
        self.tile_size = tile_size
        self.M = 500
        self.L = 128
        self.K = 1    # in the paper referred a 1.


        self._feature_extractor_ResNet50_part_1 = models.resnet50()

        self._feature_extractor_fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=1000, out_features=self.M)
        )
        
        self.feature_extractor_ResNet50 = nn.Sequential(
            self._feature_extractor_ResNet50_part_1,
            self._feature_extractor_fc
        )
        """
        self.feature_extractor_basic_2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # This layer don't change the size of input tiles.
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # This layer don't change the size of input tiles.
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # This layer don't change the size of input tiles.
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # This layer don't change the size of input tiles.
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.Dropout(0.25),
            Flatten(),  # flattening from 7 X 7 X 64
            nn.Linear(self.tile_size * self.tile_size * 128, self.M),
            nn.ReLU()
        )
        
        self.feature_extractor_basic_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # This layer don't change the size of input tiles.
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # This layer don't change the size of input tiles.
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.Dropout(0.25),
            Flatten(),  # flattening from 7 X 7 X 64
            nn.Linear(self.tile_size * self.tile_size * 64, self.M),
            nn.ReLU()
        )
        """
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

        H = self.feature_extractor_ResNet50(x)
        # H = self.feature_extractor_basic_1(x)  # NxM
        """H = H.view(-1, 50 * 4 * 4) 
        H = self.feature_extractor_part2(H)  # NxL """

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
        #Y_class = torch.tensor(Y_prob.data[0][0] < Y_prob.data[0][1]).float()

        return Y_prob, Y_class, A

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
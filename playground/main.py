# python peripherals
import os
import sys

# numpy
import numpy as np
# pandas

# ipython

# matplotlib

# pytorch
import torch

if __name__ == '__main__':
    t = torch.tensor([[[3,8],[4,3]],[[2,5],[6,6]],[[4,1],[3,1]]])
    t2 = t * 2
    b = t.reshape([t.shape[0] * t.shape[1], t.shape[2]])
    b2 = b * 2
    b3 = b2.reshape([t.shape[0], t.shape[1], t.shape[2]])
    k = 5
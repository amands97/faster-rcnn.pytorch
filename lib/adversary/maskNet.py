import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math


import torch.nn.functional as F
# from .unet_parts import *
# from unetparts import *


class MaskMan(nn.Module):
    """ Full assembly of the parts to form the complete network """


# class UNet(nn.Module):
    def __init__(self, n_channels=512, n_classes=1):
        super(MaskMan, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(512,128,1,1,0)
        self.conv2 = nn.Conv2d(128,1,1,1,0)
        self.relu1 = nn.PReLU(128)
        self.relu2 = nn.PReLU(1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x

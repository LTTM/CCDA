from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import torch

# noinspection PyTypeChecker
class FCNClassifier(nn.Module):
    def __init__(self, inplanes, num_classes):
        super(FCNClassifier, self).__init__()

        self.cast = nn.Conv2d(inplanes, num_classes, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        return self.cast(x)
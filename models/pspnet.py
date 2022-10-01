from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import torch

class PSPNetPyramidPath(nn.Module):
    def __init__(self, inplanes, pool_bins):
        super(PSPNetPyramidPath, self).__init__()
        self.cast = nn.Conv2d(inplanes, 1, kernel_size=1, padding=0, stride=1) 
        self.pool_bins = pool_bins
    
    def forward(self, x):
        out_size = x.shape[2:]
        ratio = out_size[0]/out_size[1]
        pool_size = (round(ratio*self.pool_bins), self.pool_bins) if ratio >= 1 else (self.pool_bins, round(self.pool_bins/ratio)) 
        out = F.adaptive_avg_pool2d(x, pool_size) # smallest dimension is in [1,2,3,6], bigger dimension follows to maintain the aspect ratio
        out = self.cast(out)
        out = F.interpolate(out, out_size, mode='bilinear', align_corners=True)
        return out

# noinspection PyTypeChecker
class PSPNetClassifier(nn.Module):
    def __init__(self, inplanes, num_classes):
        super(PSPNetClassifier, self).__init__()
        
        self.path1 = PSPNetPyramidPath(inplanes, 1)
        self.path2 = PSPNetPyramidPath(inplanes, 2)
        self.path3 = PSPNetPyramidPath(inplanes, 3)
        self.path4 = PSPNetPyramidPath(inplanes, 6)
        
        self.cast = nn.Conv2d(inplanes+4, num_classes, kernel_size=1, padding=0, stride=1)

    def forward(self, x):

        outplanes = [x, self.path1(x), self.path2(x), self.path3(x), self.path4(x)]
        out = torch.cat(outplanes, dim=1)
        out = self.cast(out)
        
        return out
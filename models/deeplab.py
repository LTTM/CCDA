from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import torch

# noinspection PyTypeChecker
class DeepLabV2Classifier(nn.Module):
    def __init__(self, inplanes, num_classes):
        super(DeepLabV2Classifier, self).__init__()

        self.conv2d_list = nn.ModuleList()

        dilation_series = padding_series = [6, 12, 18, 24]

        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out  # note that it was fixed !!


class MSIWDeepLabV2Classifier(nn.Module):
    def __init__(self, inplanes, num_classes):
        super(MSIWDeepLabV2Classifier, self).__init__()

        self.conv2d_list = nn.ModuleList()

        dilation_series = padding_series = [6, 12, 18, 24]

        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out  



class PooledBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(PooledBatchNorm2d, self).__init__(*args, **kwargs)
        if self.track_running_stats:
            self.register_buffer('welford_mean', torch.zeros(self.running_mean.shape, requires_grad=False,
                                 dtype=self.running_mean.dtype, device=self.running_mean.device))
            self.register_buffer('welford_msquare',torch.zeros(self.running_mean.shape, requires_grad=False,
                                 dtype=self.running_mean.dtype, device=self.running_mean.device))
            self.register_buffer('welford_var',torch.zeros(self.running_var.shape, requires_grad=False,
                                 dtype=self.running_mean.dtype, device=self.running_mean.device))
            self.register_buffer('welford_count', torch.tensor(0, dtype=torch.long, requires_grad=False,
                                 device=self.running_mean.device))

    def forward2d(self, input):
        return super(PooledBatchNorm2d, self).forward(input)

    # modified version of torch.nn._BatchNorm.forward()
    # normalizes with mean only
    def forward1d(self, input):
        if not self.track_running_stats:
            if self.affine:
                return input*self.weight.view(1,-1,1,1)+self.bias.view(1,-1,1,1)
            else:
                return input

        if self.num_batches_tracked > 1:
            return ((input-self.running_mean.view(1,-1,1,1))/torch.sqrt(self.running_var.view(1,-1,1,1)+self.eps))* \
                        self.weight.view(1,-1,1,1)+self.bias.view(1,-1,1,1)
        else:
            return input*self.weight.view(1,-1,1,1)+self.bias.view(1,-1,1,1)

    def update_welford(self, input):
        for b in range(input.shape[0]):
            self.welford_count += 1
            if self.welford_count > 1:
                x = input[b].detach().squeeze()
                d = x-self.welford_mean
                self.welford_mean += d/self.welford_count
                d2 = x-self.welford_mean
                self.welford_msquare += d*d2
                self.welford_var = self.welford_msquare/self.welford_count
            else:
                self.welford_mean += input[b].detach().squeeze()


    def forward(self, input):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            self.update_welford(input)
            if self.welford_count > 1:
                self.running_mean = (1.-exponential_average_factor)*self.running_mean+exponential_average_factor*self.welford_mean
                self.running_var = (1.-exponential_average_factor)*self.running_var+exponential_average_factor*self.welford_var

        if input.shape[0] > 1:
            return self.forward2d(input)
        else:
            return self.forward1d(input)

class DeepLabV3Classifier(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 hidden_channels=256,
                 out_stride=8,
                 pooling_size=None):
        super(DeepLabV3Classifier, self).__init__()
        self.pooling_size = pooling_size

        if out_stride == 16:
            dilations = [6, 12, 18]
        elif out_stride == 8:
            dilations = [12, 24, 32]

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilations[0], padding=dilations[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilations[1], padding=dilations[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilations[2], padding=dilations[2])
        ])
        self.map_bn = nn.BatchNorm2d(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = PooledBatchNorm2d(hidden_channels) #nn.BatchNorm2d(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, num_classes, 1, bias=False)
        self.red_conv_pool = nn.Conv2d(hidden_channels, num_classes, 1, bias=False)
        self.red_bn = nn.BatchNorm2d(num_classes)


    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)  # if training is global avg pooling 1x1, else use larger pool size
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)
        pool = self.red_conv_pool(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            # this is like Adaptive Average Pooling (1,1)
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = (min(self.try_index(self.pooling_size, 0), x.shape[2]),
                            min(self.try_index(self.pooling_size, 1), x.shape[3]))
            padding = (
                (pooling_size[1] - 1) // 2,
                (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1,
                (pooling_size[0] - 1) // 2,
                (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1
            )

            pool = F.avg_pool2d(x, pooling_size, stride=1)
            pool = F.pad(pool, pad=padding, mode="replicate")
        return pool

    @staticmethod
    def try_index(scalar_or_list, i):
        try:
            return scalar_or_list[i]
        except TypeError:
            return scalar_or_list

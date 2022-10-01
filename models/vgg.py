import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DeeplabVGG16(nn.Module):
    # noinspection PyTypeChecker
    def __init__(self, num_classes, restore_from, pretrained, clas):
        super(DeeplabVGG16, self).__init__()
        vgg = models.vgg16()
        if pretrained:
            vgg.load_state_dict(torch.load(restore_from))

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        features = nn.Sequential(*(features[i] for i in range(30) if i != 23))

        for i in [23,25,27]:
            features[i].dilation = (2,2)
            features[i].padding = (2,2)

        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)

        self.features = nn.Sequential(*([features[i] for i in range(len(features))] + [ fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))
        self.classifier = clas(1024, num_classes)


    def forward(self, x):
        input_size = x.size()[2:]
        feats = self.features(x)
        x = self.classifier(feats)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return x, feats


    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = [
            self.features
        ]

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = [
            self.classifier.parameters()
        ]

        for j in range(len(b)):
            for i in b[j]:
                yield i


    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.lr},
                {'params': self.get_10x_lr_params(), 'lr': args.lr}]


# noinspection PyTypeChecker
class DeeplabVGG13(nn.Module):
    def __init__(self, num_classes, restore_from, pretrained, clas):
        super(DeeplabVGG13, self).__init__()
        vgg = models.vgg13()
        if pretrained:
            vgg.load_state_dict(torch.load(restore_from))

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        features = nn.Sequential(*(features[i] for i in range(24) if i != 19))

        #for i in [17,19,21]:
        for i in [19,21]:
            features[i].dilation = (2,2)
            features[i].padding = (2,2)

        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)

        self.features = nn.Sequential(*([features[i] for i in range(len(features))] + [ fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))
        self.classifier = clas(1024, num_classes)


    def forward(self, x):
        input_size = x.size()[2:]
        feats = self.features(x)
        x = self.classifier(feats)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return x, feats


    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = [
            self.features
        ]

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = [
            self.classifier.parameters()
        ]

        for j in range(len(b)):
            for i in b[j]:
                yield i


    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.lr},
                {'params': self.get_10x_lr_params(), 'lr': args.lr}]

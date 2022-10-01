import torch
from models.resnet import DeeplabResnet, Bottleneck
from models.vgg import DeeplabVGG16, DeeplabVGG13
from models.deeplab import DeepLabV2Classifier, DeepLabV3Classifier, MSIWDeepLabV2Classifier
from models.fcn import FCNClassifier
from models.pspnet import PSPNetClassifier

def SegmentationModel(num_classes, backbone, classifier, pretrained=True):
    if classifier.lower() == 'DeepLabV2'.lower():
        clas = DeepLabV2Classifier
    elif classifier.lower() == 'DeepLabV2MSIW'.lower():
        clas = MSIWDeepLabV2Classifier
    elif classifier.lower() == 'DeepLabV3'.lower():
        clas = DeepLabV3Classifier
    elif classifier.lower() == 'FCN'.lower():
        clas = FCNClassifier
    elif classifier.lower() == 'PSPNet'.lower():
        clas = PSPNetClassifier
    else:
        ValueError("Unrecognized Classifier:"+classifier)

    if backbone.lower() == 'ResNet101'.lower():
        model = DeeplabResnet(Bottleneck, [3, 4, 23, 3], num_classes, clas)
        if pretrained:
            restore_from = './models/backbone_checkpoints/resnet101-5d3b4d8f.pth'
            saved_state_dict = torch.load(restore_from)

            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5' and not i_parts[0] == 'fc':
                    new_params[i] = saved_state_dict[i]
            model.load_state_dict(new_params)

    elif backbone.lower() == 'ResNet50'.lower():
        model = DeeplabResnet(Bottleneck, [3, 4, 6, 3], num_classes, clas)
        if pretrained:
            restore_from = './models/backbone_checkpoints/resnet50-19c8e357.pth'
            saved_state_dict = torch.load(restore_from)

            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5' and not i_parts[0] == 'fc':
                    new_params[i] = saved_state_dict[i]
            model.load_state_dict(new_params)

    elif backbone.lower() == 'VGG16'.lower():
        restore_from = './models/backbone_checkpoints/vgg16-397923af.pth'
        model = DeeplabVGG16(num_classes, restore_from, pretrained, clas)

    elif backbone.lower() == 'VGG13'.lower():
        restore_from = './models/backbone_checkpoints/vgg13-c768596a.pth'
        model = DeeplabVGG13(num_classes, restore_from, pretrained, clas)

    else:
        raise ValueError("Unrecognized Backbone:"+backbone)

    # add backbone to model for later use
    model.backbone_type = backbone
    model.classifier_type = classifier
    model.feature_channels =  2048 if backbone.lower() == 'ResNet101'.lower() else 1024
    model.parameters_dict = [{'params': model.get_1x_lr_params_NOscale(), 'lr': 1},
                             {'params': model.get_10x_lr_params(), 'lr': 10 * 1}]

    return model

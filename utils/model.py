from models.model import SegmentationModel
import numpy as np


def initialize_from_model(old_model, class_ids_map, use_c2f_init=True):
    num_classes = sum([len(ids) for ids in class_ids_map])
    new_model = SegmentationModel(num_classes, old_model.backbone_type, old_model.classifier_type, False).to('cuda')

    old_state = old_model.state_dict()
    state_dict = new_model.state_dict()
    
    if old_model.classifier_type.lower() == 'DeepLabV2'.lower():
        bias_scale = 4
        layer_filter = 'classifier'
    elif old_model.classifier_type.lower() == 'DeepLabV2MSIW'.lower():
        bias_scale = 2
        layer_filter = 'classifier'
    elif old_model.classifier_type.lower() == 'DeepLabV3'.lower():
        bias_scale = 1
        layer_filter = 'classifier.red'
    elif classifier.lower() == 'FCN'.lower():
        bias_scale = 1
        layer_filter = 'classifier'
    elif classifier.lower() == 'PSPNet'.lower():
        bias_scale = 1
        layer_filter = 'classifier'
    else:
        ValueError("Unrecognized Classifier:"+classifier)

    for l in old_state:
        if layer_filter not in l:
            state_dict[l] = old_state[l]
        else:
            if len(state_dict[l].shape) > 0:
                for i, ids in enumerate(class_ids_map):
                    if "bias" in l:
                        if use_c2f_init:
                            state_dict[l][ids] = old_state[l][i] - np.log(len(ids))/bias_scale # there are 4 convs in parallel
                        else:
                            state_dict[l][ids] = old_state[l][i]
                    else:
                        state_dict[l][ids] = old_state[l][i]
            else:
                state_dict[l] = old_state[l]

    new_model.load_state_dict(state_dict)
    new_model.train()
    return new_model

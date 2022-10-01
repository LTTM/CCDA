import torch
import random
import numpy as np


def memory_check(log_string):
    torch.cuda.synchronize()
    print(log_string)
    print(' peak:', '{:.3f}'.format(torch.cuda.max_memory_allocated() / 1024 ** 3), 'GB')
    print(' current', '{:.3f}'.format(torch.cuda.memory_allocated() / 1024 ** 3), 'GB')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError("Unrecognized boolean: "+v)

def str2str_none_num(v,t=float):
    if v.lower() in ('none',):
        return None
    else:
        try:
            return t(v)
        except ValueError:
            return v

def str2intlist(v):
    l = v.split(',')
    l = [str2str_none_num(el,t=int) for el in l]
    return l if len(l)>1 else l[0]

def str2floatlist(v):
    l = v.split(',')
    l = [str2str_none_num(el,t=float) for el in l]
    return l if len(l)>1 else l[0]

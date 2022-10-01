import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class MSIW(nn.Module):
    def __init__(self, ratio=.2):
        super(MSIW, self).__init__()
        self.iw = ratio
        
    def forward(self, nw_out):
        # extract dimensions
        N, C, H, W = nw_out.shape
        Np = N*H*W # total number of pixels Nx1xHxW == hist.sum()
        # compute probabilities and predicted segmentation map
        prob = torch.softmax(nw_out, dim=1)
        pred = torch.argmax(prob.detach(), dim=1, keepdim=True) # <- argmax, shape N x 1 x H x W
        # compute the predicted class frequencies
        hist = torch.histc(pred, bins=C, min=0, max=C-1) # 1-dimensional vector of length C
        # compute class weights array
        den = torch.clamp(torch.pow(hist, self.iw)*np.power(Np, 1-self.iw), min=1.)[pred] # <- cast to Nx1xHxW
        # compute the loss
        return -torch.sum(torch.pow(prob, 2)/den)/(N*C)


class MSIWC2F(nn.Module):
    def __init__(self, ratio=.2, ids_mapping=None):
        super(MSIWC2F, self).__init__()
        self.iw = ratio
        self.ids_mapping = ids_mapping
        
    def forward(self, nw_out):
        # compute probabilities and predicted segmentation map
        prob = torch.softmax(nw_out, dim=1)

        # align distributions plane-wise using the mappings
        planes = []
        for ids in self.ids_mapping:
            if len(ids)==1:
                planes.append(prob[:,ids,...])
            else:
                planes.append(torch.sum(nw_out[:,ids,...], dim=1, keepdim=True))

        # concatenate the logs
        out = torch.cat(planes, dim=1)

        # extract dimensions
        N, C, H, W = out.shape
        Np = N*H*W # total number of pixels Nx1xHxW == hist.sum()

        pred = torch.argmax(out.detach(), dim=1, keepdim=True) # <- argmax, shape N x 1 x H x W
        # compute the predicted class frequencies
        hist = torch.histc(pred, bins=C, min=0, max=C-1) # 1-dimensional vector of length C
        # compute class weights array
        den = torch.clamp(torch.pow(hist, self.iw)*np.power(Np, 1-self.iw), min=1.)[pred] # <- cast to Nx1xHxW
        # compute the loss
        return -torch.sum(torch.pow(out, 2)/den)/(N*C)

class c2f_kdc(nn.Module):
    def __init__(self, ids_mapping=None, same_kd_lambda=True):
        super(c2f_kdc, self).__init__()
        self.ids_mapping = ids_mapping
        self.same_kd_lambda = same_kd_lambda

    def forward(self, nw_out, nw_out_old):
        # compute the target distribution
        labels = torch.softmax(nw_out_old, dim=1)
        # log(softmax) -> denominator = difference
        den = torch.logsumexp(nw_out, dim=1, keepdim=True)
        # align distributions plane-wise using the mappings
        planes = []
        if self.same_kd_lambda == 1:
            for ids in self.ids_mapping:
                if len(ids)==1:
                    planes.append(nw_out[:,ids,...] - den)
                else:
                    planes.append(torch.logsumexp(nw_out[:,ids,...], dim=1, keepdim=True) - den)
            # concatenate the logs
            out = torch.cat(planes, dim=1)
            # compute point-wise cross-entropy
            loss = (labels*out).sum(dim=1)
            # compute the loss
            return -loss.mean()
        else:
            id_macro_ff = []
            id_macro_fc = []
            planes_ff = []
            planes_fc = []
            for id_m, ids in enumerate(self.ids_mapping):
                if len(ids)==1:
                    id_macro_ff.append(id_m)
                    planes_ff.append(nw_out[:,ids,...] - den)
                else:
                    id_macro_fc.append(id_m)
                    planes_fc.append(torch.logsumexp(nw_out[:,ids,...], dim=1, keepdim=True) - den)
            # concatenate the logs
            if planes_ff:
                out_f = torch.cat(planes_ff, dim=1)
                loss_f = (labels[:,id_macro_ff,...]*out_f).sum(dim=1) 

            if planes_fc:
                out_c = torch.cat(planes_fc, dim=1)
                loss_c = (labels[:,id_macro_fc,...]*out_c).sum(dim=1)

            if planes_ff and planes_fc:
                return -loss_f.mean(), -loss_c.mean()
            elif planes_ff:
                return -loss_f.mean(),  torch.tensor([0.], requires_grad=True, device=nw_out.device)
            elif planes_fc:
                return torch.tensor([0.], requires_grad=True, device=nw_out.device), -loss_c.mean()

class c2f_distance_kd(nn.Module):
    def __init__(self, ids_mapping=None, use_logits=False, metric='L2'):
        super(c2f_distance_kd, self).__init__()
        self.ids_mapping = ids_mapping
        self.use_logits = use_logits
        
        if metric == 'L2':
            self.metric = nn.MSELoss()
        elif metric == 'L1':
            self.metric = nn.L1Loss()
        else:
            raise ValueError("Unrecognized alignment loss, must be ['L1', 'L2']")

    def forward(self, nw_out, nw_out_old):
        if self.use_logits:
            nw_out = torch.softmax(nw_out, dim=1)
            nw_out_old = torch.softmax(nw_out_old, dim=1)
            
        # extract the appropriate planes
        planes = []
        for ids in self.ids_mapping:
            if len(ids)==1:
                planes.append(nw_out[:,ids,...])
            else:
                planes.append(torch.sum(nw_out[:,ids,...], dim=1, keepdim=True))
        # concatenate the planes
        out = torch.cat(planes, dim=1)
        
        return self.metric(out, nw_out_old)



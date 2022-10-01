import torch

class Metrics:
    def __init__(self, name_classes, device='cuda'):
        self.name_classes = name_classes
        self.num_classes = len(name_classes)
        self.device = device
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long, device=device)
        
    def __genterate_cm__(self, pred, gt):                                                       #       preds                    
        mask = (gt >= 0) & (gt < self.num_classes)                                              #     +------- 
        combinations = self.num_classes*gt[mask] + pred[mask] # 0 <= comb <= num_classes^2-1    #   l | . . . 
        cm_entries = torch.bincount(combinations, minlength=self.num_classes**2)                #   b | . . .
        return cm_entries.reshape(self.num_classes, self.num_classes)                           #   s | . . . 
        
    def add_sample(self, pred, gt):
        assert pred.shape == gt.shape, "Prediction and Ground Truth must have the same shape"
        self.confusion_matrix += self.__genterate_cm__(pred, gt) # labels along rows, predictions along columns
        
    def sample_percent_mIoU(self, pred, gt):
        assert pred.shape == gt.shape, "Prediction and Ground Truth must have the same shape"
        cm = self.__genterate_cm__(pred, gt) # labels along rows, predictions along columns
        return 100*self.nanmean(torch.diagonal(cm)/(cm.sum(dim=1)+cm.sum(dim=0)-torch.diagonal(cm)))
        
    def PA(self):
        # Pixel Accuracy (Recall) = TP/(TP+FN)
        return torch.diagonal(self.confusion_matrix)/self.confusion_matrix.sum(dim=1)
        
    def PP(self):
        # Pixel Precision = TP/(TP+FP)
        return torch.diagonal(self.confusion_matrix)/self.confusion_matrix.sum(dim=0)
        
    def IoU(self):
        # Intersection over Union = TP/(TP+FP+FN)
        return torch.diagonal(self.confusion_matrix)/(self.confusion_matrix.sum(dim=1)+self.confusion_matrix.sum(dim=0)-torch.diagonal(self.confusion_matrix))

    def percent_mIoU(self):
        return 100*self.nanmean(self.IoU())

    @staticmethod
    def nanmean(tensor):
        m = torch.isnan(tensor)
        return torch.mean(tensor[~m])
    
    def __str__(self):
        out = "="*39+'\n'
        out += "  Class\t\t PA %\t PP %\t IoU%\n"
        out += "-"*39+'\n'
        pa, pp, iou = self.PA(), self.PP(), self.IoU() 
        for i, n in enumerate(self.name_classes):
            if len(n)>=6:
                out += "  %s\t %.1f\t %.1f\t %.1f\n"%(n, 100*pa[i], 100*pp[i], 100*iou[i])
            else:
                out += "  %s\t\t %.1f\t %.1f\t %.1f\n"%(n, 100*pa[i], 100*pp[i], 100*iou[i])
        out += "-"*39+'\n'
        out += "  Average\t %.1f\t %.1f\t %.1f\n"%(100*self.nanmean(pa), 100*self.nanmean(pp), 100*self.nanmean(iou))
        out += "="*39+'\n'
        return out
            

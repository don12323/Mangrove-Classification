import torch
import torch.nn as nn

class MeanIoU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def IoU_coeff(self,y_pred, y_true, eps=0.0001): # both y_pred and y_true must be binary masks #TODO Try with and without one-hot enc
        
        y_true_f = y_true.flatten()  # flatten (N,512,512) - > (N*512*512,)
        y_pred_f = y_pred.flatten()
        intersection = torch.sum(y_true_f * y_pred_f)
        total = torch.sum(y_true_f + y_pred_f)  # total pixels with 1
        union = total-intersection   # substract intersection as intersection is counted twice in total
        return (intersection + eps)/(union + eps)

    def Mean_IoU(self, y_pred, y_true, numLabels=3, eps=0.0001):
        
        IoU_Score=0
        for i in range(numLabels):
            IoU_Score += self.IoU_coeff(y_true[:,i,:,:], y_pred[:,i,:,:], eps = 1)
        return IoU_Score/numLabels

    def get_class_iou(self, y_pred, y_true, numLabels=3, eps=1e-6):
        """
        Calculate IoU for each individual class

        Returns:
            dict: Dictionary with class names as keys and IoU scores as values
        """
        class_iou = []
        for i in range(numLabels):
            iou_score = self.IoU_coeff(y_true[:, i, :, :], y_pred[:, i, :, :], eps=eps)
            class_iou.append(iou_score.item())
        return class_iou

    def forward(self, y_pred, y_true):
        return self.Mean_IoU(y_pred, y_true)


class MultiDiceLoss(nn.Module):
    def __init__(self):                #TODO add param to initialise numclasses 
        super().__init__()
    
    def dice_coef(self, y_pred, y_true, eps=0.0001):

        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = torch.sum(y_true_f * y_pred_f)

        return (2. * intersection + eps) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + eps)

    def dice_coef_multiclass(self, y_pred, y_true, numLabels=3, eps=0.0001):    
        dice=0

        for i in range(numLabels):
            dice += self.dice_coef(y_true[:,i,:,:], y_pred[:,i,:,:], eps = 0.0001)

        return 1 - dice/numLabels

    def forward(self, y_pred, y_true):
        return self.dice_coef_multiclass(y_pred, y_true)


class CategoricalCrossEntropyLoss(nn.Module): #TODO
        pass





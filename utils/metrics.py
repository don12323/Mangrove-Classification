import torch
import torch.nn as nn

class MeanIoU(nn.Module):
    def __init__(self, numLabels):
        super().__init__()
        self.numLabels = numLabels
    def IoU_coeff(self,y_pred, y_true, eps=0.0001): # both y_pred and y_true must be binary masks #TODO Try with and without one-hot enc
        
        y_true_f = y_true.flatten()  # flatten (N,512,512) - > (N*512*512,)
        y_pred_f = y_pred.flatten()
        intersection = torch.sum(y_true_f * y_pred_f)
        total = torch.sum(y_true_f + y_pred_f)  # total pixels with 1
        union = total-intersection   # substract intersection as intersection is counted twice in total
        return (intersection + eps)/(union + eps)

    def Mean_IoU(self, y_pred, y_true, eps=0.0001):
        
        IoU_Score=0
        for i in range(self.numLabels):
            IoU_Score += self.IoU_coeff(y_true[:,i,:,:], y_pred[:,i,:,:], eps = 1)
        return IoU_Score/self.numLabels

    def get_class_iou(self, y_pred, y_true, eps=1e-6):
        """
        Calculate IoU for each individual class

        Returns:
            dict: Dictionary with class names as keys and IoU scores as values
        """
        class_iou = []
        for i in range(self.numLabels):
            iou_score = self.IoU_coeff(y_true[:, i, :, :], y_pred[:, i, :, :], eps=eps)
            class_iou.append(iou_score.item())
        return class_iou

    def forward(self, y_pred, y_true):
        return self.Mean_IoU(y_pred, y_true)


class MultiDiceLoss(nn.Module):
    def __init__(self, numLabels):                #TODO add param to initialise numclasses 
        super().__init__()
        self.numLabels = numLabels
    
    def dice_coef(self, y_pred, y_true, eps=0.0001):

        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = torch.sum(y_true_f * y_pred_f)

        return (2. * intersection + eps) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + eps)

    def dice_coef_multiclass(self, y_pred, y_true, numLabels=2, eps=0.0001):    
        dice=0

        for i in range(self.numLabels):
            dice += self.dice_coef(y_true[:,i,:,:], y_pred[:,i,:,:], eps = 0.0001)

        return 1 - dice/self.numLabels

    def forward(self, y_pred, y_true):
        return self.dice_coef_multiclass(y_pred, y_true)


class CategoricalCrossEntropyLoss(nn.Module): #TODO
        pass

"""
class MultiDiceLoss(DiceLoss):
    def __init__(self, num_classes, smooth=1e-6):
        super(MultiDiceLoss, self).__init__(smooth)
        self.num_classes = num_classes

    def forward(self, preds, targets):
        # preds: (B, C, H, W), targets: (B, H, W)
        preds = F.softmax(preds, dim=1)
        total_loss = 0.0

        # One-hot encode target
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

        for c in range(self.num_classes):
            total_loss += super().forward(preds[:, c], targets_onehot[:, c])

        return total_loss / self.num_classes
"""



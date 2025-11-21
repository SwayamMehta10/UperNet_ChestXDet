"""
Loss functions for semantic segmentation.
Includes Dice loss and combined Dice + Cross-Entropy loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary/multi-class segmentation.
    Directly optimizes the Dice coefficient.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] - model predictions (logits)
            target: [B, H, W] - ground truth labels
        Returns:
            Dice loss (1 - Dice coefficient)
        """
        # Get probabilities for disease class (class 1)
        pred_prob = F.softmax(pred, dim=1)[:, 1]  # [B, H, W]
        target_binary = (target == 1).float()      # [B, H, W]
        
        # Flatten
        pred_flat = pred_prob.contiguous().view(-1)
        target_flat = target_binary.contiguous().view(-1)
        
        # Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice  # Return loss (1 - Dice)


class CombinedLoss(nn.Module):
    """
    Combined Cross-Entropy + Dice Loss.
    Balances pixel-wise accuracy (CE) with region overlap (Dice).
    """
    def __init__(self, weight_ce=0.5, weight_dice=0.5, ignore_index=-1):
        super(CombinedLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce_loss = nn.NLLLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] - model predictions (log probabilities)
            target: [B, H, W] - ground truth labels
        Returns:
            Combined loss
        """
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        return self.weight_ce * ce + self.weight_dice * dice

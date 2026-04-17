import torch
import torch.nn as nn

class FRLoss(nn.Module):
    def __init__(self):
        """
        ArcFace loss function, typically cross-entropy loss.
        """
        super(FRLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        """
        Forward pass for the ArcFace loss.
        
        Args:
            logits (torch.Tensor): Output logits from the ArcFace head.
            labels (torch.Tensor): Ground truth class labels.
        
        Returns:
            torch.Tensor: Computed loss.
        """
        return self.loss_fn(logits, labels)

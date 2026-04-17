import torch
import torch.nn as nn

class AtributeLoss(nn.Module):
    def __init__(self, attribute_loss_weight=1.5, id_loss_weight=14):
        """
        Multitask loss combining attribute prediction loss and ID classification loss.

        Args:
            attribute_loss_weight (float): Weight for the attribute loss component.
            id_loss_weight (float): Weight for the ID classification loss component.
        """
        super(AtributeLoss, self).__init__()
        self.attribute_loss_fn = nn.BCEWithLogitsLoss()
        self.id_loss_fn = nn.CrossEntropyLoss()
        self.attribute_loss_weight = attribute_loss_weight
        self.id_loss_weight = id_loss_weight

    def forward(self, attr_preds=None, attr_labels=None, id_preds=None, id_labels=None):
        """
        Compute the total loss for both tasks, skipping attribute loss if ground truth attributes are None.

        Args:
            attr_preds (torch.Tensor): Predicted logits for attributes (batch_size, num_attributes), or None.
            attr_labels (torch.Tensor): Ground truth for attributes (batch_size, num_attributes), or None.
            id_preds (torch.Tensor): Predicted logits for IDs (batch_size, num_ids).
            id_labels (torch.Tensor): Ground truth for IDs (batch_size).

        Returns:
            torch.Tensor: Combined loss.
            torch.Tensor: Attribute loss (if applicable, otherwise 0).
            torch.Tensor: ID loss.
        """
        attr_loss = 0  # Default to 0 if no attribute task is present
        if attr_preds is not None and attr_labels is not None:
            attr_loss = self.attribute_loss_fn(attr_preds, attr_labels)

        id_loss = self.id_loss_fn(id_preds, id_labels) if id_preds is not None and id_labels is not None else 0

        total_loss = self.attribute_loss_weight * attr_loss + self.id_loss_weight * id_loss
        return total_loss, attr_loss, id_loss

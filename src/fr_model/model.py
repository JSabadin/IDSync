from src.blocks.backbones import IResNet
from src.blocks.heads import AdaFace, ArcFace
import torch
from torch import Tensor
from typing import Optional

class FRModel(torch.nn.Module):
    def __init__(self, embedding_size, num_classes, backbone="ir_se_50", head="adaface"):
        """
        Initializes the FR model consisting of a ResNet backbone and AdaFace head.
        Args:
            embedding_size (int): Size of the embedding output from the backbone.
            num_classes (int): Number of classes (identities) in the dataset.
            model_type (str): ResNet backbone type ('resnet50' or 'resnet100').
        """
        super(FRModel, self).__init__()
        self.backbone = IResNet(model_name=backbone)
        if head == "adaface":
            self.head = AdaFace(embedding_size=embedding_size, classnum=num_classes)
        elif head == "arcface":
            self.head = ArcFace(embedding_size=embedding_size, classnum=num_classes)

    def forward(self, images: Tensor, labels: Optional[int] = None) -> Tensor:
        embeddings, norm = self.backbone(images)
        if self.training and labels is not None:
            logits = self.head(embeddings, labels, norm)
            return logits
        else:
            return embeddings

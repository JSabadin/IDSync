import torch
import torch.nn as nn
from src.blocks.backbones import IResNet

class ImprovedAtributeModel(nn.Module):
    def __init__(self, embedding_size=512, num_attributes=None, num_ids=None, backbone="ir_se_50"):
        super(ImprovedAtributeModel, self).__init__()
        self.backbone = IResNet(model_name=backbone)
        
        # Task-specific projection layers
        self.attr_projection = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.BatchNorm1d(embedding_size // 2),
            nn.ReLU(),
            nn.Dropout(0.5)  # Dropout for regularization
        )
        self.id_projection = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.BatchNorm1d(embedding_size // 2),
            nn.ReLU(),
            nn.Dropout(0.5)  # Dropout for regularization
        )
        
        # ID Head with deeper regularization
        self.id_head = nn.Sequential(
            nn.Linear(embedding_size // 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_ids)
        )
        
        # Attribute Head
        self.attribute_head = (
            nn.Sequential(
                nn.Linear(embedding_size // 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_attributes)
            ) if num_attributes is not None else None
        )

        # Initialize weights
        self.apply(self._initialize_weights)

    def forward(self, x, task="all"):
        embeddings, _ = self.backbone(x)  # Shared backbone output
        
        attr_embeddings = self.attr_projection(embeddings)
        id_embeddings = self.id_projection(embeddings)

        if task == "attributes":
            if self.attribute_head is None:
                raise ValueError("Attribute head is not defined.")
            return self.attribute_head(attr_embeddings)
        elif task == "ids":
            return self.id_head(id_embeddings)
        elif task == "all":
            if self.attribute_head is None:
                raise ValueError("Attribute head is not defined for 'all' task.")
            return self.attribute_head(attr_embeddings), self.id_head(id_embeddings)
        else:
            raise ValueError("Invalid task type. Choose 'attributes', 'ids', or 'all'.")

    @staticmethod
    def _initialize_weights(layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.BatchNorm1d):
            if layer.weight is not None:
                nn.init.constant_(layer.weight, 1)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


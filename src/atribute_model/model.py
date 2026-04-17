import torch
import torch.nn as nn
from src.blocks.backbones import IResNet

class AtributeModel(nn.Module):
    def __init__(self, embedding_size=512, num_attributes=None, num_ids=None, backbone="ir_se_50"):
        super(AtributeModel, self).__init__()
        self.backbone = IResNet(model_name=backbone)
        
        # Task-specific projection layers
        self.attr_projection = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.ReLU()
        )
        self.id_projection = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.ReLU()
        )
        
        # Heads
        self.id_head = nn.Sequential(
            nn.Linear(embedding_size // 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_ids)
        )
        self.attribute_head = (
            nn.Linear(embedding_size // 2, num_attributes) if num_attributes is not None else None
        )

    def forward(self, x, task="all"):
        embeddings, _ = self.backbone(x)  # Shared backbone output
        
        attr_embeddings = self.attr_projection(embeddings)
        id_embeddings = self.id_projection(embeddings)

        if task == "attributes":
            return self.attribute_head(attr_embeddings)
        elif task == "ids":
            return self.id_head(id_embeddings)
        elif task == "all":
            return self.attribute_head(attr_embeddings), self.id_head(id_embeddings)
        else:
            raise ValueError("Invalid task type. Choose 'attributes', 'ids', or 'all'.")

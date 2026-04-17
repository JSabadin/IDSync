from src.atribute_model.model import AtributeModel
from src.common.utils import load_weights
from typing import Optional

def get_attribute_model(
    embedding_size: int = 512,
    num_attributes: Optional[int] = None,
    num_ids: int = 10572,
    backbone: str = "ir_18",
    weights_path: Optional[str] = None,
) -> AtributeModel:
    """
    Returns the Attribute model with the specified settings.
    Args:
        num_attributes (int): Number of attribute labels.
        num_ids (int): Number of identity labels.
        embedding_size (int): Size of the embedding output from the backbone.
        backbone (str): ResNet backbone type ('ir_50', 'ir_se_50', etc.).
        weights_path (str): Path to the model weights checkpoint file.
    """
    model = AtributeModel(embedding_size=embedding_size, num_attributes=num_attributes, num_ids=num_ids, backbone=backbone)
    if weights_path is not None:
        model = load_weights(model, weights_path)
        print(f"Loaded model weights from {weights_path}")
    return model
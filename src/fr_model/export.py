import torch
from typing import Dict
from src.fr_model.model import FRModel
from src.common import load_weights

def export_model_to_onnx(fr_config: Dict, input_size=(1, 3, 112, 112)):
    """
    Export the FRModel to ONNX format.

    Args:
        fr_config (Dict): Configuration dictionary for the model.
        output_path (str): Path to save the exported ONNX file.
        input_size (tuple): The size of the input tensor (default is (1, 3, 112, 112)).
    """
    model = FRModel(
        embedding_size=fr_config['model']['embedding_size'],
        num_classes=fr_config['model']['num_classes'],
        backbone=fr_config['model']['backbone'],
        head=fr_config['model']['head']
    )
    
    weights_path = fr_config['trainer']['weights_path']
    model = load_weights(model, weights_path)

    model.eval()

    dummy_input = torch.randn(input_size)

    torch.onnx.export(
        model,
        dummy_input,
        fr_config['trainer']['export_path'],
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},  # Allow for dynamic batch size
            'output': {0: 'batch_size'}
        }
    )

config = {
    'model': {
        'embedding_size': 512,
        'num_attributes': None,
        'num_ids': 65209,
        'backbone': 'ir_se_50',
        'attribute_loss_weight': 0,
        'id_loss_weight': 30,
    },
    'optimizer': {
        'type': 'SGD',
        'params': {
            'lr': 0.04,
            'momentum': 0.9,
            'weight_decay': 5e-5
        }
    },
    'scheduler': {
        'type': 'CosineAnnealingLR',
        'params': {
            'T_max': 100,
            'eta_min': 3e-5
        }
    },
    'trainer': {
        'dataset': {
            'type': 'webface21',
            'image_dir': '../data/Arc2Face_448x448',
            'mapping_file': './assets/webface21_mapping.json',
        },
        'weights_path': None,
        'num_workers': 8,
        'epochs': 100,
        'batch_size': 256,
        'accumulate_grad_batches': 4,
        'save_path': './experiments/atribute_model_webface_ir_se_50_webface2M/weights',  # Directory to save weights
        'model_name': 'atribute_model_webface_ir_se_50_webface2M', # Model name for saving
        'val_interval': 1,
        'export_path': './experiments/atribute_model_webface_ir_se_50_webface2M/exported_model.pth',  # Path to export final model
        'metrics_path': './experiments/atribute_model_webface_ir_se_50_webface2M/metrics.json',  # Path to save metrics
    }
}

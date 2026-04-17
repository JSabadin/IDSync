# Effective BS = 512, LR = 0.1, Epochs = 40, scheduler: milestones = [24, 30, 36], gamma = 0.1, optimizer: SGD, ir_se_50, adaface, augment = True

config = {
    'model': {
        'embedding_size': 512,
        'num_classes': 10575,
        'backbone': 'ir_se_50',
        'head': 'adaface',
    },
    'optimizer': {
        'type': 'SGD',
        'params': {
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 0.0005
        }
    },
    'scheduler': {
        'type': 'MultiStepLR',
        'params': {
            'milestones': [24, 30, 36],
            'gamma': 0.1
        }
    },
    'trainer': {
        'train_dataset': {
            'path': <dataset/path/here>,
            'dataset_name': 'webface',
            'augment': True,
        },
        'num_workers': 8,
        'epochs': 40,
        'batch_size': 128,
        'accumulate_grad_batches': 2,
        'save_path': './experiments/fr_model/weights',
        'model_name': 'resnet100_adaface_webface_augs',  # Name for saving checkpoints
        'val_interval': 1,
        'weights_path': None,  # Path to weights file to resume training or evaluate
        'export_path': None,        
        'eval_datasets': {
            'agedb': {
                "path": '../data/AgeDB/', 
                "num_pairs": 6000,
            },
            'cfp-fp': {
                "path": '../data/cfp-fp/', 
                "num_pairs": 6000,
            },
            'lfw': {
                "path": '../data/lfw/', 
                "num_pairs": 6000,
            },
            'cplfw': {
                "path": '../data/cplfw/', 
                "num_pairs": 6000,
            },
            'calfw': {
                "path": '../data/calfw/', 
                "num_pairs": 6000,
            },
        }
    }
}
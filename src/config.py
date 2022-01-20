import torch.nn as nn


base_net = [
    {
        'type': 'transform',
        'matrix_size': 3,
        'blocks': {
            'lnet': [
                {'in_features': 3, 'out_features': 64, 'normalize': True, 'activation': nn.ReLU()},
                {'in_features': 64, 'out_features': 128, 'normalize': True, 'activation': nn.ReLU()},
                {'in_features': 128, 'out_features': 1024, 'normalize': True, 'activation': nn.ReLU()},
            ],
            'gnet': [
                {'in_features': 1024, 'out_features': 512, 'normalize': True, 'activation': nn.ReLU()},
                {'in_features': 512, 'out_features': 256, 'normalize': True, 'activation': nn.ReLU()},
            ],
        },
    },
    {
        'type': 'extraction',
        'blocks': [
            {'in_features': 3, 'out_features': 64, 'normalize': True, 'activation': nn.ReLU()},
            {'in_features': 64, 'out_features': 64, 'normalize': True, 'activation': nn.ReLU()},
        ]
    },
    {
        'type': 'transform',
        'matrix_size': 64,
        'blocks': {
            'lnet': [
                {'in_features': 64, 'out_features': 128, 'normalize': True, 'activation': nn.ReLU()},
                {'in_features': 128, 'out_features': 256, 'normalize': True, 'activation': nn.ReLU()},
                {'in_features': 256, 'out_features': 2048, 'normalize': True, 'activation': nn.ReLU()},
            ],
            'gnet': [
                {'in_features': 2048, 'out_features': 1024, 'normalize': True, 'activation': nn.ReLU()},
                {'in_features': 1024, 'out_features': 512, 'normalize': True, 'activation': nn.ReLU()},
            ]
        }
    },
    {
        'type': 'extraction',
        'blocks': [
            {'in_features': 64, 'out_features': 128, 'normalize': True, 'activation': nn.ReLU()},
            {'in_features': 128, 'out_features': 1024, 'normalize': True, 'activation': nn.ReLU()},
        ]
    },
    {
        'type': 'globalization'
    },
    {
        'type': 'classification',
        'blocks': [
            {'in_features': 1024, 'out_features': 512, 'normalize': True, 'activation': nn.ReLU()},
            {'in_features': 512, 'out_features': 256, 'normalize': True, 'activation': nn.ReLU()},
            {'in_features': 256, 'out_features': 40, 'normalize': False, 'activation': None}
        ]
    }
]


dataset = {
    'src_dir': '/data/modelnet40',
    'train_dir': '/data/modelnet40-train',
    'test_dir': '/data/modelnet40-test',
    'n_samples': 100,
    'n_points': 1024
}


config = {
    'lr': 0.001,
    'alpha': 0.001,
    'batch_size': 128
}

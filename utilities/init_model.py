import torch.nn as nn

def weights_init(model, activation):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nonlinearity = 'leaky_relu' if activation == 'leaky' else 'relu'
            nn.init.kaiming_normal_(
                m.weight, 
                mode='fan_out', 
                nonlinearity=nonlinearity
            )
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            print(f'{m.__class__.__name__} module initialized ({nonlinearity})')

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
            print(f'{m.__class__.__name__} module initialized')

    print(f'{model.__class__.__name__} initialization successful.')
import torch.nn as nn
import torchaudio
from torchvision.transforms import Compose
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt 

import yaml
import os
import sys
sys.path.append(os.curdir)
from utilities.init_model import weights_init



class Encoder(nn.Module):

    def __init__(self, in_dims, kernel_size, activation):
        super().__init__()

        self.in_layer = nn.Conv2d(in_channels=in_dims, out_channels=8, kernel_size=7, stride=1, padding=3)
        self.conv_1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=kernel_size, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.bn4 = nn.BatchNorm2d(num_features=64)

        self.dropout = nn.Dropout2d(0.2)

        assert activation in ('relu', 'leaky'), "Invalid activation. Use 'relu' or 'leaky'"
        self.activation = (
            nn.ReLU() if activation == 'relu' 
            else nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = self.dropout(self.activation(self.bn1(self.in_layer(x))))
        x = self.dropout(self.activation(self.bn2(self.conv_1(x))))
        x = self.dropout(self.activation(self.bn3(self.conv_2(x))))
        x = self.dropout(self.activation(self.bn4(self.conv_3(x))))
        return x
    

class Decoder(nn.Module):

    def __init__(self, out_dims, kernel_size, activation):
        super().__init__()
        
        self.conv_1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=kernel_size, stride=2, padding=1, output_padding=(1, 0))
        self.conv_2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=kernel_size, stride=2, padding=1, output_padding=(1, 0))
        self.conv_3 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=kernel_size, stride=2, padding=1, output_padding=(1, 0))
        self.out_layer = nn.Conv2d(in_channels=8, out_channels=out_dims, kernel_size=7, stride=1, padding=3)
        
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.bn3 = nn.BatchNorm2d(num_features=8)

        self.dropout = nn.Dropout2d(0.2)

        assert activation in ('relu', 'leaky'), "Invalid activation. Use 'relu' or 'leaky'"
        self.activation = (
            nn.ReLU() if activation == 'relu' 
            else nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = self.dropout(self.activation(self.bn1(self.conv_1(x))))
        x = self.dropout(self.activation(self.bn2(self.conv_2(x))))
        x = self.dropout(self.activation(self.bn3(self.conv_3(x))))
        x = torch.tanh(self.out_layer(x))
        return x


class Embedding(nn.Module):

    def __init__(self, kernel_size, activation):
        super().__init__()

        self.embedding_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=1, padding=1)
        self.embedding_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=1, padding=1)
        
        self.bn_1 = nn.BatchNorm2d(num_features=64)
        self.bn_2 = nn.BatchNorm2d(num_features=64)

        assert activation in ('relu', 'leaky'), "Invalid activation. Use 'relu' or 'leaky'"
        self.activation = (
            nn.ReLU() if activation == 'relu' 
            else nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        x = self.activation(self.bn_1(self.embedding_1(x)))
        x = self.activation(self.bn_2(self.embedding_2(x)))
        return x


class VisResMAE(nn.Module):
    
    def __init__(self, in_dims, out_dims, kernel_size, activation, alpha):
        super().__init__()
        self.alpha = alpha
        self.enc = Encoder(in_dims=in_dims, kernel_size=kernel_size, activation=activation)
        self.emb = Embedding(kernel_size=kernel_size, activation=activation)
        self.dec = Decoder(out_dims=out_dims, kernel_size=kernel_size, activation=activation)

        self.initialize_weights(activation=activation)

    def forward(self, x):
        self.in_shape = x.shape
        x = self.enc(x)
        identity = x * self.alpha
        emb = self.emb(x)
        emb += identity
        x = self.dec(emb)

        if x.shape[-2:] != self.in_shape[-2:]:
            x = F.interpolate(x, size=self.in_shape[-2:])
            print('Warning: out signal interpolated. Risk of possible artifacts in the reshaping of the tensor. Try different out padding in the transposed convolution.')
        
        return x, emb
    
    def initialize_weights(self, activation):
        weights_init(self, activation=activation)



if __name__ == '__main__':

    from utilities.mask import *
    
    masking = SpectrogramMasking(mask_ratio=0.75, patch_size=16)

    with open('/Users/inigoparra/Desktop/PROJECTS/MAE/VisMAE/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    sample_path = '/Users/inigoparra/Desktop/Datasets/speech/7059-77900-0003.flac'
    sample, sr = torchaudio.load(sample_path)
    sample = sample[:, :64000]
    print(f'Original samples: {sample.size(1)}')

    transforms = Compose([
        torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000),
        torchaudio.transforms.MelSpectrogram(**config['spectrogram_config']),
        torchaudio.transforms.AmplitudeToDB()
    ])

    original_spec = transforms(sample)
    original_spec = original_spec.unsqueeze(1)
    mask_spec, mask = masking(original_spec.clone())

    autoencoder = VisResMAE(in_dims=1, out_dims=1, kernel_size=3, activation='leaky', alpha=0.1)
    x, emb = autoencoder(mask_spec)

    loss = masked_mse_loss(x, original_spec, mask)
    print(loss.item())

    emb_transform = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1)
    emb = emb_transform(emb)

    def plot_pass(tensor_a, tensor_b, tensor_c, tensor_d):

        plt.figure(figsize=(12, 6))

        plt.subplot(4, 1, 1)
        plt.imshow(tensor_a.squeeze().detach().numpy(), cmap='magma', origin='lower')
        plt.title('Mel Spectrogram - Original')
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')

        plt.subplot(4, 1, 2)
        plt.imshow(tensor_b.squeeze().detach().numpy(), cmap='magma', origin='lower')
        plt.title('Mel Spectrogram - Masked (75%)')
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')


        plt.subplot(4, 1, 3)
        plt.imshow(tensor_c.squeeze().detach().numpy(), cmap='magma', origin='lower')
        plt.title('Mel Spectrogram - Processed (single pass)')
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')

        plt.subplot(4, 1, 4)
        plt.imshow(tensor_d.squeeze().detach().numpy(), cmap='magma', origin='lower')
        plt.title('Latent Representation')
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')

        plt.tight_layout()
        plt.show()

    plot_pass(original_spec, mask_spec, x, emb)

    
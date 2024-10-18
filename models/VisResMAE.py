import torch.nn as nn
import torchaudio
from torchvision.transforms import Compose
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt 

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

        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.bn3 = nn.BatchNorm2d(num_features=32)

        self.dropout = nn.Dropout2d(0.2)

        assert activation in ('relu', 'leaky'), "Invalid activation. Use 'relu' or 'leaky'"
        self.activation = (
            nn.ReLU() if activation == 'relu' 
            else nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):
        x = self.dropout(self.activation(self.bn1(self.in_layer(x))))
        x = self.dropout(self.activation(self.bn2(self.conv_1(x))))
        x = self.dropout(self.activation(self.bn3(self.conv_2(x))))
        return x
    

class Decoder(nn.Module):

    def __init__(self, out_dims, kernel_size, activation):
        super().__init__()

        self.conv_1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
        self.conv_2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=kernel_size, stride=2, padding=1, output_padding=(1, 0))
        self.out_layer = nn.Conv2d(in_channels=8, out_channels=out_dims, kernel_size=7, stride=1, padding=3)
        
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.bn2 = nn.BatchNorm2d(num_features=8)

        self.dropout = nn.Dropout2d(0.2)

        assert activation in ('relu', 'leaky'), "Invalid activation. Use 'relu' or 'leaky'"
        self.activation = (
            nn.ReLU() if activation == 'relu' 
            else nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):
        x = self.dropout(self.activation(self.bn1(self.conv_1(x))))
        x = self.dropout(self.activation(self.bn2(self.conv_2(x))))
        x = torch.tanh(self.out_layer(x))
        return x


class Embedding(nn.Module):

    def __init__(self, kernel_size, activation):
        super().__init__()

        self.embedding_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=1, padding=1)
        self.embedding_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=1, padding=1)
        
        self.bn_1 = nn.BatchNorm2d(num_features=32)
        self.bn_2 = nn.BatchNorm2d(num_features=32)

        assert activation in ('relu', 'leaky'), "Invalid activation. Use 'relu' or 'leaky'"
        self.activation = (
            nn.ReLU() if activation == 'relu' 
            else nn.LeakyReLU(negative_slope=0.1)
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
            print('Interpolated')
        return x, emb
    
    def initialize_weights(self, activation):
        weights_init(self, activation=activation)



if __name__ == '__main__':

    sample_path = '/Users/inigoparra/Desktop/Datasets/tone_perfect_wav_16/sun3_MV3_MP3_16.wav'
    sample, sr = torchaudio.load(sample_path)

    transforms = Compose([
        torchaudio.transforms.Resample(orig_freq=sr, new_freq=8000),
        torchaudio.transforms.MelSpectrogram(
            sample_rate=8000,
            n_fft=256,                 # Small window for better temporal precision
            hop_length=64,             # High overlap to capture temporal structure
            n_mels=64,                 # Mel bands (128 is common for speech)
            f_min=60,                  # Capture low-frequency components like voice
            f_max=4000,                # Respect Nyquist limit at 8 kHz sample rate
        )
    ])

    original = transforms(sample)

    original = original.unsqueeze(1)
    print(f'Original Shape: {original.shape}')

    autoencoder = VisResMAE(in_dims=1, out_dims=1, kernel_size=3, activation='leaky', alpha=0.1)
    x, emb = autoencoder(original)
    print(f'Predicted Shape: {x.shape}')

    emb_transform = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1)
    emb = emb_transform(emb)

    def plot_pass(tensor_a, tensor_b, tensor_c):

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(tensor_a.squeeze().detach().numpy(), cmap='magma', origin='lower')
        plt.title('Mel Spectrogram - Original')
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')

        plt.subplot(1, 3, 2)
        plt.imshow(tensor_b.squeeze().detach().numpy(), cmap='magma', origin='lower')
        plt.title('Mel Spectrogram - Processed')
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')

        plt.subplot(1, 3, 3)
        plt.imshow(tensor_c.squeeze().detach().numpy(), cmap='magma', origin='lower')
        plt.title('Latent Representation')
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')

        plt.tight_layout()
        plt.show()

    plot_pass(original, x, emb)
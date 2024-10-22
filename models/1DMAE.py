import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import einops
import math


class SinusoidalPositionalEncoding(nn.Module):
                                              
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
                                         
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class Patchify(nn.Module):

    """
    Patch audio segments using a rolling 1D convolution
    """

    def __init__(self, d_model, patch_size, n_patches, in_channels):
        super().__init__()
        self.patcher = nn.Sequential(
           nn.Conv1d(
               in_channels=in_channels,
               out_channels=d_model,
               kernel_size=patch_size,
               stride=patch_size
           ),
           nn.Flatten(2)
        )

        self.position_embeddings = nn.Parameter(torch.randn(size=(1, n_patches, d_model)), requires_grad=True)

    def forward(self, x):
        x = self.patcher(x)
        x = einops.rearrange(x, 'B C L -> B L C')
        x = self.position_embeddings + x
        return x


class Encoder(nn.Module):

    """
    Learn meaningful info about the parts that the model CAN see
    """

    def __init__(self, n_layers, d_model, n_heads, h_dims, dropout, activation, batch_first):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=h_dims,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=n_layers,
        )

    def forward(self, x):
        # Step 1: MASK PATCHES
        
        out = self.encoder(x)
        return out
    

class Decoder(nn.Module):

    """
    Learn to reconstruct using the representations of the encoder
    """

    def __init__(self, n_layers, d_model, n_heads, h_dims, dropout, activation, batch_first):
        super().__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=h_dims,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=n_layers
        )

    def forward(self, x, memory):
        out = self.decoder(x, memory)
        return out



class AudiT(nn.Module):

    def __init__(self, patch_size, n_patches, in_channels, n_layers, d_model, n_heads, h_dims, dropout, activation, batch_first, mask_ratio):
        super().__init__()

        self.patchify = Patchify(d_model, patch_size, n_patches, in_channels)
        self.encoder = Encoder(n_layers, d_model, n_heads, h_dims, dropout, activation, batch_first)
        self.decoder = Decoder(n_layers, d_model, n_heads, h_dims, dropout, activation, batch_first)

        # Add decoder prediction head
        self.decoder_pred = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mask_ratio = mask_ratio
        self.n_patches = n_patches

    def forward(self, x):
        pass


if __name__ == "__main__":

    x, _ = torchaudio.load('/Users/inigoparra/Desktop/Datasets/speech/7059-77900-0003.flac')
    x = x[:, 0:48000].unsqueeze(1)

    patcher = Patchify(512, 100, 480, 1)
    patches = patcher(x)

    def visualize_embeddings(tensor):
        plt.imshow(tensor.squeeze().detach().numpy(), cmap='magma', origin='lower')
        plt.colorbar()
        plt.xlabel('d_model')
        plt.ylabel('Patch Embeddings')
        plt.tight_layout()
        plt.show()

    # visualize_embeddings(patches[0])
import torch
import torchaudio
import torch.nn as nn
import matplotlib.pyplot as plt
import einops


class Patchify(nn.Module):

    """
    Patch audio segments using a rolling 1D convolution
    """

    def __init__(self, d_model, patch_size, n_patches, dropout, in_channels):
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

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embeddings = nn.Parameter(torch.randn(size=(1, n_patches+1, d_model)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patcher(x)
        x = einops.rearrange(x, 'B C L -> B L C')
        x = torch.cat([cls_token, x], dim=1)
        x = self.position_embeddings + x
        x = self.dropout(x)
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

    def __init__(self, patch_size, n_patches, in_channels, n_layers, d_model, n_heads, h_dims, dropout, activation, batch_first):
        super().__init__()

        self.patchify = Patchify(d_model, patch_size, n_patches, dropout, in_channels)
        self.encoder = Encoder(n_layers, d_model, n_heads, h_dims, dropout, activation, batch_first)
        self.encoder = Decoder(n_layers, d_model, n_heads, h_dims, dropout, activation, batch_first)

    def forward(self, x, mask_ratio):
        pass





if __name__ == "__main__":

    patch = Patchify(512, 50, 960, 0.2, 1)
    enc = Encoder(n_layers=6, d_model=512, n_heads=8, h_dims=2048, dropout=0.2, activation='gelu', batch_first=True)
    dec = Decoder(n_layers=6, d_model=512, n_heads=8, h_dims=2048, dropout=0.2, activation='gelu', batch_first=True)

    x, _ = torchaudio.load('/Users/inigoparra/Desktop/Datasets/speech/7059-77900-0003.flac')
    x = x[:, 0:48000].unsqueeze(1)
    patches = patch(x)
    print(f'Patchified input: {patches.shape}')

    

    # shuffled = torch.randperm()
    

    

    def visualize_embeddings(tensor):
        plt.imshow(tensor[0].squeeze().detach().numpy(), cmap='magma', origin='lower')
        plt.colorbar()
        plt.xlabel('d_model')
        plt.ylabel('Patch Embeddings')
        plt.tight_layout()
        plt.show()




import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
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

    def __init__(self, patch_size, n_patches, in_channels, n_layers, d_model, n_heads, h_dims, dropout, activation, batch_first, mask_ratio):
        super().__init__()

        self.patchify = Patchify(d_model, patch_size, n_patches, dropout, in_channels)
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
        # Step 1: Patchify the input audio
        x = self.patchify(x)
        
        # Step 2: Separate patches (excluding CLS token for encoding)
        patches = x[:, 1:, :]
        
        # Step 3: Generate random mask
        batch_size, n_patches, _ = patches.shape
        num_mask = int(self.mask_ratio * n_patches)
        rand_indices = torch.rand(batch_size, n_patches, device=x.device).argsort(dim=1)
        mask_indices = rand_indices[:, :num_mask]
        unmask_indices = rand_indices[:, num_mask:]
        
        # Step 4: Create masks
        mask = torch.zeros(batch_size, n_patches, dtype=torch.bool, device=x.device)
        mask.scatter_(1, mask_indices, True)
        
        # Step 5: Separate masked and unmasked patches
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1)
        unmasked_patches = patches[batch_indices, unmask_indices]
        
        # Step 6: Encode unmasked patches (without CLS token)
        encoded = self.encoder(unmasked_patches)
        
        # Step 7: Prepare decoder input
        mask_tokens = self.mask_token.expand(batch_size, num_mask, -1)
        
        # Prepare decoder input patches
        decoder_input_patches = torch.zeros(batch_size, n_patches, patches.size(-1), device=x.device)
        decoder_input_patches[batch_indices, unmask_indices] = encoded
        decoder_input_patches[batch_indices, mask_indices] = mask_tokens
        
        # Add position embeddings (excluding CLS token embeddings)
        decoder_input = decoder_input_patches + self.patchify.position_embeddings[:, 1:, :]

        # Step 8: Decode to reconstruct
        decoded = self.decoder(decoder_input, encoded)
        
        # Step 9: Apply prediction head
        pred = self.decoder_pred(decoded)
        
        # Step 10: Calculate reconstruction loss
        loss = F.mse_loss(pred[mask], patches[mask])

        return loss, pred, mask, patches


if __name__ == "__main__":

    x, _ = torchaudio.load('/Users/inigoparra/Desktop/Datasets/speech/7059-77900-0003.flac')
    x = x[:, 0:48000].unsqueeze(1)

    # Define model parameters
    patch_size = 400
    n_patches = 120
    in_channels = 1
    n_layers = 6
    d_model = 768
    n_heads = 8
    h_dims = 2048
    dropout = 0.1
    activation = 'gelu'
    batch_first = True
    mask_ratio = 0.75

    # Initialize model
    model = AudiT(
        patch_size, 
        n_patches, 
        in_channels, 
        n_layers, 
        d_model,
        n_heads, 
        h_dims, 
        dropout, 
        activation, 
        batch_first, 
        mask_ratio
    )

    # Forward pass
    loss, pred, mask, patches = model(x)
    print(loss.item())


    def visualize_embeddings(tensor):
        plt.imshow(tensor[0].squeeze().detach().numpy(), cmap='magma', origin='lower')
        plt.colorbar()
        plt.xlabel('d_model')
        plt.ylabel('Patch Embeddings')
        plt.tight_layout()
        plt.show()

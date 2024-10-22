import torch
import torch.nn as nn
import einops


class Patchify(nn.Module):
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

        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, d_model)), requires_grad=True)
        self.position_embeddings = nn.Parameter(torch.randn(size=(1, n_patches+1, d_model)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.position_embeddings + x
        x = self.dropout(x)
        return x


class Encoder(nn.Module):

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

    def transform(self, x):
        pass

    def forward(self, x):
        out = self.encoder(x)
        return out
    

class Decoder(nn.Module):

   def __init__(self):
       super().__init__()

   def forward(self, x):
       pass
    

if __name__ == "__main__":

    dummy_in = torch.randn(16, 1, 512)
    enc = Encoder(n_layers=6, d_model=512, n_heads=8, h_dims=2048, dropout=0.2, activation='gelu', batch_first=True)
    out = enc(dummy_in)
    print(out.shape)

    print(enc)
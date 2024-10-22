import torch
import torch.nn as nn


class EncoderBlock(nn.Module):

    def __init__(
            self, 
            layers: int, 
            io_dims: int,
            activation: str,
            pooling_mode: str,
            dropout: float
        ):
        super().__init__()

        self.convolutional_layers = nn.ModuleList([])
        self.norm_layers = nn.ModuleList([])
        for _ in range(layers):
            conv_layer = nn.Conv1d(
                in_channels=io_dims, 
                out_channels=io_dims, 
                kernel_size=25, 
                stride=1, 
                padding=12
            )
            norm_layer = nn.BatchNorm1d(num_features=io_dims)
            self.convolutional_layers.append(conv_layer)
            self.norm_layers.append(norm_layer)

        assert activation in ('relu', 'leaky'), "Use 'relu' or 'leaky' for activation parameter"
        self.activation = (
            nn.ReLU() if activation == 'relu' 
            else nn.LeakyReLU(negative_slope=0.1)
        )

        assert pooling_mode in ('avg', 'max'), "Use 'avg' or 'max' for pooling_mode parameter"
        self.pooling_layer = (
            nn.AvgPool1d(kernel_size=2, stride=2) if pooling_mode == 'avg'
            else nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for conv, norm in zip(self.convolutional_layers, self.norm_layers):
            x = self.activation(norm(conv(x)))
            x = self.pooling_layer(x)
            x = self.dropout(x)
        return x
    

class DecoderBlock(nn.Module):

    def __init__(
            self, 
            layers: int, 
            io_dims: int, 
            kernel_size: int, 
            activation: str,
            upsample_mode: str,
            dropout: float
        ):
        super().__init__()

        self.layers = layers
        self.io_dims = io_dims

        self.convolutional_layers = nn.ModuleList([])
        self.norm_layers = nn.ModuleList([])
        for _ in range(layers):
            conv_layer = nn.Conv1d(
                in_channels=io_dims,
                out_channels=io_dims, 
                kernel_size=kernel_size, 
                stride=1,
                padding=12
            )
            norm_layer = nn.BatchNorm1d(num_features=io_dims)
            self.convolutional_layers.append(conv_layer)
            self.norm_layers.append(norm_layer)

        assert activation in ('relu', 'leaky'), "Use 'relu' or 'leaky' for activation parameter"
        self.activation = (
            nn.ReLU() if activation == 'relu' 
            else nn.LeakyReLU(negative_slope=0.1)
        )

        assert upsample_mode in ('linear', 'nearest'), "Use 'linear' or 'nearest' for upsample_mode parameter"
        self.upsample_layer = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=True)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for conv, norm in zip(self.convolutional_layers, self.norm_layers):
            x = self.activation(norm(conv(x)))
            x = self.upsample_layer(x)
            x = self.dropout(x)
        return x
    


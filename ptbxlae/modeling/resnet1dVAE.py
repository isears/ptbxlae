# Modified for 1d from: https://medium.com/@chen-yu/building-a-customized-residual-cnn-with-pytorch-471810e894ed

import torch
from torch.nn import (
    Conv1d,
    ReLU,
    BatchNorm1d,
    MaxPool1d,
    AvgPool1d,
    Linear,
    ConvTranspose1d,
    Upsample,
    Sequential,
    Flatten,
    Dropout,
    Unflatten,
    Tanh,
    Module,
    AdaptiveAvgPool1d,
    MSELoss,
)

from torchinfo import summary
import lightning as L
from typing import Literal
from ptbxlae.modeling import BaseVAE

# TODO: all this based on the assumption of seq_len = 1000, can add support for other lengths


# https://www.digitalocean.com/community/tutorials/writing-resnet-from-scratch-in-pytorch
class ResidualBlock1D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        task: Literal["encoder", "decoder"] = "encoder",
    ):
        super(ResidualBlock1D, self).__init__()

        if task == "encoder":
            self.conv1 = Sequential(
                Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                ),
                BatchNorm1d(out_channels),
                ReLU(),
            )

            self.conv2 = Sequential(
                Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                ),
                BatchNorm1d(out_channels),
            )

            self.scaling = Sequential(
                Conv1d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
                BatchNorm1d(out_channels),
            )
        elif task == "decoder":
            self.conv1 = Sequential(
                ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    output_padding=1,
                ),
                BatchNorm1d(out_channels),
                ReLU(),
            )

            self.conv2 = Sequential(
                ConvTranspose1d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                ),
                BatchNorm1d(out_channels),
            )

            self.scaling = Sequential(
                ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                    output_padding=1,
                ),
                BatchNorm1d(out_channels),
            )

        self.activation = ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        residual = self.scaling(x)

        out += residual
        return self.activation(out)


class Resnet1DEncoder(Module):

    def __init__(self, kernel_size: int, latent_dim: int):
        super(Resnet1DEncoder, self).__init__()

        assert (
            latent_dim < 192
        ), f"[-] Early bottleneck detected: latent_dim {latent_dim} greater than resnet features output dim"

        self.start_layer = Sequential(
            Conv1d(12, 24, kernel_size=13, stride=2, padding=6),
            BatchNorm1d(24),
            ReLU(),
            # MaxPool1d(kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
        )

        self.residual_blocks = Sequential(
            ResidualBlock1D(24, 48, kernel_size=kernel_size, task="encoder"),
            ResidualBlock1D(48, 96, kernel_size=kernel_size, task="encoder"),
            ResidualBlock1D(96, 192, kernel_size=kernel_size, task="encoder"),
        )

        self.final_layer = Sequential(AdaptiveAvgPool1d(output_size=1), Flatten())

    def forward(self, x):
        out = self.start_layer(x)
        out = self.residual_blocks(out)
        out = self.final_layer(out)

        return out


class Resnet1DDecoder(Module):
    def __init__(self, kernel_size: int, latent_dim: int):
        super(Resnet1DDecoder, self).__init__()

        self.start_layer = Sequential(
            Linear(latent_dim, 192),
            Unflatten(dim=1, unflattened_size=(192, 1)),
            Linear(1, 63),
        )

        self.residual_blocks = Sequential(
            ResidualBlock1D(192, 96, kernel_size=kernel_size, task="decoder"),
            ResidualBlock1D(96, 48, kernel_size=kernel_size, task="decoder"),
            ResidualBlock1D(48, 24, kernel_size=kernel_size, task="decoder"),
        )

        self.final_layer = Sequential(
            ConvTranspose1d(
                24, 12, kernel_size=13, stride=2, padding=6, output_padding=1
            ),
        )

    def forward(self, x):
        out = self.start_layer(x)
        out = self.residual_blocks(out)
        out = self.final_layer(out)

        # Decoder outputs 12, 1008
        # Will take only the middle 1000
        return out[:, :, 4:1004].contiguous()


class ResnetEcgVAE(BaseVAE):
    def __init__(
        self,
        lr,
        kernel_size: int = 7,
        latent_dim: int = 100,
        loss=None,
        base_model_path=None,
    ):
        super().__init__(lr, loss, base_model_path)

        self.encoder = Resnet1DEncoder(kernel_size=kernel_size, latent_dim=latent_dim)
        self.decoder = Resnet1DDecoder(kernel_size=kernel_size, latent_dim=latent_dim)

        self.fc_mean = Linear(192, 100)
        self.fc_logvar = Linear(192, 100)

    def encode_mean_logvar(self, x):
        e = self.encoder(x)
        return self.fc_mean(e), self.fc_logvar(e)

    def decode(self, encoded):
        return self.decoder(encoded)


if __name__ == "__main__":
    x = torch.rand(32, 12, 1000)

    e = Resnet1DEncoder(kernel_size=7, latent_dim=100)
    d = Resnet1DDecoder(kernel_size=7, latent_dim=100)

    encoded = e(x)
    print(encoded.shape)
    decoded = d(encoded)
    print(decoded.shape)

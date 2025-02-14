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
)

from torchinfo import summary
import lightning as L
from typing import Literal
from ptbxlae.modeling import BaseVAE


# https://www.digitalocean.com/community/tutorials/writing-resnet-from-scratch-in-pytorch
class ResidualBlock1D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernel_size: int = 7,
        task: Literal["encoder", "decoder"] = "encoder",
    ):
        super(ResidualBlock1D, self).__init__()

        if task == "encoder":
            _Conv = Conv1d
        elif task == "decoder":
            _Conv = ConvTranspose1d

        self.conv1 = Sequential(
            _Conv(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            ),
            BatchNorm1d(out_channels),
            ReLU(),
        )

        self.conv2 = Sequential(
            _Conv(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            BatchNorm1d(out_channels),
        )

        if stride != 1:
            self.scaling = Sequential(
                _Conv(
                    in_channels, out_channels, kernel_size=1, stride=stride, padding=0
                ),
                BatchNorm1d(out_channels),
            )
        else:
            self.scaling = None

        self.activation = ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.scaling:
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
            MaxPool1d(kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
        )

        self.residual_blocks = Sequential(
            ResidualBlock1D(24, 48, stride=2, kernel_size=kernel_size, task="encoder"),
            ResidualBlock1D(48, 96, stride=2, kernel_size=kernel_size, task="encoder"),
            ResidualBlock1D(96, 192, stride=2, kernel_size=kernel_size, task="encoder"),
        )

        self.final_layer = Sequential(
            AdaptiveAvgPool1d(output_size=1), Flatten(), Linear(192, latent_dim)
        )

    def forward(self, x):
        out = self.start_layer(x)
        out = self.residual_blocks(out)
        out = self.final_layer(out)

        return out


class Resnet1DDecoder(Module):
    def __init__(self, kernel_size: int, latent_dim: int):
        super(Resnet1DDecoder, self).__init__()

        self.start_layer = Sequential(
            AdaptiveAvgPool1d(output_size=1), Linear(192, latent_dim)
        )

        self.residual_blocks = Sequential(
            ResidualBlock1D(192, 96, stride=2, kernel_size=kernel_size, task="decoder"),
            ResidualBlock1D(96, 48, stride=2, kernel_size=kernel_size, task="decoder"),
            ResidualBlock1D(48, 24, stride=2, kernel_size=kernel_size, task="decoder"),
        )

        self.final_layer = Sequential(
            Conv1d(12, 24, kernel_size=13, stride=2, padding=6),
            BatchNorm1d(24),
            ReLU(),
            MaxPool1d(kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
        )


if __name__ == "__main__":
    x = torch.rand(32, 12, 1000)

    e = Resnet1DEncoder(kernel_size=7, latent_dim=100)
    out = e(x)

    print(out.shape)

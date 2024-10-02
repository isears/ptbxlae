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
)

from torchinfo import summary
import lightning as L


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ResidualBlock, self).__init__()

        self.conv1 = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            bias=False,
        )

        self.bn1 = BatchNorm1d(out_channels)
        self.activation = ReLU()

        self.conv2 = Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            bias=False,
        )

        self.bn2 = BatchNorm1d(out_channels)

        self.downsample = None

        if in_channels != out_channels:
            self.downsample = Sequential(
                Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                    bias=False,
                ),
                BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x + identity)

        return x


class Resnet1dEncoder(torch.nn.Module):

    def __init__(
        self,
        depth: int,
        kernel_size: int,
        dropout: float,
        latent_dim: int,
    ):
        super(Resnet1dEncoder, self).__init__()

        channels = 12

        self.conv = Conv1d(
            in_channels=channels,
            out_channels=24,
            kernel_size=kernel_size,
            padding="same",
            bias=False,
        )
        self.bn = BatchNorm1d(24)
        self.activation = ReLU()

        self.layer1 = Sequential(
            ResidualBlock(24, 24, kernel_size),
            ResidualBlock(24, 24, kernel_size),
            ResidualBlock(24, 24, kernel_size),
            AvgPool1d(kernel_size=(kernel_size - 1), stride=(kernel_size - 1)),
        )

        self.layer2 = Sequential(
            ResidualBlock(24, 48, kernel_size),
            ResidualBlock(48, 48, kernel_size),
            ResidualBlock(48, 48, kernel_size),
            AvgPool1d(kernel_size=(kernel_size - 1), stride=(kernel_size - 1)),
        )

        self.layer3 = Sequential(
            ResidualBlock(48, 96, kernel_size),
            ResidualBlock(96, 96, kernel_size),
            ResidualBlock(96, 96, kernel_size),
            AvgPool1d(kernel_size=(kernel_size - 1), stride=(kernel_size - 1)),
        )

        self.layer4 = Sequential(
            ResidualBlock(96, 192, kernel_size),
            ResidualBlock(192, 192, kernel_size),
            ResidualBlock(192, 192, kernel_size),
            AvgPool1d(kernel_size=(kernel_size - 1), stride=(kernel_size - 1)),
        )

        self.flatten = Flatten(1)
        self.dropout = Dropout(p=dropout)
        self.linear = Linear(in_features=3648, out_features=latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x


if __name__ == "__main__":
    e = Resnet1dEncoder(0, 5, 0.1, 40)

    x = torch.randn((16, 12, 5000)).cuda()
    summary(e.cuda(), x.shape)

    encoded = e(x)

    print(encoded.shape)

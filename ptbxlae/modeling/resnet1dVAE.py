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
)

from torchinfo import summary
import lightning as L


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        decoder: bool = False,
    ):
        super(ResidualBlock, self).__init__()

        same_padding = (kernel_size - 1) // 2

        if decoder:
            self.conv_layer_builder = (
                lambda in_channels, out_channels, kernel_size: ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=same_padding,
                    bias=False,
                )
            )
        else:
            self.conv_layer_builder = (
                lambda in_channels, out_channels, kernel_size: Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=same_padding,
                    bias=False,
                )
            )

        self.conv1 = self.conv_layer_builder(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

        self.bn1 = BatchNorm1d(out_channels)
        self.activation = ReLU()

        self.conv2 = self.conv_layer_builder(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

        self.bn2 = BatchNorm1d(out_channels)

        self.down_or_up_sample = None

        if in_channels != out_channels:
            self.down_or_up_sample = Sequential(
                self.conv_layer_builder(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                ),
                BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = self.down_or_up_sample(x) if self.down_or_up_sample else x

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
        # depth: int, TODO
        kernel_size: int,
        dropout: float,
        latent_dim: int,
        channels_in: int = 12,
        seq_len: int = 5000,
    ):
        super(Resnet1dEncoder, self).__init__()

        same_padding = (kernel_size - 1) // 2
        subsample_rate = 4

        self.conv = Conv1d(
            in_channels=channels_in,
            out_channels=24,
            kernel_size=kernel_size,
            padding=same_padding,
            bias=False,
        )
        self.bn = BatchNorm1d(24)
        self.activation = ReLU()

        self.layer1 = Sequential(
            ResidualBlock(24, 24, kernel_size),
            ResidualBlock(24, 24, kernel_size),
            ResidualBlock(24, 24, kernel_size),
            MaxPool1d(subsample_rate),
        )

        self.layer2 = Sequential(
            ResidualBlock(24, 48, kernel_size),
            ResidualBlock(48, 48, kernel_size),
            ResidualBlock(48, 48, kernel_size),
            MaxPool1d(subsample_rate),
        )

        self.layer3 = Sequential(
            ResidualBlock(48, 96, kernel_size),
            ResidualBlock(96, 96, kernel_size),
            ResidualBlock(96, 96, kernel_size),
            MaxPool1d(subsample_rate),
        )

        self.layer4 = Sequential(
            ResidualBlock(96, 192, kernel_size),
            ResidualBlock(192, 192, kernel_size),
            ResidualBlock(192, 192, kernel_size),
            MaxPool1d(subsample_rate),
        )

        last_layer_size = 192 * (seq_len // ((kernel_size - 1) ** 4))

        self.flatten = Flatten(1)
        self.dropout = Dropout(p=dropout)
        self.linear = Linear(in_features=last_layer_size, out_features=latent_dim)

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


class Resnet1dDecoder(torch.nn.Module):

    def __init__(
        self,
        # depth: int,
        kernel_size: int,
        dropout: float,
        latent_dim: int,
        channels_out: int = 12,
        seq_len: int = 5000,
    ):
        super(Resnet1dDecoder, self).__init__()

        first_layer_size = 192 * (seq_len // ((kernel_size - 1) ** 4))
        same_padding = (kernel_size - 1) // 2
        subsample_rate = 4

        # Need to track what the encoder seq dims would have been to upsample to the correct sizes
        encoder_seq_dims = [(seq_len // (subsample_rate**x)) for x in range(0, 4)]

        self.linear = Linear(in_features=latent_dim, out_features=first_layer_size)
        self.dropout = Dropout(dropout)
        self.unflatten = Unflatten(dim=1, unflattened_size=(192, -1))

        self.layer1 = Sequential(
            Upsample(size=encoder_seq_dims[-1]),
            ResidualBlock(192, 96, kernel_size, decoder=True),
            ResidualBlock(96, 96, kernel_size, decoder=True),
            ResidualBlock(96, 96, kernel_size, decoder=True),
        )

        self.layer2 = Sequential(
            Upsample(size=encoder_seq_dims[-2]),
            ResidualBlock(96, 48, kernel_size, decoder=True),
            ResidualBlock(48, 48, kernel_size, decoder=True),
            ResidualBlock(48, 48, kernel_size, decoder=True),
        )

        self.layer3 = Sequential(
            Upsample(size=encoder_seq_dims[-3]),
            ResidualBlock(48, 24, kernel_size, decoder=True),
            ResidualBlock(24, 24, kernel_size, decoder=True),
            ResidualBlock(24, 24, kernel_size, decoder=True),
        )

        self.layer4 = Sequential(
            Upsample(size=encoder_seq_dims[-4]),
            ResidualBlock(24, 24, kernel_size, decoder=True),
            ResidualBlock(24, 24, kernel_size, decoder=True),
            ResidualBlock(24, 24, kernel_size, decoder=True),
        )

        self.conv = ConvTranspose1d(
            in_channels=24,
            out_channels=channels_out,
            kernel_size=kernel_size,
            padding=same_padding,
            bias=False,
        )

        self.bn = BatchNorm1d(12)
        # TODO: is relu really appropriate here?
        self.activation = ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        x = self.unflatten(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)

        return x


class Resnet1dVAE(L.LightningModule):

    def __init__(
        self,
        kernel_size: int = 5,
        latent_dim: int = 40,
        dropout: float = 0.1,
        lr: float = 1e-3,
    ):
        super(Resnet1dVAE, self).__init__()
        self.encoder = Resnet1dEncoder(kernel_size, dropout, latent_dim)
        self.decoder = Resnet1dDecoder(kernel_size, dropout, latent_dim)

        self.mu = Linear(latent_dim, latent_dim)
        self.sigma = Linear(latent_dim, latent_dim)

        self.lr = lr

    def _reparameterization(self, mean, var):
        e = torch.randn_like(var)
        return mean + (var * e)

    def _loss_fn(self, x, x_hat, mean, var):
        reproduction_loss = torch.nn.functional.mse_loss(x_hat, x, reduction="mean")
        KLD = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp())

        return reproduction_loss + KLD

    def forward(self, x):
        encoded = self.encoder(x)
        m, s = self.mu(encoded), self.sigma(encoded)
        z = self._reparameterization(m, s)
        x_hat = self.decoder(z)

        loss = self._loss_fn(x, x_hat, m, s)

        return loss

    def training_step(self, x):
        loss = self.forward(x)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, x):
        loss = self.forward(x)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    e = Resnet1dEncoder(5, 0.1, 40).cuda()
    d = Resnet1dDecoder(5, 0.1, 40).cuda()

    x = torch.randn((16, 12, 5000)).cuda()

    summary(e, x.shape)

    encoded = e(x)

    print(encoded.shape)

    # summary(d, encoded.shape)

    decoded = d(encoded)

    print(decoded.shape)

import torch
from torch.nn import (
    Conv1d,
    ReLU,
    LeakyReLU,
    BatchNorm1d,
    MaxPool1d,
    Linear,
    ConvTranspose1d,
    Upsample,
    Sequential,
    Flatten,
    Unflatten,
    Dropout,
)

from ptbxlae.modeling import BaseVAE

import lightning as L


class SingleCycleConvVAE(BaseVAE):

    def __init__(
        self,
        lr: float = 1e-3,
        kernel_size: int = 5,
        latent_dim: int = 40,
        seq_len: int = 500,
        n_channels: int = 12,
        conv_depth: int = 2,
        fc_depth: int = 1,
        batchnorm: bool = False,
        dropout: float = None,
    ):
        super(SingleCycleConvVAE, self).__init__()

        self.lr = lr
        self.latent_dim = latent_dim
        self.conv_depth = conv_depth
        self.fc_depth = fc_depth
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.padding_required = False
        self.seq_len = seq_len
        same_padding = (kernel_size - 1) // 2

        if not (seq_len % (2**self.conv_depth) == 0):
            self.padding_required = True

            pad_amount = (2**self.conv_depth) - (seq_len % (2**self.conv_depth))
            self.seq_len = seq_len + pad_amount
            self.left_pad = pad_amount // 2
            self.right_pad = pad_amount - self.left_pad

            print(
                f"Warning: seq_len {seq_len} not divisible by 2 ** {conv_depth}, will pad up to {seq_len + pad_amount}"
            )

        linear_input_sizes = [
            (self.seq_len // (2**self.conv_depth)) * ((2**self.conv_depth) * n_channels)
        ]

        for idx in range(0, self.fc_depth - 1):
            next_layer_size = linear_input_sizes[-1] // 4

            if next_layer_size > self.latent_dim:
                linear_input_sizes.append(next_layer_size)
            else:
                linear_input_sizes.append(latent_dim)

        # Build encoder
        encoder_layers = list()

        for idx in range(0, self.conv_depth):
            in_channels = n_channels * (2**idx)

            encoder_layers += [
                Conv1d(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=same_padding,
                ),
                LeakyReLU(),
            ]

            if self.batchnorm:
                encoder_layers.append(BatchNorm1d(num_features=(2 * in_channels)))

        encoder_layers.append(Flatten(start_dim=1, end_dim=-1))

        for idx in range(1, self.fc_depth):
            encoder_layers += [
                Linear(linear_input_sizes[idx - 1], linear_input_sizes[idx]),
                LeakyReLU(),
            ]

            if self.dropout:
                encoder_layers.append(Dropout(self.dropout))

        self.encoder = Sequential(*encoder_layers)

        # Mean / logvar layers
        self.fc_mean = Linear(
            linear_input_sizes[-1],
            latent_dim,
        )
        self.fc_logvar = Linear(
            linear_input_sizes[-1],
            latent_dim,
        )

        # Build decoder
        decoder_layers = list()
        decoder_layers += [Linear(latent_dim, linear_input_sizes[-1]), LeakyReLU()]

        for idx in range(self.fc_depth - 1, 0, -1):
            decoder_layers += [
                Linear(linear_input_sizes[idx], linear_input_sizes[idx - 1]),
                LeakyReLU(),
            ]

            if self.dropout:
                decoder_layers.append(Dropout(self.dropout))

        decoder_layers.append(
            Unflatten(
                dim=1,
                unflattened_size=(
                    n_channels * (2**self.conv_depth),
                    self.seq_len // (2**self.conv_depth),
                ),
            ),
        )

        for idx in range(self.conv_depth, 0, -1):
            in_channels = n_channels * (2**idx)

            decoder_layers += [
                ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=in_channels // 2,
                    kernel_size=kernel_size,
                    padding=same_padding,
                    stride=2,
                    output_padding=1,
                ),
                LeakyReLU(),
            ]

            if self.batchnorm:
                decoder_layers.append(BatchNorm1d(num_features=in_channels // 2))

        self.decoder = Sequential(*decoder_layers)

    def encode_mean_logvar(self, x):
        if self.padding_required:
            x = torch.nn.functional.pad(
                x, pad=(self.left_pad, self.right_pad), value=0.0
            )

        e = self.encoder(x)
        return self.fc_mean(e), self.fc_logvar(e)

    def decode(self, encoded):
        reconstruction = self.decoder(encoded)

        if self.padding_required:
            reconstruction = reconstruction[:, :, self.left_pad : -self.right_pad]

        return reconstruction


if __name__ == "__main__":
    x = torch.rand((4, 12, 500))

    m = SingleCycleConvVAE(
        conv_depth=3, fc_depth=2, latent_dim=40, dropout=0.1, batchnorm=True
    )
    e = m.encoder(x)
    z = m.encode(x)

    mean, logvar = m.encode_mean_logvar(x)

    print(f"Encoder shape:\t {e.shape}")
    print(f"Encoded shape:\t {z.shape}")
    print(f"Mean shape:\t {mean.shape}")
    print(f"Logvar shape:\t {logvar.shape}")

    d = m.decode(z)

    print(f"Decoded shape:\t {d.shape}")

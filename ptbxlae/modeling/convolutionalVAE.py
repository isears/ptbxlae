import torch
from torch.nn import Linear, Sequential, LeakyReLU, Sigmoid

from ptbxlae.modeling import BaseVAE
from ptbxlae.modeling.convolutionalModules import (
    ConvolutionalEcgEncoder,
    ConvolutionalEcgDecoder,
    ConvolutionalEcgEncoderDecoderSharedParams,
)

import lightning as L


class ConvolutionalEcgVAE(BaseVAE):

    def __init__(
        self,
        lr: float = 1e-3,
        kernel_size: int = 5,
        latent_dim: int = 40,
        seq_len: int = 500,
        n_channels: int = 12,
        conv_depth: int = 2,
        fc_depth: int = 1,
        fc_scale_factor: int = 4,
        batchnorm: bool = False,
        dropout: float = None,
    ):
        super(ConvolutionalEcgVAE, self).__init__()

        self.lr = lr
        self.padding_required = False
        self.seq_len = seq_len

        if not (seq_len % (2**conv_depth) == 0):
            self.padding_required = True

            pad_amount = (2**conv_depth) - (seq_len % (2**conv_depth))
            self.seq_len = seq_len + pad_amount
            self.left_pad = pad_amount // 2
            self.right_pad = pad_amount - self.left_pad

            print(
                f"Warning: seq_len {seq_len} not divisible by 2 ** {conv_depth}, will pad up to {seq_len + pad_amount}"
            )

        shared_params = ConvolutionalEcgEncoderDecoderSharedParams(
            seq_len,
            n_channels,
            kernel_size,
            conv_depth,
            fc_depth,
            latent_dim,
            fc_scale_factor,
        )

        self.encoder = ConvolutionalEcgEncoder(
            shared_params,
            batchnorm,
            dropout,
        )

        self.decoder = ConvolutionalEcgDecoder(
            shared_params,
            batchnorm,
            dropout,
        )

        # Mean / logvar layers
        self.fc_mean = Linear(
            self.encoder.architecture_params.linear_input_sizes[-1],
            latent_dim,
        )
        self.fc_logvar = Linear(
            self.encoder.architecture_params.linear_input_sizes[-1],
            latent_dim,
        )

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

    m = ConvolutionalEcgVAE(
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

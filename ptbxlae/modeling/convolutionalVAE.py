import torch
from torch.nn import Linear, Sequential, LeakyReLU, Sigmoid

from ptbxlae.modeling import BaseVAE
from ptbxlae.modeling.convolutionalModules import (
    ConvolutionalEcgEncoder,
    ConvolutionalEcgDecoder,
    ConvolutionalEcgEncoderDecoderSharedParams,
)
from torchinfo import summary

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
        self.seq_len = seq_len

        shared_params = ConvolutionalEcgEncoderDecoderSharedParams(
            seq_len,
            n_channels,
            kernel_size,
            conv_depth,
            fc_depth,
            latent_dim,
            fc_scale_factor,
        )

        self.encoder = ConvolutionalEcgEncoder(shared_params, batchnorm, dropout)

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

        summary(self)

    def encode_mean_logvar(self, x):
        e = self.encoder(x)
        return self.fc_mean(e), self.fc_logvar(e)

    def decode(self, encoded):
        return self.decoder(encoded)


if __name__ == "__main__":
    x = torch.rand((8, 12, 1000))

    m = ConvolutionalEcgVAE(
        seq_len=1000,
        conv_depth=2,
        fc_depth=2,
        kernel_size=7,
        latent_dim=100,
        dropout=0.1,
        batchnorm=True,
    )
    e = m.encoder(x)
    print(f"Encoder shape:\t {e.shape}")
    z = m.encode(x)
    print(f"Encoded shape:\t {z.shape}")

    mean, logvar = m.encode_mean_logvar(x)
    print(f"Mean shape:\t {mean.shape}")
    print(f"Logvar shape:\t {logvar.shape}")

    d = m.decode(z)

    print(f"Decoded shape:\t {d.shape}")

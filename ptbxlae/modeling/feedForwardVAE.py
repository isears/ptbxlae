import torch
from torch.nn import Sequential, Linear, LeakyReLU
import lightning as L
from torch.distributions.normal import Normal
from ptbxlae.modeling import BaseVAE


class SingleChannelFFNNVAE(BaseVAE):

    def __init__(
        self,
        seq_len: int = 500,
        latent_dim: int = 40,
        n_layers: int = 3,
        lr: float = 1e-3,
    ):
        super(SingleChannelFFNNVAE, self).__init__()

        self.lr = lr
        self.latent_dim = latent_dim

        encoder_layers = list()
        decoder_layers = list()
        this_layer_dim = seq_len

        for _ in range(0, n_layers):

            if this_layer_dim // 2 > latent_dim:
                next_layer_dim = this_layer_dim // 2
            else:
                next_layer_dim = latent_dim

            encoder_layers += [Linear(this_layer_dim, next_layer_dim), LeakyReLU()]

            decoder_layers = [
                LeakyReLU(),
                Linear(next_layer_dim, this_layer_dim),
            ] + decoder_layers

            this_layer_dim = next_layer_dim

        self.encoder = Sequential(*encoder_layers)
        self.decoder = Sequential(*decoder_layers)

        self.mean = Linear(self.latent_dim, self.latent_dim)
        self.logvar = Linear(self.latent_dim, self.latent_dim)

    def encode_mean_logvar(self, x):
        e = self.encoder(x)
        return self.mean(e), self.logvar(e)

    def decode(self, encoded):
        return self.decoder(encoded)

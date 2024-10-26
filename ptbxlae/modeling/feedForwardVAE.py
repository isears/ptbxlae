import torch
from torch.nn import Sequential, Linear, LeakyReLU
import lightning as L
from torch.distributions.normal import Normal
from ptbxlae.modeling import BaseVAE


class SingleChannelFFNNVAE(BaseVAE):

    def __init__(self, seq_len: int = 500, latent_dim: int = 40, lr: float = 1e-3):
        super(SingleChannelFFNNVAE, self).__init__()

        self.lr = lr
        self.latent_dim = latent_dim

        self.encoder = Sequential(
            Linear(seq_len, seq_len // 2),
            LeakyReLU(),
            Linear(seq_len // 2, seq_len // 4),
            LeakyReLU(),
        )

        self.decoder = Sequential(
            Linear(latent_dim, seq_len // 4),
            LeakyReLU(),
            Linear(seq_len // 4, seq_len // 2),
            LeakyReLU(),
            Linear(seq_len // 2, seq_len),
            # TODO: activation? Sigmoid?
        )

        self.mean = Linear(seq_len // 4, self.latent_dim)
        self.logvar = Linear(seq_len // 4, self.latent_dim)

    def encode_mean_logvar(self, x):
        e = self.encoder(x)
        return self.mean(e), self.logvar(e)

    def decode(self, encoded):
        return self.decoder(encoded)

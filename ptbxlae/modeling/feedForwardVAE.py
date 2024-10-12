import torch
from torch.nn import Sequential, Linear, LeakyReLU
import lightning as L
from torch.distributions.normal import Normal


class SingleChannelFFNNVAE(L.LightningModule):

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

    def reparamterization(self, z_mean, z_logvar):
        # get the shape of the tensor for the mean and log variance
        batch, dim = z_mean.shape
        # generate a normal random tensor (epsilon) with the same shape as z_mean
        # this tensor will be used for reparameterization trick
        epsilon = Normal(0, 1).sample((batch, dim)).to(z_mean.device)
        # apply the reparameterization trick to generate the samples in the
        # latent space
        return z_mean + torch.exp(0.5 * z_logvar) * epsilon

    def _loss_fn(self, sig, padmask, reconstruction, mean, var):
        reconstruction = reconstruction * padmask
        reproduction_loss = torch.nn.functional.mse_loss(
            reconstruction, sig, reduction="mean"
        )
        KLD = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp())
        return reproduction_loss + KLD

    def encode(self, x):
        sig, padmask = x
        lead_II = sig[:, 1, :]
        encoded = self.encoder(lead_II)

        z_mean = self.mean(encoded)
        z_logvar = self.logvar(encoded)

        z = self.reparamterization(z_mean, z_logvar)

        return z

    def forward(self, x):
        sig, padmask = x
        lead_II = sig[:, 1, :]

        encoded = self.encoder(lead_II)
        z_mean = self.mean(encoded)
        z_logvar = self.logvar(encoded)

        z = self.reparamterization(z_mean, z_logvar)
        reconstruction = self.decoder(z)

        loss = self._loss_fn(lead_II, padmask, reconstruction, z_mean, z_logvar)

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

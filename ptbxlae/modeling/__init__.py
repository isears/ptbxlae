# https://pyimagesearch.com/2023/10/02/a-deep-dive-into-variational-autoencoders-with-pytorch/

import lightning as L
from torch.distributions.normal import Normal
import torch
from abc import ABC, abstractmethod


class BaseVAE(L.LightningModule, ABC):

    def __init__(self):
        super(BaseVAE, self).__init__()
        self.save_hyperparameters()

    @abstractmethod
    def encode_mean_logvar(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        pass

    def generate(self, latent: torch.Tensor, smoothing_iterations: int = 10):
        return torch.mean(
            torch.stack(
                [self.decode(latent) for idx in range(0, smoothing_iterations)]
            ),
            dim=0,
        )

    def _reparameterization(self, z_mean, z_logvar):
        batch, dim = z_mean.shape
        epsilon = Normal(0, 1).sample((batch, dim)).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_logvar) * epsilon

    def _loss_fn(self, x, reconstruction, mean, logvar):
        # NOTE: the loss reduction for variational autoencoder must be sum
        reproduction_loss = torch.nn.functional.mse_loss(
            reconstruction, x, reduction="sum"
        )
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        return reproduction_loss + KLD

    def encode(self, x):
        """
        For getting encodings outside of training loop
        """
        z = self._reparameterization(*self.encode_mean_logvar(x))
        return z

    def forward(self, x):
        mean, logvar = self.encode_mean_logvar(x)
        z = self._reparameterization(mean, logvar)
        reconstruction = self.decode(z)

        loss = self._loss_fn(x, reconstruction, mean, logvar)

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

    # For use with autolr feature that never really worked
    # def on_fit_start(self):
    #     if self.lr:
    #         # Overwrite learning rate after running LearningRateFinder
    #         for optimizer in self.trainer.optimizers:
    #             for param_group in optimizer.param_groups:
    #                 param_group["lr"] = self.lr

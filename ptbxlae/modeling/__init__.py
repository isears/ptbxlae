# https://pyimagesearch.com/2023/10/02/a-deep-dive-into-variational-autoencoders-with-pytorch/

import lightning as L
from torch.distributions.normal import Normal
import torch
from abc import ABC, abstractmethod
from neptune.types import File
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger
from torchmetrics.regression.mse import MeanSquaredError
import matplotlib.pyplot as plt
from tslearn.metrics import SoftDTWLossPyTorch
from torch.nn import MSELoss

class SumReducingSoftDTWLoss(SoftDTWLossPyTorch):
    def __init__(self, gamma=1, normalize=False, dist_func=None):
        super().__init__(gamma, normalize, dist_func)

    def forward(self, x, y):
        return super().forward(x, y).sum()
    
class NeptuneUploadingModelCheckpoint(ModelCheckpoint):
    def on_train_start(self, trainer, pl_module):

        self.example_batch = torch.stack(
            [trainer.val_dataloaders.dataset[idx] for idx in range(0, 20)]
        )

        return super().on_train_start(trainer, pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        pl_module.eval()
        with torch.no_grad():
            recon, _, _ = pl_module(self.example_batch.to("cuda"))

        for idx in range(0, self.example_batch.shape[0]):
            # ensure distribution over available channels while never exceeding number of channels
            channel_idx = idx % recon.shape[1]

            fig, ax = plt.subplots()
            x = range(0, recon.shape[-1])
            ax.plot(x, self.example_batch[idx, channel_idx, :], label="original")
            ax.plot(x, recon[idx, channel_idx, :].to("cpu"), label="reconstruction")
            fig.suptitle(f"Epoch {trainer.current_epoch}")

            trainer.logger.experiment[
                f"visuals/reconstruction-epoch{trainer.current_epoch}-example{idx}"
            ].upload(File.as_html(fig))

        pl_module.train()

        return super().on_save_checkpoint(trainer, pl_module, checkpoint)

    def on_fit_end(self, trainer, pl_module):
        self.best_model_path

        if type(trainer.logger) == NeptuneLogger:
            trainer.logger.experiment["model/checkpoints/best.ckpt"].upload(
                self.best_model_path
            )

        return super().on_fit_end(trainer, pl_module)


class BaseVAE(L.LightningModule, ABC):

    def __init__(self, loss=None):
        super(BaseVAE, self).__init__()

        if not loss:
            self.loss = MSELoss(reduction='sum')
        else:
            self.loss = loss

        self.train_mse = MeanSquaredError()
        self.valid_mse = MeanSquaredError()
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
        reproduction_loss = self.loss(reconstruction, x)
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
        return reconstruction, mean, logvar

    def training_step(self, x):
        reconstruction, mean, logvar = self.forward(x)
        loss = self._loss_fn(x, reconstruction, mean, logvar)
        self.train_mse.update(reconstruction, x)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "train_mse", self.train_mse, on_step=False, on_epoch=True, prog_bar=True
        )

        return loss

    def validation_step(self, x):
        reconstruction, mean, logvar = self.forward(x)
        loss = self._loss_fn(x, reconstruction, mean, logvar)
        self.valid_mse.update(reconstruction, x)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_mse", self.valid_mse, on_step=False, on_epoch=True, prog_bar=True)

        return loss

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

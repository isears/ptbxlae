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
from ptbxlae.evaluation import LatentRepresentationUtilityMetric
from typing import Optional


class SumReducingSoftDTWLoss(SoftDTWLossPyTorch):
    def __init__(self, gamma=1, normalize=False, dist_func=None):
        super().__init__(gamma, normalize, dist_func)

    def forward(self, x, y):
        return super().forward(x, y).sum()


class NeptuneUploadingModelCheckpoint(ModelCheckpoint):

    def __init__(
        self, log_sample_reconstructions: bool, num_examples: int = 1, **kwargs
    ):
        super().__init__(**kwargs)

        self.log_sample_reconstructions = log_sample_reconstructions
        self.num_examples = num_examples
        self.best_model_epoch = 0

    def on_train_start(self, trainer, pl_module):
        if self.log_sample_reconstructions:
            self.example_batch = torch.stack(
                [
                    trainer.val_dataloaders.dataset[idx][0]
                    for idx in range(0, self.num_examples)
                ]
            )

        if type(trainer.logger) == NeptuneLogger:
            trainer.logger.experiment["sys/tags"].add(
                [
                    f"model_{pl_module.__class__.__name__}",
                ]
            )

        return super().on_train_start(trainer, pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        self.best_model_epoch = trainer.current_epoch
        if self.log_sample_reconstructions:
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

                if type(trainer.logger) == NeptuneLogger:
                    trainer.logger.experiment[
                        f"val/reconstructions/epoch-{trainer.current_epoch}/example-{idx}"
                    ].upload(File.as_html(fig))

                plt.close(fig=fig)

            pl_module.train()

        return super().on_save_checkpoint(trainer, pl_module, checkpoint)

    def on_fit_end(self, trainer, pl_module):
        # Only save model once at end of training to avoid overhead / delays associated with uploading every model checkpoint
        if type(trainer.logger) == NeptuneLogger and self.log_sample_reconstructions:
            trainer.logger.experiment["model/checkpoints/best"].upload(
                self.best_model_path
            )

        return super().on_fit_end(trainer, pl_module)


class BaseModel(L.LightningModule, ABC):
    def __init__(
        self,
        lr: float,
        loss: Optional[torch.nn.Module] = None,
        base_model_path: Optional[str] = None,
    ):
        super(BaseModel, self).__init__()
        self.lr = lr

        if not loss:
            self.loss = MSELoss(reduction="sum")
        else:
            self.loss = loss

        self.base_model_path = base_model_path
        self.test_label_evaluator = LatentRepresentationUtilityMetric()
        self.train_mse = MeanSquaredError()
        self.valid_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()
        self.save_hyperparameters()

    def on_train_start(self):
        if self.base_model_path:
            weights = torch.load(
                self.base_model_path, map_location=self.device, weights_only=True
            )
            self.load_state_dict(weights["state_dict"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class BaseVAE(BaseModel, ABC):

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

    def training_step(self, batch):
        x, _ = batch
        reconstruction, mean, logvar = self.forward(x)
        loss = self._loss_fn(x, reconstruction, mean, logvar)
        self.train_mse.update(reconstruction, x)

        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "train_mse",
            self.train_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch):
        x, _ = batch
        reconstruction, mean, logvar = self.forward(x)
        loss = self._loss_fn(x, reconstruction, mean, logvar)
        self.valid_mse.update(reconstruction, x)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "val_mse",
            self.valid_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def test_step(self, batch):
        x, labels = batch
        mean, logvar = self.encode_mean_logvar(x)
        z = self._reparameterization(mean, logvar)
        reconstruction = self.decode(z)

        loss = self._loss_fn(x, reconstruction, mean, logvar)
        self.test_mse.update(reconstruction, x)
        self.test_label_evaluator.update(z, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "test_mse", self.test_mse, on_step=False, on_epoch=True, sync_dist=True
        )

        return loss

    def on_test_epoch_end(self):
        label_evals = self.test_label_evaluator.compute()
        for label, score in label_evals.items():
            self.log(f"AUROC ({label})", score)

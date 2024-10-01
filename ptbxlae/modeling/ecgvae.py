import torch
from torch.nn import (
    Conv1d,
    ReLU,
    BatchNorm1d,
    MaxPool1d,
    Linear,
    ConvTranspose1d,
    Upsample,
)

import lightning as L

"""
Simple 3-layer encoder / decoder combo without any adjustable parameters for debugging
"""


class StaticSimpleEncoder(torch.nn.Module):

    def __init__(self):
        super(StaticSimpleEncoder, self).__init__()

        # NOTE: Maxpool kernel size 5 is likely too high, should be smaller (maybe 2)
        self.conv1 = Conv1d(in_channels=12, out_channels=24, kernel_size=5, padding=2)
        self.bn1 = BatchNorm1d(num_features=24)
        self.nonlinearity1 = ReLU()
        self.maxpool1 = MaxPool1d(kernel_size=5)

        self.conv2 = Conv1d(in_channels=24, out_channels=48, kernel_size=5, padding=2)
        self.bn2 = BatchNorm1d(num_features=48)
        self.nonlinearity2 = ReLU()
        self.maxpool2 = MaxPool1d(kernel_size=5)

        self.conv3 = Conv1d(in_channels=48, out_channels=96, kernel_size=5, padding=2)
        self.bn3 = BatchNorm1d(num_features=96)
        self.nonlinearity3 = ReLU()
        self.maxpool3 = MaxPool1d(kernel_size=5)

        self.conv4 = Conv1d(in_channels=96, out_channels=1, kernel_size=5, padding=2)

        # self.mu = Linear(40, 20)
        # self.sd = Sequential(
        #     Linear(40, 20),
        #     Softplus()
        # )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nonlinearity1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.nonlinearity2(out)
        out = self.maxpool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.nonlinearity3(out)
        out = self.maxpool3(out)

        out = self.conv4(out)
        out = torch.flatten(out, start_dim=1, end_dim=-1)

        return out


class StaticSimpleDecoder(torch.nn.Module):

    def __init__(self):
        super(StaticSimpleDecoder, self).__init__()

        self.conv1 = ConvTranspose1d(
            in_channels=1, out_channels=96, kernel_size=5, padding=2
        )
        self.bn1 = BatchNorm1d(num_features=96)
        self.nonlinearity1 = ReLU()
        self.upsample1 = Upsample(scale_factor=5)

        self.conv2 = ConvTranspose1d(
            in_channels=96, out_channels=48, kernel_size=5, padding=2
        )
        self.bn2 = BatchNorm1d(num_features=48)
        self.nonlinearity2 = ReLU()
        self.upsample2 = Upsample(scale_factor=5)

        self.conv3 = ConvTranspose1d(
            in_channels=48, out_channels=24, kernel_size=5, padding=2
        )
        self.bn3 = BatchNorm1d(num_features=24)
        self.nonlinearity3 = ReLU()
        self.upsample3 = Upsample(scale_factor=5)

        self.conv4 = ConvTranspose1d(
            in_channels=24, out_channels=12, kernel_size=5, padding=2
        )

    def forward(self, x):
        out = torch.unflatten(x, 1, (1, 40))
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.nonlinearity1(out)
        out = self.upsample1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.nonlinearity2(out)
        out = self.upsample2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.nonlinearity3(out)
        out = self.upsample3(out)

        out = self.conv4(out)

        return out


class StaticSimpleVAE(L.LightningModule):

    def __init__(self, lr: float = 1e-3):
        super(StaticSimpleVAE, self).__init__()

        self.encoder = StaticSimpleEncoder()
        self.decoder = StaticSimpleDecoder()

        self.mu = Linear(40, 40)
        self.sigma = Linear(40, 40)

        self.lr = lr

    def _reparameterization(self, mean, var):
        e = torch.randn_like(var)
        return mean + (var * e)

    def _loss_fn(self, x, x_hat, mean, var):
        reproduction_loss = torch.nn.functional.mse_loss(x_hat, x, reduction="sum")
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

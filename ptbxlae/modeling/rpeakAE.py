from ptbxlae.modeling import BaseVAE
from ptbxlae.modeling.convolutionalModules import (
    ConvolutionalEcgEncoder,
    ConvolutionalEcgDecoder,
)
from torch.nn import (
    Linear,
    LeakyReLU,
    Sequential,
    Sigmoid,
    Conv1d,
    MaxPool1d,
    Flatten,
    Unflatten,
    Upsample,
    ConvTranspose1d,
)
import torch


class RpeakAE(BaseVAE):
    """
    Relatively simple encoder / decoder to identify R peaks
    """

    def __init__(
        self,
        lr=0.001,
        kernel_size=5,
        conv_depth=2,
        fc_depth=1,
    ):
        super().__init__()
        self.lr = lr
        pad = (kernel_size - 1) // 2

        self.encoder = Sequential(Linear(5000, 2500), LeakyReLU(), Linear(2500, 10))

        self.decoder = Sequential(
            Linear(10, 2500),
            LeakyReLU(),
            Linear(2500, 5000),
            Sigmoid(),
        )

    def encode_mean_logvar(self, x):
        raise NotImplementedError()

    def decode(self, encoded):
        return self.decoder(encoded)

    def forward(self, x):
        sig, rpeaks = x
        z = self.encoder(sig[:, 1, :].unsqueeze(1))
        reconstruction = torch.nn.functional.sigmoid(self.decode(z))

        loss = self._loss_fn(rpeaks.squeeze(), reconstruction.squeeze())

        return loss

    def _loss_fn(self, x, reconstruction):
        reproduction_loss = torch.nn.functional.binary_cross_entropy(
            reconstruction, x, reduction="mean"
        )
        return reproduction_loss

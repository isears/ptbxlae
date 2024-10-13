import torch
from torch.nn import (
    Conv1d,
    ReLU,
    LeakyReLU,
    BatchNorm1d,
    MaxPool1d,
    Linear,
    ConvTranspose1d,
    Upsample,
    Sequential,
    Flatten,
    Unflatten,
)

from ptbxlae.modeling import BaseVAE

import lightning as L


class SingleCycleConvVAE(BaseVAE):

    def __init__(
        self,
        lr: float = 1e-3,
        kernel_size: int = 5,
        latent_dim: int = 25,
        seq_len: int = 500,
        n_channels: int = 12,
    ):
        super(SingleCycleConvVAE, self).__init__()

        self.lr = lr

        same_padding = (kernel_size - 1) // 2

        assert (
            seq_len % 4 == 0
        ), f"{seq_len} not divisible by 4, which is required by current architecture"

        self.encoder = Sequential(
            Conv1d(
                in_channels=n_channels,
                out_channels=(2 * n_channels),
                kernel_size=kernel_size,
                stride=2,
                padding=same_padding,
            ),
            LeakyReLU(),
            Conv1d(
                in_channels=(2 * n_channels),
                out_channels=(4 * n_channels),
                kernel_size=kernel_size,
                stride=2,
                padding=same_padding,
            ),
            LeakyReLU(),
            Flatten(start_dim=1, end_dim=-1),
        )

        self.fc_mean = Linear((seq_len // 4) * (4 * n_channels), latent_dim)
        self.fc_logvar = Linear((seq_len // 4) * (4 * n_channels), latent_dim)

        self.decoder = Sequential(
            Linear(latent_dim, 1200),
            Unflatten(dim=1, unflattened_size=(48, 25)),
            Upsample(scale_factor=4),
            ConvTranspose1d(
                in_channels=48,
                out_channels=24,
                kernel_size=kernel_size,
                padding=same_padding,
            ),
            BatchNorm1d(num_features=24),
            LeakyReLU(),
            Upsample(scale_factor=5),
            ConvTranspose1d(
                in_channels=24,
                out_channels=12,
                kernel_size=kernel_size,
                padding=same_padding,
            ),
            # TODO: do we need these last two?
            # BatchNorm1d(num_features=12),
            # LeakyReLU(),
        )

    def encode_mean_logvar(self, x):
        e = self.encoder(x)
        return self.fc_mean(e), self.fc_logvar(e)

    def decode(self, encoded):
        return self.decoder(encoded)


if __name__ == "__main__":
    x = torch.rand((4, 12, 500))

    m = SingleCycleConvVAE()
    e = m.encoder(x)
    z = m.encode(x)

    mean, logvar = m.encode_mean_logvar(x)

    print(e.shape)
    print(z.shape)
    print(mean.shape)
    print(logvar.shape)

    d = m.decoder(z)

    print(d.shape)

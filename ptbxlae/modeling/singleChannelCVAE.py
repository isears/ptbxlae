import torch
from ptbxlae.modeling import BaseVAE
from ptbxlae.modeling.convolutionalModules import *
from torch.nn import MSELoss
import torch.nn.functional as F
from typing import Tuple
from torchmetrics.regression.mse import MeanSquaredError


class CustomUnwrappingMSE(MeanSquaredError):
    """
    Although model returns both a signal reconstruction and a class reconstruction
    only care about the signal reconstruction MSE
    """

    def update(self, reconstruction, x):
        original_sig, _ = x
        sig_recon, _ = reconstruction
        return super().update(sig_recon, original_sig)


class SingleChannelCVAE(BaseVAE):
    """
    Architecture for autoencoding a single channel
    An additional decoder guesses what kind of waveform is present (classifier output)
    For now, classifier will guess which EKG lead (I-V6)
    But later intend to add arterial waveforms and pleth
    """

    def __init__(
        self,
        lr: float = 1e-3,
        kernel_size: int = 5,
        latent_dim: int = 10,
        seq_len: int = 500,
        conv_depth: int = 2,
        fc_depth: int = 1,
        fc_scale_factor: int = 4,
        batchnorm: bool = False,
        dropout: Optional[float] = None,
        classification_loss_weight: float = 0.1,
        n_classes: int = 12,
        **kwargs,
    ):
        super(SingleChannelCVAE, self).__init__(
            lr=lr, loss=MSELoss(reduction="sum"), **kwargs
        )

        self.classification_loss_weight = classification_loss_weight

        shared_params = ConvolutionalEcgEncoderDecoderSharedParams(
            seq_len, 1, kernel_size, conv_depth, fc_depth, latent_dim, fc_scale_factor
        )

        self.encoder = ConvolutionalEcgEncoder(shared_params, batchnorm, dropout)

        self.signal_decoder = ConvolutionalEcgDecoder(shared_params, batchnorm, dropout)

        self.class_decoder = Sequential(
            Linear(latent_dim, n_classes), LeakyReLU(), Linear(n_classes, n_classes)
        )

        self.fc_mean = Linear(
            self.encoder.architecture_params.linear_input_sizes[-1], latent_dim
        )

        self.fc_logvar = Linear(
            self.encoder.architecture_params.linear_input_sizes[-1], latent_dim
        )

        self.train_mse = CustomUnwrappingMSE()
        self.valid_mse = CustomUnwrappingMSE()
        self.test_mse = CustomUnwrappingMSE()

        # summary(self, input_size=((32, 1, seq_len), (32, 1)))
        summary(self, input_data=((torch.rand((32, 1, seq_len)), torch.rand((32, 1))),))

    def encode_mean_logvar(self, x: Tuple[torch.Tensor, torch.Tensor]):
        sig_data, sig_name = x
        e = self.encoder(sig_data)
        return self.fc_mean(e), self.fc_logvar(e)

    def decode(self, encoded):
        sig_decoded = self.signal_decoder(encoded)
        sig_logits = self.class_decoder(encoded)

        return sig_decoded, sig_logits

    def _loss_fn(self, x, reconstruction, mean, logvar):
        # NOTE: sig_name should be (batch_size,) vector of ints (0-11)
        # The ints refer to the class INDEX (e.g. 0 = lead I, 1 = lead II, etc.)
        sig_data, sig_name = x
        sig_decoded, sig_logits = reconstruction
        reproduction_loss = self.loss(sig_decoded, sig_data)
        classification_loss = F.cross_entropy(sig_logits, torch.squeeze(sig_name))

        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        return (
            reproduction_loss
            + KLD
            + (self.classification_loss_weight * classification_loss)
        )


if __name__ == "__main__":
    x = (
        torch.rand((32, 1, 1000)).to("cuda"),
        (torch.rand(32, 1) * 12).to(torch.long).to("cuda"),
    )

    m = SingleChannelCVAE(
        seq_len=1000,
        conv_depth=2,
        fc_depth=2,
        kernel_size=7,
        latent_dim=10,
        dropout=0.5,
        batchnorm=False,
    ).to("cuda")

    e = m.encoder(x[0])
    print(f"Encoder shape:\t {e.shape}")
    z = m.encode(x)
    print(f"Encoded shape:\t {z.shape}")

    mean, logvar = m.encode_mean_logvar(x)
    print(f"Mean shape:\t {mean.shape}")
    print(f"Logvar shape:\t {logvar.shape}")

    d = m.decode(z)

    print(f"Decoded shape:\t {d[0].shape, d[1].shape}")

    reconstruction, mean, logvar = m.forward(x)
    loss = m._loss_fn(x, reconstruction, mean, logvar)
    m.backward(loss)
    m.training_step((x, {}))
    m.validation_step((x, {}))

import torch
from torch.nn import (
    Conv1d,
    LeakyReLU,
    BatchNorm1d,
    MaxPool1d,
    Linear,
    ConvTranspose1d,
    Upsample,
    Sequential,
    Flatten,
    Unflatten,
    Dropout,
)

from torchinfo import summary
from dataclasses import dataclass
from typing import Optional


# For debugging sequentials
def print_sizes(model, input_tensor):
    output = input_tensor
    for m in model.children():
        output = m(output)
        print(m, output.shape)
    return output


@dataclass
class ConvolutionalEcgEncoderDecoderSharedParams:
    """
    Util class that allows various architecture params to be computed only once
    and shared between encoder and decoder

    Ensures symmetry of architecture between encoder and decoder
    """

    seq_len: int
    n_channels: int
    kernel_size: int
    conv_depth: int
    fc_depth: int
    latent_dim: int
    fc_scale_factor: int

    def __post_init__(self):
        self.input_padding_required = False

        if not (self.seq_len % (2**self.conv_depth) == 0):
            self.input_padding_required = True
            pad_amount = (2**self.conv_depth) - (self.seq_len % (2**self.conv_depth))

            print(
                f"Warning: seq_len {self.seq_len} not divisible by 2 ** {self.conv_depth}, will pad up to {self.seq_len + pad_amount}"
            )

            self.seq_len = self.seq_len + pad_amount
            self.left_pad = pad_amount // 2
            self.right_pad = pad_amount - self.left_pad

        self.layer_padding = (self.kernel_size - 1) // 2

        self.linear_input_sizes = [
            (self.seq_len // (2**self.conv_depth))
            * ((2**self.conv_depth) * self.n_channels)
        ]

        for idx in range(0, self.fc_depth - 1):
            next_layer_size = self.linear_input_sizes[-1] // self.fc_scale_factor

            if next_layer_size > self.latent_dim:
                self.linear_input_sizes.append(next_layer_size)
            else:
                self.linear_input_sizes.append(self.latent_dim)

    def build_tapered_encoding_fc_layer(self, layer_idx: int) -> Linear:
        return Linear(
            self.linear_input_sizes[layer_idx - 1], self.linear_input_sizes[layer_idx]
        )

    def build_tapered_decoding_fc_layer(self, layer_idx: int) -> Linear:
        return Linear(
            self.linear_input_sizes[layer_idx], self.linear_input_sizes[layer_idx - 1]
        )

    def get_conv_padding(self) -> int:
        return self.layer_padding


class ConvolutionalEcgEncoder(torch.nn.Module):

    def __init__(
        self,
        shared_params: ConvolutionalEcgEncoderDecoderSharedParams,
        batchnorm: bool = False,
        dropout: Optional[float] = None,
        include_final_layer: bool = False,
    ):
        super().__init__()

        self.architecture_params = shared_params
        self.batchnorm = batchnorm
        self.dropout = dropout

        layers = list()

        # Build Conv layers
        for idx in range(0, self.architecture_params.conv_depth):
            in_channels = shared_params.n_channels * (2**idx)

            layers += [
                Conv1d(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    kernel_size=shared_params.kernel_size,
                    stride=2,
                    padding=self.architecture_params.get_conv_padding(),
                ),
                LeakyReLU(),
            ]

            if self.batchnorm:
                layers.append(BatchNorm1d(num_features=(2 * in_channels)))

        layers.append(Flatten(start_dim=1, end_dim=-1))

        # Build FC layers
        for idx in range(1, self.architecture_params.fc_depth):
            layers += [
                self.architecture_params.build_tapered_encoding_fc_layer(idx),
                LeakyReLU(),
            ]

            if self.dropout:
                layers.append(Dropout(self.dropout))

        if include_final_layer:
            layers.append(
                Linear(
                    self.architecture_params.linear_input_sizes[-1],
                    self.architecture_params.latent_dim,
                )
            )

        self.net = Sequential(*layers)

    def forward(self, x):
        if self.architecture_params.input_padding_required:
            x = torch.nn.functional.pad(
                x,
                pad=(
                    self.architecture_params.left_pad,
                    self.architecture_params.right_pad,
                ),
                value=0.0,
            )

        return self.net(x)


class ConvolutionalEcgDecoder(torch.nn.Module):

    def __init__(
        self,
        shared_params: ConvolutionalEcgEncoderDecoderSharedParams,
        batchnorm: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()

        self.batchnorm = batchnorm
        self.dropout = dropout
        self.padding_required = False

        self.architecture_params = shared_params

        layers = list()
        layers += [
            Linear(
                self.architecture_params.latent_dim,
                self.architecture_params.linear_input_sizes[-1],
            ),
            LeakyReLU(),
        ]

        for idx in range(self.architecture_params.fc_depth - 1, 0, -1):
            layers += [
                self.architecture_params.build_tapered_decoding_fc_layer(idx),
                LeakyReLU(),
            ]

            if self.dropout:
                layers.append(Dropout(self.dropout))

        layers.append(
            Unflatten(
                dim=1,
                unflattened_size=(
                    self.architecture_params.n_channels
                    * (2**self.architecture_params.conv_depth),
                    self.architecture_params.seq_len
                    // (2**self.architecture_params.conv_depth),
                ),
            ),
        )

        for idx in range(self.architecture_params.conv_depth, 0, -1):
            in_channels = self.architecture_params.n_channels * (2**idx)

            layers += [
                ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=in_channels // 2,
                    kernel_size=self.architecture_params.kernel_size,
                    padding=self.architecture_params.get_conv_padding(),
                    stride=2,
                    output_padding=1,
                ),
                LeakyReLU(),
            ]

            if self.batchnorm:
                layers.append(BatchNorm1d(num_features=in_channels // 2))

        self.net = Sequential(*layers)

    def forward(self, x):
        out = self.net(x)

        if self.architecture_params.input_padding_required:
            return out[
                :,
                :,
                self.architecture_params.left_pad : -self.architecture_params.right_pad,
            ].contiguous()
        else:
            return out


if __name__ == "__main__":

    sp = ConvolutionalEcgEncoderDecoderSharedParams(
        seq_len=1000,
        n_channels=12,
        kernel_size=7,
        conv_depth=5,
        fc_depth=3,
        latent_dim=100,
        fc_scale_factor=4,
    )

    encoder = ConvolutionalEcgEncoder(
        shared_params=sp,
        include_final_layer=True,
    ).to("cuda")

    decoder = ConvolutionalEcgDecoder(shared_params=sp).to("cuda")

    summary(encoder, input_size=(8, 12, 1000))

    print()

    summary(decoder, input_size=(8, 100))

    x = torch.randn((100, 12, 1000)).to("cuda")
    print(f"X:\t\t{x.shape}")
    e = encoder(x)
    print(f"E:\t\t{e.shape}")
    d = decoder(e)
    print(f"D:\t\t{d.shape}")

import torch
import math
from torch.nn import BCELoss, Linear, Sequential, Flatten, ReLU, Sigmoid, MSELoss
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    AUROC,
    AveragePrecision,
    MultilabelAUROC,
    MultilabelAveragePrecision,
)
from torchmetrics.regression.mse import MeanSquaredError
from torchinfo import summary
from torch.nn import AdaptiveAvgPool1d
from typing import Literal, Optional
import lightning as L
from abc import ABC, abstractmethod


class NanTolerantMSELoss(MSELoss):
    """
    Accepts nans in target tensor

    - Identify indices of nans in target
    - Replace all values in input and target at nan indices with 0
    - Store count of all nan indices
    - Compute loss as usual, with sum reduction
    - Divide loss by number of non-nan elements
    """

    def __init__(self):
        super().__init__(reduction="sum")

    def forward(self, input, target):
        nanmask = torch.isnan(target)
        target[nanmask] = 0.0
        input[nanmask] = 0.0

        sum_reduction_loss = super().forward(input, target)

        return sum_reduction_loss / torch.sum(nanmask)


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, max_len, dropout=0.1, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer(
            "pe", pe
        )  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[: x.size(0), :]  # type: ignore
        return self.dropout(x)


class ConvEmbeddingTST(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int,
        embedding_kernel: int,
        nhead: int,
        nlayers: int,
    ):
        super(ConvEmbeddingTST, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        # TODO: make this strided if model gets too big (would have to adjust positional encoding max_len as well)
        self.conv_embedding = torch.nn.Conv1d(
            12, self.d_model, kernel_size=embedding_kernel
        )
        self.positional_encoding = FixedPositionalEncoding(
            self.d_model, max_len=self.max_len
        )

        self.backbone = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=nhead,
                batch_first=True,
            ),
            num_layers=nlayers,
        )

        self.output_layer = torch.nn.ConvTranspose1d(
            self.d_model, 12, kernel_size=embedding_kernel
        )

    def encode(self, x):
        x_embeded = self.conv_embedding(x)
        # Pytorch transformer convention features dimension last, but convolutional convention is channels first
        x_embeded = x_embeded.permute(0, 2, 1)  # [batch_dim, seq_len, feat_dim]
        inp = self.positional_encoding(x_embeded)

        encoded = self.backbone(
            src=inp,
            # TODO: for padding masks when support for variable-length sequences is implemented
            # src_key_padding_mask=mask,
        )

        return encoded.permute(0, 2, 1)

    def forward(self, x_masked):
        encoded = self.encode(x_masked)
        reconstruction = self.output_layer(encoded)

        return reconstruction


class BaseTransformerLM(L.LightningModule, ABC):
    def __init__(
        self,
        lr: float,
        loss: torch.nn.Module,
        train_metrics: MetricCollection,
        valid_metrics: MetricCollection,
    ):
        super(BaseTransformerLM, self).__init__()
        self.lr = lr
        self.loss = loss
        self.train_metrics = train_metrics
        self.valid_metrics = valid_metrics
        self.save_hyperparameters()

    @abstractmethod
    def _run_model(self, batch) -> tuple[torch.Tensor, torch.Tensor]: ...

    def _load_base_with_options(
        self,
        path: str,
        options: Optional[Literal["freeze", "reset"]] = None,
    ) -> ConvEmbeddingTST:
        base_transformer_lm = ImputationTransformerLM.load_from_checkpoint(path)

        if options == "reset":
            print("Resetting model to un-pretrained state")
            old_tst = base_transformer_lm.tst
            # TODO: this depends a lot on current tst architecture and will break if architecture changes too much
            # But it's the most reliable way to make sure the model is fresh
            old_args = None

            try:

                old_args = dict(
                    d_model=old_tst.d_model,
                    max_len=old_tst.max_len,
                    embedding_kernel=old_tst.conv_embedding.kernel_size[0],
                    nhead=old_tst.backbone.layers[0].self_attn.num_heads,
                    nlayers=old_tst.backbone.num_layers,
                )

                tst = ConvEmbeddingTST(**old_args)  # type: ignore
            except Exception as e:
                print(
                    "There was an error re-initializing a new TST with old arguments. It is very possible there was an architecture change that was not properly refactored here"
                )

                if old_args:
                    print("Tried initializing with the following:")
                    for k, v in old_args:
                        print(f"{k} ==> {v}")

                raise e

        elif options == "freeze":
            print("Freezing base transformer")
            tst = base_transformer_lm.tst

            for param in tst.parameters():
                param.requires_grad = False

            tst.eval()
        else:
            tst = base_transformer_lm.tst

        return tst

    def _do_step(
        self, batch, stage: Literal["train", "val", "test"], metrics: MetricCollection
    ):
        expected_output, actual_output = self._run_model(batch)

        loss = self.loss(actual_output, expected_output.float())

        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )

        for metric_name, metric_obj in metrics.items():
            metric_obj.update(actual_output, expected_output)
            self.log(
                f"{stage}_{metric_name}",
                metric_obj,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return loss

    def training_step(self, batch):
        return self._do_step(batch, stage="train", metrics=self.train_metrics)

    def validation_step(self, batch):
        return self._do_step(batch, stage="val", metrics=self.valid_metrics)

    def test_step(self, batch): ...

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class ImputationTransformerLM(BaseTransformerLM):

    def __init__(self, lr: float, tst: ConvEmbeddingTST):
        super(ImputationTransformerLM, self).__init__(
            lr=lr,
            loss=MSELoss(reduction="sum"),
            train_metrics=MetricCollection([MeanSquaredError()]),
            valid_metrics=MetricCollection([MeanSquaredError()]),
        )

        self.tst = tst

    def _run_model(self, batch):
        x, x_masked, mask, meta = batch

        reconstruction = self.tst(x_masked)
        reconstruction_masks_only = reconstruction * mask
        x_masks_only = x * mask

        return x_masks_only, reconstruction_masks_only


class ClassificationTransformerLM(BaseTransformerLM):

    def __init__(
        self,
        lr: float,
        pretrained_path: str,
        n_labels: int,
        base_options: Optional[Literal["freeze", "reset"]] = None,
    ):
        super(ClassificationTransformerLM, self).__init__(
            lr=lr,
            loss=BCELoss(),
            train_metrics=MetricCollection(
                [
                    MultilabelAUROC(num_labels=n_labels),
                    MultilabelAveragePrecision(num_labels=n_labels),
                ]
            ),
            valid_metrics=MetricCollection(
                [
                    MultilabelAUROC(num_labels=n_labels),
                    MultilabelAveragePrecision(num_labels=n_labels),
                ]
            ),
        )

        self.base_transformer = self._load_base_with_options(
            pretrained_path, base_options
        )

        self.classification_head = Sequential(
            # ReLU(),
            AdaptiveAvgPool1d(1),
            Flatten(),
            Linear(
                self.base_transformer.d_model,
                n_labels,
            ),
            Sigmoid(),
        )

    def _run_model(self, batch):
        x, y = batch
        z = self.base_transformer.encode(x)
        preds = self.classification_head(z)
        return y, preds


class RegressionTransformerLM(BaseTransformerLM):

    def __init__(
        self,
        lr: float,
        pretrained_path: str,
        n_outputs: int,
        base_options: Optional[Literal["freeze", "reset"]] = None,
    ):
        super(RegressionTransformerLM, self).__init__(
            lr=lr,
            loss=NanTolerantMSELoss(),
            train_metrics=MetricCollection([MeanSquaredError()]),
            valid_metrics=MetricCollection([MeanSquaredError()]),
        )

        self.lr = lr
        self.base_transformer = self._load_base_with_options(
            pretrained_path, base_options
        )
        self.regression_head = Sequential(
            # ReLU(),
            AdaptiveAvgPool1d(1),
            Flatten(),
            Linear(
                self.base_transformer.d_model,
                n_outputs,
            ),
        )

    def _run_model(self, batch):
        x, y = batch
        z = self.base_transformer.encode(x)
        preds = self.regression_head(z)

        # TODO: nan masking everything to zero results in optimistic loss for high-missingness examples
        # this may be suboptimal. Maybe better to calculate loss only for examples that were present

        return y, preds


if __name__ == "__main__":
    sample_ekg = torch.rand(32, 12, 1000)

    tst = ConvEmbeddingTST(
        d_model=64, max_len=1000, nhead=4, nlayers=3, embedding_kernel=7
    )

    out = tst.encode(sample_ekg)

    print(out.shape)

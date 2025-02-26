from ptbxlae.modeling import BaseModel
import torch
import math
from torch.nn import BCELoss, Linear, Sequential, Flatten, ReLU, Sigmoid, MSELoss
from torchmetrics import MetricCollection
from torchmetrics.classification import AUROC, AveragePrecision
from torchmetrics.regression.mse import MeanSquaredError
from torchmetrics.metric import Metric
from torchinfo import summary
from torch.nn import AdaptiveAvgPool1d, ModuleList
from typing import Optional, Literal, Iterable
import lightning as L
from abc import ABC, abstractmethod


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

        self.model = torch.nn.TransformerEncoder(
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

        encoded = self.model(
            src=inp,
            # TODO: for padding masks when support for variable-length sequences is implemented
            # src_key_padding_mask=mask,
        )

        return encoded.permute(0, 2, 1)

    def forward(self, x_masked):
        encoded = self.encode(x_masked)
        reconstruction = self.output_layer(encoded)

        return reconstruction


class BaseTransformerLM(L.LightningModule):
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

    @abstractmethod
    def _run_model(self, batch) -> tuple[torch.Tensor, torch.Tensor]: ...

    def _do_step(
        self, batch, stage: Literal["train", "val", "test"], metrics: MetricCollection
    ):
        expected_output, actual_output = self._run_model(batch)

        loss = self.loss(actual_output, expected_output)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

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
    ):
        super(ClassificationTransformerLM, self).__init__(
            lr=lr,
            loss=BCELoss(),
            train_metrics=MetricCollection(
                [
                    AUROC(task="multilabel", num_labels=n_labels),
                    AveragePrecision(task="multilabel", num_labels=n_labels),
                ]
            ),
            valid_metrics=MetricCollection(
                [
                    AUROC(task="multilabel", num_labels=n_labels),
                    AveragePrecision(task="multilabel", num_labels=n_labels),
                ]
            ),
        )

        self.base_transformer = ImputationTransformerLM.load_from_checkpoint(
            pretrained_path
        ).tst

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
        z = self.base_transformer(x)
        preds = self.classification_head(z)
        return y, preds


class RegressionTransformerLM(BaseTransformerLM):

    def __init__(
        self,
        lr: float,
        base_transformer: ConvEmbeddingTST,
        regression_head: torch.nn.Module,
        n_outputs: int,
    ):
        super(RegressionTransformerLM, self).__init__(
            lr=lr,
            loss=MSELoss(reduction="sum"),
            train_metrics=MetricCollection([MeanSquaredError()]),
            valid_metrics=MetricCollection([MeanSquaredError()]),
        )

        self.lr = lr
        self.base_transformer = base_transformer
        self.regression_head = regression_head

        self.classification_head = Sequential(
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
        z = self.base_transformer(x)
        preds = self.regression_head(z)

        return y, preds


if __name__ == "__main__":
    sample_ekg = torch.rand(32, 12, 1000)

    tst = ConvEmbeddingTST(
        d_model=64, max_len=1000, nhead=4, nlayers=3, embedding_kernel=7
    )

    out = tst.encode(sample_ekg)

    print(out.shape)

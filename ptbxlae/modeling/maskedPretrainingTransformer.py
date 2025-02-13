from ptbxlae.modeling import BaseModel
import torch
import math
from torch.nn import BCELoss, Linear, Sequential, Flatten, ReLU, Sigmoid
from torchmetrics.classification import AUROC, AveragePrecision


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


class MaskedPretrainingTransformer(BaseModel):

    def __init__(
        self,
        lr: float,
        max_len: int,
        d_model: int,
        nhead: int,
        nlayers: int,
        embedding_kernel: int,
        loss=None,
        base_model_path=None,
    ):
        super(MaskedPretrainingTransformer, self).__init__(lr, loss, base_model_path)

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

    def training_step(self, batch):
        x, x_masked, mask, _ = batch

        reconstruction = self.forward(x_masked)

        reconstruction_masks_only = reconstruction * mask
        x_masks_only = x * mask

        loss = self.loss(
            reconstruction_masks_only,
            x_masks_only,
        )
        self.train_mse.update(
            reconstruction_masks_only,
            x_masks_only,
        )

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
        x, x_masked, mask, _ = batch

        reconstruction = self.forward(x_masked)

        reconstruction_masks_only = reconstruction * mask
        x_masks_only = x * mask

        loss = self.loss(
            reconstruction_masks_only,
            x_masks_only,
        )
        self.valid_mse.update(
            reconstruction_masks_only,
            x_masks_only,
        )

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
        raise NotImplementedError()


class MaskedPretrainingTransformerClassifier(BaseModel):

    def __init__(
        self,
        lr,
        pretrained_path: str,
        freeze_base_model: bool,
        n_classes: int,
    ):
        super(MaskedPretrainingTransformerClassifier, self).__init__(lr, BCELoss())

        self.train_auroc = AUROC(task="multilabel", num_labels=n_classes)
        self.train_auprc = AveragePrecision(task="multilabel", num_labels=n_classes)
        self.valid_auroc = AUROC(task="multilabel", num_labels=n_classes)
        self.valid_auprc = AveragePrecision(task="multilabel", num_labels=n_classes)

        self.train_metrics = [self.train_auroc, self.train_auprc]
        self.valid_metrics = [self.valid_auroc, self.valid_auprc]

        self.freeze_base_model = freeze_base_model

        self.pretrained_transformer = MaskedPretrainingTransformer.load_from_checkpoint(
            pretrained_path
        ).to(self.device)

        _sample = torch.rand((2, 12, self.pretrained_transformer.max_len)).to(
            self.device
        )

        with torch.no_grad():
            _sample_encoded = self.pretrained_transformer.encode(_sample)
            expected_encoding_shape = _sample_encoded.shape

        print(
            f"Encoder outputs ({expected_encoding_shape[1]}, {expected_encoding_shape[2]})"
        )

        if self.freeze_base_model:
            for param in self.pretrained_transformer.parameters():
                param.requires_grad = False

        self.classification_head = Sequential(
            Flatten(),
            ReLU(),
            Linear(
                expected_encoding_shape[1] * expected_encoding_shape[2],
                n_classes,
            ),
            Sigmoid(),
        )

    def forward(self, x):
        encoded = self.pretrained_transformer.encode(x)
        return self.classification_head(encoded)

    def training_step(self, batch):
        x, y = batch

        preds = self.forward(x)

        loss = self.loss(preds, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        for m in self.train_metrics:
            m.update(preds, y.int())
            self.log(
                m.__class__.__name__,
                m,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        return loss

    def validation_step(self, batch):
        x, y = batch

        preds = self.forward(x)

        loss = self.loss(preds, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        for m in self.valid_metrics:
            m.update(preds, y.int())
            self.log(
                m.__class__.__name__,
                m,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        return loss

    def test_step(self, batch):
        raise NotImplementedError()

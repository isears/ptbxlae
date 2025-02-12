from ptbxlae.modeling import BaseModel
import torch
import math


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

    def __init__(self, lr: float, max_len: int, loss=None, base_model_path=None):
        super(MaskedPretrainingTransformer, self).__init__(lr, loss, base_model_path)

        # TODO: eventually pass all these as args
        self.d_model = 512

        # TODO: make this strided if model gets too big (would have to adjust positional encoding max_len as well)
        self.conv_embedding = torch.nn.Conv1d(12, self.d_model, kernel_size=7)
        self.positional_encoding = FixedPositionalEncoding(
            self.d_model, max_len=max_len
        )

        self.model = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
            ),
            num_layers=6,
        )

        self.output_layer = torch.nn.ConvTranspose1d(self.d_model, 12, kernel_size=7)

    def forward(self, x_masked):
        # Pytorch transformer convention [seq_length, batch_size, feat_dim]
        x_masked = x_masked.permute(0, 2, 1)
        x_embeded = self.conv_embedding(x_masked)
        inp = self.positional_encoding(x_embeded)

        reconstruction_embedded = self.model(
            src=inp,
            # TODO: for padding masks when support for variable-length sequences is implemented
            # src_key_padding_mask=mask,
        )

        reconstruction = self.output_layer(reconstruction_embedded)

        return reconstruction.permute(1, 0, 2)

    def training_step(self, batch):
        x, x_masked, mask, _ = batch

        reconstruction = self.forward(x_masked)
        loss = self.loss(
            reconstruction,
            x.contiguous(),
        )
        self.train_mse.update(
            reconstruction,
            x.contiguous(),
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
        loss = self.loss(
            reconstruction,
            x.contiguous(),
        )
        self.valid_mse.update(
            reconstruction,
            x.contiguous(),
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

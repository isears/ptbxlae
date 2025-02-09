from ptbxlae.modeling import BaseModel
import torch


class MaskedPretrainingTransformer(BaseModel):

    def __init__(self, lr: float, loss=None, base_model_path=None):
        super(MaskedPretrainingTransformer, self).__init__(lr, loss, base_model_path)

        # TODO: eventually pass all these as args
        self.model = torch.nn.Transformer(
            d_model=12,
            nhead=3,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            batch_first=True,
        )

    def forward(self, x, x_masked, attn_mask):

        reconstruction = self.model(
            src=x_masked,
            tgt=x,
            # TODO: need to think more about how to properly implement this, but for now channel masking strategy doesn't need
            # src_key_padding_mask=attn_mask,
        )

        return reconstruction

    def training_step(self, batch):
        x, x_masked, attn_mask, _ = batch

        # Pytorch convention seq_len before features
        # TODO: there's probably a less verbose way to handle all the .unsqeeze() and .permute() ops
        x = x.permute(0, 2, 1)
        x_masked = x_masked.permute(0, 2, 1)

        reconstruction = self.forward(x, x_masked, attn_mask)
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
        x, x_masked, attn_mask, _ = batch

        # Pytorch convention seq_len before features
        x = x.permute(0, 2, 1)
        x_masked = x_masked.permute(0, 2, 1)

        reconstruction = self.forward(x, x_masked, attn_mask)
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

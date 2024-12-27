from ptbxlae.modeling import BaseVAE
from lightning.pytorch.cli import LightningCLI
from ptbxlae.dataprocessing.dataModules import BaseDM
import torch


class BaseModelLoadingCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--base_ckpt_path",
            help="Path to load a pretrained checkpoint before .fit()",
            type=str,
        )


def cli_main():
    cli = BaseModelLoadingCLI(
        BaseVAE,
        BaseDM,
        subclass_mode_data=True,
        subclass_mode_model=True,
        save_config_callback=None,
        run=False,
    )

    # Need to load weights only because otherwise statedict of other objects e.g. early stopping checkpoint also get loaded
    # https://github.com/Lightning-AI/pytorch-lightning/issues/15705

    if cli.config.base_ckpt_path:
        print(f"Loading weights from {cli.config.base_ckpt_path}")
        weights = torch.load(cli.config.base_ckpt_path, map_location=cli.model.device)
        cli.model.load_state_dict(weights["state_dict"])

    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule)


if __name__ == "__main__":
    cli_main()

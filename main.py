from ptbxlae.modeling import BaseVAE
from lightning.pytorch.cli import LightningCLI
from ptbxlae.dataprocessing.dataModules import BaseDM
import torch


def cli_main():
    cli = LightningCLI(
        BaseVAE,
        BaseDM,
        subclass_mode_data=True,
        subclass_mode_model=True,
        save_config_callback=None,
        parser_kwargs={"default_config_files": ["configs/default_trainer.yaml"]},
        run=False,
    )

    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule, ckpt_path="best")


if __name__ == "__main__":
    cli_main()

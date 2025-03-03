from ptbxlae.modeling import BaseModel
from lightning.pytorch.cli import LightningCLI
from ptbxlae.dataprocessing.dataModules import BaseDM
import torch


def cli_main():
    # torch.set_float32_matmul_precision('medium' | 'high')
    torch.set_float32_matmul_precision("medium")
    cli = LightningCLI(
        BaseModel,
        BaseDM,
        subclass_mode_data=True,
        subclass_mode_model=True,
        save_config_callback=None,
        parser_kwargs={"default_config_files": ["configs/default_trainer.yaml"]},
        # run=False,
    )

    # cli.trainer.fit(cli.model, cli.datamodule)
    # cli.trainer.test(cli.model, cli.datamodule, ckpt_path="best")
    # cli.trainer.test(cli.model, cli.datamodule)


if __name__ == "__main__":
    cli_main()

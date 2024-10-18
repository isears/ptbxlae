from ptbxlae.modeling import BaseVAE
from lightning.pytorch.cli import LightningCLI
from ptbxlae.dataprocessing.dataModules import BaseDM


def cli_main():
    cli = LightningCLI(
        BaseVAE, BaseDM, subclass_mode_data=True, subclass_mode_model=True
    )


if __name__ == "__main__":
    cli_main()

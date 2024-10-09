from ptbxlae.modeling.resnet1dVAE import Resnet1dVAE
from lightning.pytorch.cli import LightningCLI
from ptbxlae.dataprocessing.dataModules import PtbxlCleanDM


def cli_main():
    cli = LightningCLI(Resnet1dVAE, PtbxlCleanDM)


if __name__ == "__main__":
    cli_main()

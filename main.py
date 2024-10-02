from ptbxlae.modeling.simpleStaticVAE import StaticSimpleVAE
from lightning.pytorch.cli import LightningCLI
from ptbxlae.dataprocessing.ptbxlDS import PtbxlDM


def cli_main():
    cli = LightningCLI(StaticSimpleVAE, PtbxlDM)


if __name__ == "__main__":
    cli_main()

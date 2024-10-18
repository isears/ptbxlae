from ptbxlae.modeling.singleCycleConv import SingleCycleConvVAE
from ptbxlae.modeling.multiCycleConv import MultiCycleConv
from lightning.pytorch.cli import LightningCLI
from ptbxlae.dataprocessing.dataModules import PtbxlCleanDM


def cli_main():
    cli = LightningCLI(MultiCycleConv, PtbxlCleanDM)


if __name__ == "__main__":
    cli_main()

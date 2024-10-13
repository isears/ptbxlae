from ptbxlae.modeling.singleCycleConv import SingleCycleConvVAE
from lightning.pytorch.cli import LightningCLI
from ptbxlae.dataprocessing.dataModules import SingleCycleCachedDM


def cli_main():
    cli = LightningCLI(SingleCycleConvVAE, SingleCycleCachedDM)


if __name__ == "__main__":
    cli_main()

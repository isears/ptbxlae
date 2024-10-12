from ptbxlae.modeling.feedForwardVAE import SingleChannelFFNNVAE
from lightning.pytorch.cli import LightningCLI
from ptbxlae.dataprocessing.dataModules import PtbxlSingleCycleDM


def cli_main():
    cli = LightningCLI(SingleChannelFFNNVAE, PtbxlSingleCycleDM)


if __name__ == "__main__":
    cli_main()

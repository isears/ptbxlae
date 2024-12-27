from torchmetrics import Metric
import torch
import pandas as pd

# TODO: must handle NONE labels (return na)


class LatentRepresentationUtilityMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, latent_representations: torch.Tensor, labels: pd.DataFrame):
        raise NotImplementedError()

    def compute(self) -> torch.Tensor:
        raise NotImplementedError()

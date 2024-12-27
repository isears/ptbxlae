from torchmetrics import Metric
import torch
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score


class LatentRepresentationUtilityMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latent_representations_list = list()
        self.labels = dict()

    def update(self, latent_representations: torch.Tensor, labels: dict) -> None:
        self.latent_representations_list.append(latent_representations)

        if len(self.labels) == 0:
            self.labels = {k: [v] for k, v in labels.items()}
        else:
            # TODO: could add an assert here to make sure labels are consistent across batches
            for k, v in labels.items():
                self.labels[k].append(v)

    def compute(self) -> torch.Tensor:
        # TODO: log 2D scatter plot of latent representations with labels
        if len(self.labels) == 0:
            return float("nan")

        ret = {}
        collected_latent_representations = torch.cat(
            self.latent_representations_list
        ).cpu()
        collected_labels = {k: torch.cat(v).cpu() for k, v in self.labels.items()}

        for label_name, label_values in collected_labels.items():
            clf = SVC(kernel="linear")
            clf.fit(collected_latent_representations, label_values)

            score = roc_auc_score(
                label_values,
                clf.decision_function(collected_latent_representations),
            )

            ret[label_name] = score

        return ret

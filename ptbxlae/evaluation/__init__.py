from torchmetrics import Metric
import torch
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


class LatentRepresentationUtilityMetric(Metric):
    full_state_update: bool = False  # type: ignore

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

    def compute(self) -> dict:
        # TODO: log 2D scatter plot of latent representations with labels
        if len(self.labels) == 0:
            return {}

        ret = {}
        collected_latent_representations = torch.cat(
            self.latent_representations_list
        ).cpu()
        collected_labels = {k: torch.cat(v).cpu() for k, v in self.labels.items()}

        print("Computing latent representation utility scores...")
        for label_name, label_values in tqdm(collected_labels.items()):
            # clf = SVC(kernel="linear")
            # clf.fit(collected_latent_representations.numpy(), label_values)

            # score = roc_auc_score(
            #     label_values,
            #     clf.decision_function(collected_latent_representations.numpy()),
            # )

            clf = LogisticRegression(max_iter=1000)
            # clf = RandomForestClassifier()
            score = cross_val_score(
                clf,
                collected_latent_representations.numpy(),
                label_values,
                scoring="roc_auc",
                n_jobs=-1,
            ).mean()

            ret[label_name] = score

        return ret

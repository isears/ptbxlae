import lightning as L
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from ptbxlae.dataprocessing.ptbxlDS import *
from ptbxlae.dataprocessing.cachedDS import *
from ptbxlae.dataprocessing.nkSyntheticDS import *
from ptbxlae.dataprocessing.mimicDS import *
from typing import Type, Optional


class BaseDM(L.LightningDataModule):

    def __init__(self, batch_size: int = 32, workers: Optional[int] = None):
        super().__init__()

        self.batch_size = batch_size

        if workers:
            self.cores_available = workers
        else:
            self.cores_available = len(os.sched_getaffinity(0))

        print(f"Initializing DM with {self.cores_available} workers")

    def setup(self, stage: str):
        raise NotImplementedError("Base DM not subclassed!")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, num_workers=self.cores_available, batch_size=self.batch_size
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds, num_workers=self.cores_available, batch_size=self.batch_size
        )

    def test_dataloader(self):
        # If the dataset (i.e. ptbxl) has computationally expensive labels to return, we only want to do that during the test phase
        if hasattr(self.test_ds.dataset, "set_return_labels") and callable(
            self.test_ds.dataset.set_return_labels
        ):
            self.test_ds.dataset.set_return_labels(True)

        return DataLoader(
            self.test_ds, num_workers=self.cores_available, batch_size=self.batch_size
        )


class DefaultDM(BaseDM):
    def __init__(
        self,
        ds: Dataset,
        train_valid_test_splits: tuple[float, float, float] = (0.8, 0.1, 0.1),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_valid_test_splits = train_valid_test_splits
        self.core_ds = ds

    def setup(self, stage: str):
        self.train_ds, self.valid_ds, self.test_ds = random_split(
            self.core_ds,
            [0.8, 0.1, 0.1],
            generator=torch.Generator().manual_seed(42),
        )


class MimicTrainPtbxlTestDM(BaseDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self, stage: str):
        mimic_ds = MimicDS()

        self.train_ds, self.valid_ds = random_split(
            mimic_ds,
            [0.9, 0.1],
            generator=torch.Generator().manual_seed(42),
        )

    def test_dataloader(self):
        ptbxl_ds = PtbxlDS(return_labels=True, lowres=True)
        return DataLoader(
            ptbxl_ds, num_workers=self.cores_available, batch_size=self.batch_size
        )

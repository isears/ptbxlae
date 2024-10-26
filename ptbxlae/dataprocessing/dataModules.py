import lightning as L
from torch.utils.data import DataLoader, random_split
import torch
from ptbxlae.dataprocessing.ptbxlDS import (
    PtbxlDS,
    PtbxlCleanDS,
    PtbxlSingleCycleDS,
    PtbxlSigWithRpeaksDS,
)
from ptbxlae.dataprocessing.cachedDS import SingleCycleCachedDS


class BaseDM(L.LightningDataModule):
    def __init__(self, root_folder: str = "./data", batch_size: int = 32):
        super().__init__()
        self.root_folder = root_folder
        self.batch_size = batch_size

    def _get_ds(self):
        raise NotImplementedError(
            "Base DM called, need to call a subclass that implements _get_ds()"
        )

    def setup(self, stage: str):
        ds = self._get_ds()

        self.train_ds, self.valid_ds, self.test_ds = random_split(
            ds, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, num_workers=16, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, num_workers=16, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, num_workers=16, batch_size=self.batch_size)


class PtbxlDM(BaseDM):
    def _get_ds(self):
        return PtbxlDS(root_folder=self.root_folder, lowres=False)


class PtbxlCleanDM(BaseDM):
    def _get_ds(self):
        return PtbxlCleanDS(root_folder=self.root_folder, lowres=False)


class PtbxlSingleCycleDM(BaseDM):
    def _get_ds(self):
        return PtbxlSingleCycleDS(root_folder=self.root_folder)


class SingleCycleCachedDM(BaseDM):
    def __init__(
        self, cache_folder: str = "./cache/singlecycle_data", batch_size: int = 32
    ):
        super().__init__()
        self.cache_folder = cache_folder
        self.batch_size = batch_size

    def _get_ds(self):
        return SingleCycleCachedDS(cache_path=self.cache_folder)


class PtbxlSigWithRpeaksDM(BaseDM):
    def __init__(
        self,
        root_folder: str = "./data",
        batch_size: int = 32,
        smoothing=False,
        stacked=False,
    ):
        super().__init__(root_folder, batch_size)
        self.smoothing = smoothing
        self.stacked = stacked

    def _get_ds(self):
        return PtbxlSigWithRpeaksDS(
            root_folder=self.root_folder, smoothing=self.smoothing, stacked=self.stacked
        )


def load_testset_to_mem(root_folder: str = "./data"):
    dm = PtbxlDM(root_folder=root_folder)
    dm.setup(stage="test")
    dl = dm.test_dataloader()

    all_batches = [x for x in dl]
    return torch.cat(all_batches)

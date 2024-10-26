import torch


class NkSyntheticDS(torch.utils.data.Dataset):

    def __init__(self, examples_per_epoch: int = 1e6):
        super().__init__()

        self.examples_per_epoch = examples_per_epoch

    def __len__(self):
        return self.examples_per_epoch

    def __getitem__(self, index: int):
        pass

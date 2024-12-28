import torch
import pandas as pd
import random
import wfdb
import neurokit2 as nk
import numpy as np


class MimicDS(torch.utils.data.Dataset):
    def __init__(
        self,
        root_folder: str = "./data/mimiciv-ecg",
        return_labels: bool = False,
        freq: int = 100,
    ):
        super().__init__()
        self.root_folder = root_folder
        self.return_labels = return_labels
        self.record_list = (
            pd.read_csv(f"{root_folder}/record_list.csv")
            .groupby("subject_id")
            .agg(list)
        )

        self.freq = freq

        random.seed(42)

    def __len__(self):
        return len(self.record_list)

    def __getitem__(self, idx):
        possible_ecgs = self.record_list.iloc[idx]["path"]
        path = f"{self.root_folder}/{random.choice(possible_ecgs)}"

        sig, meta = wfdb.rdsamp(path)

        assert meta["sig_name"] == [
            "I",
            "II",
            "III",
            "aVR",
            "aVF",
            "aVL",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]

        sig_resamp = np.apply_along_axis(
            nk.signal_resample,
            1,
            sig.transpose(),
            sampling_rate=meta["fs"],
            desired_sampling_rate=self.freq,
        )

        assert sig_resamp.shape == (12, 10 * self.freq)

        sig_clean = np.apply_along_axis(
            nk.ecg_clean, 1, sig_resamp, sampling_rate=self.freq
        )

        return torch.Tensor(sig_clean).float(), {}


if __name__ == "__main__":
    ds = MimicDS()

    sig, labels = ds[0]

    print(sig.shape)

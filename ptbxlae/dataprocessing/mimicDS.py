import torch
import pandas as pd
import random
import wfdb
import neurokit2 as nk
import numpy as np
from typing import Optional


class MimicDS(torch.utils.data.Dataset):
    def __init__(
        self,
        root_folder: str = "./data/mimiciv-ecg",
        return_labels: bool = False,
        freq: int = 100,
        study_ids: Optional[list] = None,
    ):
        super().__init__()
        self.root_folder = root_folder
        self.return_labels = return_labels

        records = pd.read_csv(f"{root_folder}/record_list.csv")

        if study_ids is not None:
            records = records[records["study_id"].apply(lambda x: int(x) in study_ids)]

        self.record_list = records.groupby("subject_id").agg(list)
        self.freq = freq

        random.seed(42)

    def __len__(self):
        return len(self.record_list)

    def __getitem__(self, index: int):
        possible_ecgs = self.record_list.iloc[index]["path"]
        path = f"{self.root_folder}/{random.choice(possible_ecgs)}"

        sig, meta = wfdb.rdsamp(path)

        sig = sig.transpose()

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

        # Sometimes sig contains nans, unfortunately
        # Addressing this with linear interpolation, as it seems to be a small number in most cases
        if np.isnan(sig).any():
            # print(
            #     f"[WARN] Found {np.isnan(sig).sum()} nans in {path}, interpolating..."
            # )
            for i in range(0, len(meta["sig_name"])):
                nans = np.isnan(sig[i, :])
                sig[i, nans] = np.interp(
                    nans.nonzero()[0], (~nans).nonzero()[0], sig[i, ~nans]
                )

        sig_resamp = np.apply_along_axis(
            nk.signal_resample,  # type: ignore
            1,
            sig,
            sampling_rate=meta["fs"],
            desired_sampling_rate=self.freq,
        )  # type: ignore

        assert sig_resamp.shape == (12, 10 * self.freq)

        sig_clean = np.apply_along_axis(
            nk.ecg_clean, 1, sig_resamp, sampling_rate=self.freq  # type: ignore
        )  # type: ignore

        return torch.Tensor(sig_clean).float(), {}


if __name__ == "__main__":
    ds = MimicDS()

    sig, labels = ds[0]

    print(sig.shape)

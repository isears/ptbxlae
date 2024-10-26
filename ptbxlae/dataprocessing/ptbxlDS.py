import torch
import pandas as pd
import os
from ptbxlae.dataprocessing import load_single_record
import neurokit2 as nk
import numpy as np


class PtbxlDS(torch.utils.data.Dataset):
    def __init__(self, root_folder: str = "./data", lowres=False):
        super().__init__()

        # If not filtered already, do filtering
        if os.path.isfile(f"{root_folder}/ptbxl_database_filtered.csv"):
            metadata = pd.read_csv(f"{root_folder}/ptbxl_database_filtered.csv")
            metadata.patient_id = metadata.patient_id.astype(int)
        # Filtering: only take one EKG per patient, and attempt to find a "clean" EKG for each patient
        else:
            metadata = pd.read_csv(f"{root_folder}/ptbxl_database.csv")
            metadata.patient_id = metadata.patient_id.astype(int)

            def get_first_clean(g):
                clean_only = g[
                    g[
                        [
                            "baseline_drift",
                            "static_noise",
                            "burst_noise",
                            "electrodes_problems",
                        ]
                    ]
                    .isna()
                    .all(axis=1)
                ]
                if len(clean_only) > 0:
                    return clean_only.iloc[0]
                else:
                    return None

            metadata = metadata.groupby("patient_id").apply(get_first_clean)
            metadata = metadata.dropna(subset="ecg_id")

            metadata.to_csv(f"{root_folder}/ptbxl_database_filtered.csv")

        metadata.ecg_id = metadata.ecg_id.astype(int)
        self.metadata = metadata
        self.lowres = lowres
        self.root_folder = root_folder

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        # Outputs ECG data of shape sig_len x num_leads (e.g. for low res 1000 x 12)
        this_meta = self.metadata.iloc[index]
        ecg_id = this_meta["ecg_id"]
        sig, sigmeta = load_single_record(
            ecg_id, lowres=self.lowres, root_dir=self.root_folder
        )
        return torch.Tensor(sig).transpose(1, 0).float()


class PtbxlCleanDS(PtbxlDS):
    def _clean(self, sig_raw: np.ndarray, sigmeta: dict) -> np.ndarray:
        sig_clean = np.apply_along_axis(
            nk.ecg_clean, 1, sig_raw.transpose(), sampling_rate=sigmeta["fs"]
        )

        return sig_clean

    def __getitem__(self, index: int):
        this_meta = self.metadata.iloc[index]
        ecg_id = this_meta["ecg_id"]
        sig, sigmeta = load_single_record(
            ecg_id, lowres=self.lowres, root_dir=self.root_folder
        )

        sig_clean = self._clean(sig, sigmeta)

        return torch.Tensor(sig_clean).float()


class PtbxlSigWithRpeaksDS(PtbxlDS):

    def __init__(
        self, root_folder="./data", lowres=False, smoothing=False, stacked=False
    ):
        super().__init__(root_folder, lowres)

        self.stacked = stacked
        self.smoothing = smoothing

    def __getitem__(self, index):
        this_meta = self.metadata.iloc[index]
        ecg_id = this_meta["ecg_id"]
        sig, sigmeta = load_single_record(
            ecg_id, lowres=self.lowres, root_dir=self.root_folder
        )

        sig_clean = np.apply_along_axis(
            nk.ecg_clean, 1, sig.transpose(), sampling_rate=sigmeta["fs"]
        )

        # def get_rpeaks_binary(sig_in):
        #     info = nk.ecg_findpeaks(sig_in, sampling_rate=sigmeta["fs"])
        #     rpeaks = np.zeros_like(sig_clean[0, :])
        #     rpeaks[info["ECG_R_Peaks"]] = 1

        #     return rpeaks

        # rpeaks_all_channels = np.apply_along_axis(get_rpeaks_binary, 1, sig_clean)

        info = nk.ecg_findpeaks(sig_clean[1], sampling_rate=sigmeta["fs"])
        rpeaks = np.zeros_like(sig_clean[1, :])
        rpeaks[info["ECG_R_Peaks"]] = 1

        if self.smoothing:
            raise NotImplementedError()

        if self.stacked:
            return torch.concat(
                (
                    torch.Tensor(sig_clean[1, :]).unsqueeze(dim=0).float(),
                    torch.Tensor(rpeaks).unsqueeze(dim=0).float(),
                ),
                dim=0,
            )

        else:

            return (
                torch.Tensor(sig_clean).float(),
                torch.Tensor(rpeaks).float(),
            )


class PtbxlSmallSig(PtbxlCleanDS):

    def __init__(
        self, root_folder="./data", seq_len: int = None, channel_indices: list = None
    ):
        super().__init__(root_folder, lowres=True)

        if seq_len and seq_len > 1000:
            raise ValueError(
                f"Invalid seq_len {seq_len}. Do not use this dataset to create signals longer than 1000"
            )

        self.seq_len = seq_len
        self.channel_indices = channel_indices

    def __getitem__(self, index):
        sig = super().__getitem__(index)

        smallsig = sig

        if self.seq_len:
            smallsig = smallsig[:, 0 : self.seq_len]

        if self.channel_indices:
            smallsig = smallsig[self.channel_indices, :]

        return smallsig


if __name__ == "__main__":
    ds = PtbxlSigWithRpeaksDS(lowres=False)

    print(ds[0])

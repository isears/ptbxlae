import torch
import pandas as pd
import os
from ptbxlae.dataprocessing import load_single_record
import neurokit2 as nk
import numpy as np
import ast


class PtbxlDS(torch.utils.data.Dataset):
    def __init__(
        self,
        root_folder: str = "./data",
        lowres: bool = False,
        return_labels: bool = False,
    ):
        """Base PTBXL dataset initialization

        Args:
            root_folder (str, optional): Path to PTBXL data. Defaults to "./data".
            lowres (bool, optional): Whether to use the 100Hz (True) or 500Hz (False) data. Defaults to False.
            return_labels (bool, optional): Whether to return diagnostic labels for each EKG. Label returning was made optional because it is not necessary for the autoencoder training loop and will probably slow down the dataloaders significantly. Defaults to False.

        Returns:
            None: None
        """
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
        self.return_labels = return_labels

        # Get PTBXL labels
        self.metadata.scp_codes = self.metadata.scp_codes.apply(ast.literal_eval)

        # Modified from physionet example.py
        scp_codes = pd.read_csv(f"{root_folder}/scp_statements.csv", index_col=0)
        scp_codes = scp_codes[scp_codes.diagnostic == 1]

        self.ordered_labels = list()

        for diagnostic_code, description in zip(scp_codes.index, scp_codes.description):
            self.ordered_labels.append(description)
            self.metadata[description] = self.metadata.scp_codes.apply(
                lambda x: diagnostic_code in x.keys()
            ).astype(float)

    def __len__(self):
        return len(self.metadata)

    def _get_label(self, index: int):
        this_meta = self.metadata.iloc[index]
        # Probably over-kill but want to make certain order of labels is consistent
        return torch.Tensor([this_meta[c] for c in self.ordered_labels]).float()

    def __getitem__(self, index: int):
        # Outputs ECG data of shape sig_len x num_leads (e.g. for low res 1000 x 12)
        this_meta = self.metadata.iloc[index]
        ecg_id = this_meta["ecg_id"]
        sig, sigmeta = load_single_record(
            ecg_id, lowres=self.lowres, root_dir=self.root_folder
        )

        if self.return_labels:
            labels = self._get_label(index)
        else:
            labels = torch.Tensor([])

        return torch.Tensor(sig).transpose(1, 0).float(), labels

    def set_return_labels(self, return_labels: bool):
        """Set return labels flag
        Args:
            return_labels (bool): If true, dataset will return true labels instead of dummies
        """
        self.return_labels = return_labels


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

        if self.return_labels:
            return torch.Tensor(sig_clean).float(), self._get_label(index)
        else:
            return torch.Tensor(sig_clean).float(), torch.Tensor([])


if __name__ == "__main__":
    ds = PtbxlCleanDS(lowres=True, return_labels=True)

    print(ds[0])

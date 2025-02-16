import torch
import pandas as pd
import os
from ptbxlae.dataprocessing import load_single_ptbxl_record
import neurokit2 as nk
import numpy as np
import ast


class PtbxlDS(torch.utils.data.Dataset):
    def __init__(
        self,
        root_folder: str = "./data/ptbxl",
        lowres: bool = False,
        return_labels: bool = False,
    ):
        """Base PTBXL dataset initialization

        Args:
            root_folder (str, optional): Path to PTBXL data. Defaults to "./data/ptbxl".
            lowres (bool, optional): Whether to use the 100Hz (True) or 500Hz (False) data. Defaults to False.
            return_labels (bool, optional): Whether to return diagnostic labels for each EKG. Label returning was made optional because it is not necessary for the autoencoder training loop and will probably slow down the dataloaders significantly. Defaults to False.

        Returns:
            None: None
        """
        super(PtbxlDS, self).__init__()

        metadata = pd.read_csv(f"f{root_folder}/ptbxl_database.csv")
        metadata.ecg_id = metadata.ecg_id.astype(int)
        metadata.patient_id = metadata.patient_id.astype(int)
        self.metadata = metadata
        self.patient_ids = self.metadata["patient_id"].unique()
        self.pid_groups = self.metadata.groupby("patient_id")

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
        return len(self.pid_groups)

    def __getitem__(self, index: int):
        # Outputs ECG data of shape sig_len x num_leads (e.g. for low res 1000 x 12)
        patient_id = self.patient_ids[index]
        available_exams = self.pid_groups.get_group(patient_id)
        selected_exam = available_exams.sample(1, random_state=42).iloc[0]

        ecg_id = selected_exam["ecg_id"]
        sig, sigmeta = load_single_ptbxl_record(
            ecg_id, lowres=self.lowres, root_dir=self.root_folder
        )

        sig_clean = np.apply_along_axis(
            nk.ecg_clean,  # type: ignore
            1,
            sig.transpose(),  # type: ignore
            sampling_rate=sigmeta["fs"],
        )  # type: ignore

        if self.return_labels:
            labels = {c: selected_exam[c] for c in self.ordered_labels}
        else:
            labels = {}

        return torch.Tensor(sig_clean).float(), labels

    def set_return_labels(self, return_labels: bool):
        """Set return labels flag
        Args:
            return_labels (bool): If true, dataset will return true labels instead of dummies
        """
        self.return_labels = return_labels


class PtbxlMultilabeledDS(PtbxlDS):
    def __getitem__(self, index):
        sig, labels_dict = super().__getitem__(index)
        return sig, torch.Tensor(list(labels_dict.values())).float()


if __name__ == "__main__":
    ds = PtbxlDS(lowres=True, return_labels=True)

    print(ds[0])

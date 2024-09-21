import torch
import pandas as pd
import os
from ptbxlae.dataprocessing import load_single_record


class ptbxlDS(torch.utils.data.Dataset):
    def __init__(self, root_folder: str = "./data", lowres=True):

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

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        # Outputs ECG data of shape sig_len x num_leads (e.g. for low res 1000 x 12)
        ecg_id = self.metadata.iloc[index]["ecg_id"]
        sig, sigmeta = load_single_record(ecg_id, lowres=self.lowres)
        return sig

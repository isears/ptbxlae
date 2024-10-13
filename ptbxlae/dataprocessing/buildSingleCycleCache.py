from ptbxlae.dataprocessing.ptbxlDS import PtbxlCleanDS
from torch.utils.data import DataLoader
import torch
import pandas as pd
import os
from ptbxlae.dataprocessing import load_single_record
import neurokit2 as nk
import numpy as np
from tqdm import tqdm


class PtbxlSingleCycleCachingDS(torch.utils.data.Dataset):

    def __init__(self, root_folder: str = "./data", seq_len: int = 500):
        super().__init__()

        self.root_folder = root_folder
        self.metadata = pd.read_csv(f"{self.root_folder}/ptbxl_database.csv")
        self.metadata["patient_id"] = self.metadata["patient_id"].astype(int)

        self.seq_len = seq_len

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        this_meta = self.metadata.iloc[index]
        ecg_id = this_meta["ecg_id"]
        pt_id = this_meta["patient_id"]

        cache_dir = f"./cache/singlecycle_data/{pt_id:05d}/{ecg_id:05d}"

        sig, sigmeta = load_single_record(
            ecg_id, lowres=False, root_dir=self.root_folder
        )

        sig_clean = np.apply_along_axis(
            nk.ecg_clean, 1, sig.transpose(), sampling_rate=sigmeta["fs"]
        )

        # Use lead II for rpeak detection

        lead_II = sig_clean[1]
        _, info = nk.ecg_peaks(lead_II, sampling_rate=sigmeta["fs"])
        rpeaks = info["ECG_R_Peaks"]

        if len(rpeaks) == 0:
            return 0

        single_cycle_data = list()

        for peak_idx, peak in enumerate(rpeaks):
            # Make sure we have enough signal for a full cycle
            if ((peak - (self.seq_len // 2)) > 0) and (
                (peak + (self.seq_len // 2)) < sig_clean.shape[-1]
            ):
                df = pd.DataFrame(
                    data=sig_clean[
                        :, (peak - self.seq_len // 2) : (peak + self.seq_len // 2)
                    ].transpose(),
                    columns=[
                        "I",
                        "II",
                        "III",
                        "aVR",
                        "aVL",
                        "aVF",
                        "V1",
                        "V2",
                        "V3",
                        "V4",
                        "V5",
                        "V6",
                    ],
                )

                single_cycle_data.append(df)

        # Only create directory and save to disk if there is a valid peak
        if len(single_cycle_data) > 0:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            for idx, df in enumerate(single_cycle_data):
                df.to_parquet(f"{cache_dir}/cycle_{idx:02d}.parquet")

        return peak_idx


if __name__ == "__main__":
    ds = PtbxlSingleCycleCachingDS()

    dl = DataLoader(ds, num_workers=16, batch_size=32)

    for idx, batch in tqdm(enumerate(dl)):
        pass

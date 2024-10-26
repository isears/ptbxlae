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


class PtbxlSingleCycleDS(PtbxlCleanDS):
    """
    Selects a single cardiac cycle at random from the 10-second sample

    NOTE: each epoch will have slightly different examples based on the random selection
    """

    def __init__(self, root_folder: str = "./data"):
        # Keep things high-res given set seq_len
        super().__init__(root_folder=root_folder, lowres=False)

        # Most beats are < 500, small beats will be padded up to this size
        self.seq_len = 500
        # Keep track of the number of beats that were too big to fit in seq_len (ideally want this close to 0)
        self.exceeded_seq_len_count = 0
        self.random_generator = torch.Generator().manual_seed(42)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        this_meta = self.metadata.iloc[index]
        ecg_id = this_meta["ecg_id"]
        sig, sigmeta = load_single_record(
            ecg_id, lowres=self.lowres, root_dir=self.root_folder
        )

        sig_clean = np.apply_along_axis(
            nk.ecg_clean, 1, sig.transpose(), sampling_rate=sigmeta["fs"]
        )
        lead_II = sig_clean[1]

        # Get segments based off single lead (lead II)
        _, info = nk.ecg_peaks(lead_II, sampling_rate=sigmeta["fs"])

        # TODO: fix this, too
        try:
            _, delineations = nk.ecg_delineate(
                lead_II, rpeaks=info["ECG_R_Peaks"], sampling_rate=sigmeta["fs"]
            )
        except (ZeroDivisionError, ValueError) as e:
            print("[-] WARN neurokit unable to generate delinetions")
            padded_cycle = sig_clean[:, 0:500]
            padmask = np.ones_like(padded_cycle[0])
            return torch.Tensor(padded_cycle).float(), torch.Tensor(padmask).int()

        # Assuming delineations return equal-length arrays
        assert len(delineations["ECG_P_Onsets"]) == len(delineations["ECG_T_Offsets"])

        # Must have p_onset and t_offset
        valid_waves = [
            wave_idx
            for wave_idx, (p, t) in enumerate(
                zip(delineations["ECG_P_Onsets"], delineations["ECG_T_Offsets"])
            )
            if (not np.isnan(p)) and (not np.isnan(t))
        ]

        if len(valid_waves) == 0:
            # TODO: really need to fix this
            print("[-] WARN: signal without valid waves")
            padded_cycle = sig_clean[:, 0:500]
            padmask = np.ones_like(padded_cycle[0])
            return torch.Tensor(padded_cycle).float(), torch.Tensor(padmask).int()

        random_selection_idx = torch.randint(
            low=0, high=len(valid_waves), size=(1,), generator=self.random_generator
        ).item()
        valid_wave_idx = valid_waves[random_selection_idx]

        assert (not np.isnan(delineations["ECG_P_Onsets"][valid_wave_idx])) and (
            not np.isnan(delineations["ECG_T_Offsets"][valid_wave_idx])
        )

        single_cycle = sig_clean[
            :,
            delineations["ECG_P_Onsets"][valid_wave_idx] : delineations[
                "ECG_T_Offsets"
            ][valid_wave_idx],
        ]
        if single_cycle.shape[1] > self.seq_len:
            print(f"[-] WARN abnormally large cardiac cycle {single_cycle.shape[1]}")
            trim_total = single_cycle.shape[1] - self.seq_len

            left_trim = trim_total // 2 + (trim_total % 2)
            right_trim = trim_total // 2

            trimmed_cycle = single_cycle[
                :, left_trim : (single_cycle.shape[1] - right_trim)
            ]
            assert trimmed_cycle.shape[1] == self.seq_len, trimmed_cycle.shape[1]
            padmask = np.ones(self.seq_len)

            return torch.Tensor(trimmed_cycle).float(), torch.Tensor(padmask).int()

        pad_size = self.seq_len - single_cycle.shape[1]
        left_pad = pad_size // 2 + (pad_size % 2)
        right_pad = pad_size // 2

        padded_cycle = np.pad(
            single_cycle, ((0, 0), (left_pad, right_pad)), constant_values=0
        )
        padmask = np.concatenate(
            [
                np.zeros(
                    left_pad,
                ),
                np.ones(
                    single_cycle.shape[1],
                ),
                np.zeros(
                    right_pad,
                ),
            ]
        )

        return torch.Tensor(padded_cycle).float(), torch.Tensor(padmask).int()


if __name__ == "__main__":
    ds = PtbxlSigWithRpeaksDS(lowres=False)

    print(ds[0])

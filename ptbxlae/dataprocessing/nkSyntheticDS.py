import torch
import neurokit2 as nk
import numpy as np


class NkSyntheticDS(torch.utils.data.Dataset):

    def __init__(
        self,
        examples_per_epoch: float = 1e6,
        duration_s: int = 10,
        sampling_rate_hz: int = 100,
    ):
        super().__init__()

        self.examples_per_epoch = int(examples_per_epoch)
        self.generator = torch.Generator().manual_seed(0)
        self.seq_len = duration_s * sampling_rate_hz
        self.duration_s = duration_s
        self.sampling_rate_hz = sampling_rate_hz

        # Normal parameters (used by default)
        # ===================================
        # t, the starting position along the circle of each interval in radius
        self.ti = np.array((-70, -15, 0, 15, 100))
        # a, the amplitude of each spike
        self.ai = np.array((1.2, -5, 30, -7.5, 0.75))
        # b, the width of each spike
        self.bi = np.array((0.25, 0.1, 0.1, 0.1, 0.4))

    def __len__(self):
        return self.examples_per_epoch

    def __getitem__(self, index: int):
        ecg = nk.ecg_simulate(
            duration=self.duration_s * 2,
            sampling_rate=self.sampling_rate_hz,
            # https://pmc.ncbi.nlm.nih.gov/articles/PMC11137473
            heart_rate=np.random.normal(74.5, 8.5),
            method="multileads",
            random_state=42,
            ti=np.random.normal(self.ti, np.ones(5) * 3),
            ai=np.random.normal(self.ai, np.abs(self.ai / 5)),
            bi=np.random.normal(self.bi, np.abs(self.bi / 5)),
        )

        ecg_clean = np.apply_along_axis(
            nk.ecg_clean, 1, ecg.transpose(), sampling_rate=self.sampling_rate_hz
        )

        # Need to do random sliding window so that sequence doesn't always start on rpeak
        random_start = int(np.random.uniform(0, ecg_clean.shape[1] // 2))
        ecg_clean = ecg_clean[
            :, random_start : (ecg_clean.shape[1] // 2) + random_start
        ]

        return torch.Tensor(ecg_clean), torch.Tensor([])


class SinglechannelSyntheticDS(NkSyntheticDS):

    def __getitem__(self, index):
        ecg = nk.ecg_simulate(
            duration=self.duration_s * 2,
            sampling_rate=self.sampling_rate_hz,
            # https://pmc.ncbi.nlm.nih.gov/articles/PMC11137473
            heart_rate=np.random.normal(74.5, 8.5),
            method="ecgsyn",
            random_state=42,
            ti=np.random.normal(self.ti, np.ones(5) * 3),
            ai=np.random.normal(self.ai, np.abs(self.ai / 5)),
            bi=np.random.normal(self.bi, np.abs(self.bi / 5)),
        )

        ecg_clean = nk.ecg_clean(ecg, sampling_rate=self.sampling_rate_hz)

        # Need to do random sliding window so that sequence doesn't always start on rpeak
        random_start = int(np.random.uniform(0, len(ecg_clean) // 2))
        ecg_clean = ecg_clean[random_start : (len(ecg_clean) // 2) + random_start]

        return torch.Tensor(ecg_clean.astype(float)).unsqueeze(0), torch.Tensor([])


if __name__ == "__main__":
    ds = SinglechannelSyntheticDS()

    example = ds[0]

    print(example.shape)

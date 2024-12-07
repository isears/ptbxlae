import torch
import neurokit2 as nk


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

    def random_uniform(self, low: float, high: float) -> float:
        range = high - low
        return (torch.rand(1, generator=self.generator).item() * range) + low

    def random_normal(self, mean: float, std: float) -> float:
        return torch.normal(
            mean=torch.Tensor([mean]), std=torch.Tensor([std]), generator=self.generator
        ).item()

    def __len__(self):
        return self.examples_per_epoch

    def __getitem__(self, index: int):
        ecg = nk.ecg_simulate(
            duration=self.duration_s * 2,
            sampling_rate=self.sampling_rate_hz,
            noise=self.random_uniform(0, 0.5),
            # https://pmc.ncbi.nlm.nih.gov/articles/PMC11137473
            heart_rate=self.random_normal(74.5, 8.5),
            heart_rate_std=self.random_normal(1, 0.1),
            method="multileads",
            random_state=42,
        )

        # TODO: need to call nk.clean() here?

        # TODO: need to do random sliding window so that sequence doesn't always start on rpeak
        random_start = int(self.random_uniform(0, ecg.shape[-1] // 2))
        ecg = ecg[random_start : (ecg.shape[0] // 2) + random_start]

        return torch.Tensor(ecg.to_numpy().transpose())


class SinglechannelSyntheticDS(NkSyntheticDS):

    def __getitem__(self, index):
        ecg = nk.ecg_simulate(
            duration=self.duration_s * 2,
            sampling_rate=self.sampling_rate_hz,
            noise=self.random_uniform(0, 0.5),
            # https://pmc.ncbi.nlm.nih.gov/articles/PMC11137473
            heart_rate=self.random_normal(74.5, 8.5),
            heart_rate_std=self.random_normal(1, 0.1),
            method="ecgsyn",
            random_state=42,
        )

        # TODO: need to call nk.clean() here?

        # TODO: need to do random sliding window so that sequence doesn't always start on rpeak
        random_start = int(self.random_uniform(0, ecg.shape[-1] // 2))
        ecg = ecg[random_start : (ecg.shape[-1] // 2) + random_start]
        assert ecg.shape[-1] > 10, f"{random_start}"

        return torch.Tensor(ecg.transpose()).unsqueeze(0)


if __name__ == "__main__":
    ds = NkSyntheticDS()

    example = ds[0]

    print(example.shape)

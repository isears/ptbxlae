import torch
import pandas as pd
import glob
import os


class SingleCycleCachedDS(torch.utils.data.Dataset):

    def __init__(
        self, cache_path: str = "cache/singlecycle_data", randomness: bool = True
    ):
        super().__init__()

        self.cache_path = cache_path
        self.metadata = pd.read_csv(
            f"{self.cache_path}/ptbxl_database.csv",
            dtype={"patient_id": int},
            index_col="ecg_id",
        )

        self.patient_ids = [
            fname for fname in os.listdir(cache_path) if fname.isdigit()
        ]
        self.randomness = randomness
        self.random_generator = torch.Generator().manual_seed(42)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index: int):
        pid = self.patient_ids[index]

        available_ecg_paths = glob.glob(f"{self.cache_path}/{pid}/*")

        ecg_idx = torch.randint(
            low=0,
            high=len(available_ecg_paths),
            size=(1,),
            generator=self.random_generator,
        ).item()

        available_cycle_paths = glob.glob(f"{available_ecg_paths[ecg_idx]}/*")

        cycle_idx = torch.randint(
            low=0,
            high=len(available_cycle_paths),
            size=(1,),
            generator=self.random_generator,
        ).item()

        if self.randomness:
            df = pd.read_parquet(f"{available_cycle_paths[cycle_idx]}")

        else:
            first_ecg_dir = os.listdir(f"{self.cache_path}/{pid}/")[0]
            first_cycle_fname = os.listdir(f"{self.cache_path}/{pid}/{first_ecg_dir}/")[
                0
            ]

            df = pd.read_parquet(
                f"{self.cache_path}/{pid}/{first_ecg_dir}/{first_cycle_fname}"
            )

        return torch.Tensor(df.to_numpy().transpose()).float()


if __name__ == "__main__":
    ds = SingleCycleCachedDS(randomness=True)

    print(ds[0].shape)

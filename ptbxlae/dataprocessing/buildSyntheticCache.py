from ptbxlae.dataprocessing.nkSyntheticDS import NkSyntheticDS
from torch.utils.data import DataLoader
import os
import torch
from tqdm import tqdm


SAMPLES_PER_SAVE_BATCH = 1000
TOTAL_SAMPLES = 1e6


if __name__ == "__main__":
    ds = NkSyntheticDS(
        examples_per_epoch=TOTAL_SAMPLES, duration_s=10, sampling_rate_hz=100
    )

    dl = DataLoader(
        ds, num_workers=len(os.sched_getaffinity(0)), batch_size=SAMPLES_PER_SAVE_BATCH
    )

    for batch_idx, batch in tqdm(enumerate(dl), total=len(dl)):
        torch.save(batch, f"cache/synthetic-ekgs/{batch_idx:05d}.pt")

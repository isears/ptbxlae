from torch.utils.data import Dataset
import torch
import random


class BaseMaskingDS(Dataset):
    def __init__(self, unmasked_ekg_dataset: Dataset):
        super().__init__()
        self.ds = unmasked_ekg_dataset
        random.seed(42)

    def __len__(self):
        return len(self.ds)  # type: ignore


class ChannelMaskingDS(BaseMaskingDS):
    """
    Randomly mask a channel
    """

    def __getitem__(self, index):
        sig, meta = self.ds[index]

        # Make sure we have 12-lead EKG
        assert sig.shape[0] == 12

        masked_channel_idx = random.choice(range(0, 12))

        # NOTE: intended for internal logic, not return value
        _mask = torch.ones_like(sig[:, 0]).bool()

        _mask[masked_channel_idx] = False
        sig_masked = torch.zeros_like(sig)
        sig_masked[_mask] = sig[_mask]

        return sig, sig_masked, masked_channel_idx, meta


class SegmentMaskingDS(BaseMaskingDS):
    """
    Randomly mask a part of the sequence
    """

    def __init__(self, unmasked_ekg_dataset: Dataset, mask_proportion: float):
        super().__init__(unmasked_ekg_dataset)

        assert mask_proportion <= 1.0 and mask_proportion >= 0.0
        self.mask_proportion = mask_proportion

    def set_masking_propoprtion(self, mask_proportion: float):
        # TODO: eventually may want to gradually increase mask size as training goes on
        raise NotImplementedError()

    def __getitem__(self, index):
        sig, meta = self.ds[index]

        assert sig.shape[0] == 12

        seq_len = sig.shape[1]
        mask_len = int(seq_len * self.mask_proportion)

        mask_start_idx = random.choice(range(0, (seq_len - mask_len)))

        mask = torch.zeros_like(sig[0])
        mask[mask_start_idx : mask_start_idx + mask_len] = 1
        mask = mask.bool()
        sig_masked = sig * ~mask

        return sig, sig_masked, mask, meta


class MixedSegmentMaskingDS(SegmentMaskingDS):
    """
    Randomly mask part of each channel (as opposed to segment masking ds that masks same part of every channel)
    """

    def __getitem__(self, index):
        sig, meta = self.ds[index]

        assert sig.shape[0] == 12

        seq_len = sig.shape[1]
        mask_len = int(seq_len * self.mask_proportion)

        mask = torch.zeros_like(sig)

        for channel_idx in range(0, 12):
            mask_start_idx = random.choice(range(0, (seq_len - mask_len)))
            mask[channel_idx, mask_start_idx : mask_start_idx + mask_len] = 1

        mask = mask.bool()
        sig_masked = sig * ~mask

        return sig, sig_masked, mask, meta

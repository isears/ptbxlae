import torch
import pandas as pd
import random
import wfdb
import neurokit2 as nk
import numpy as np
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    String,
    DateTime,
    select,
    and_,
)
from sqlalchemy.ext.automap import automap_base

import datetime


class MimicDS(torch.utils.data.Dataset):
    def __init__(
        self,
        root_folder: str = "./data/mimiciv-ecg",
        return_labels: bool = False,
        freq: int = 100,
    ):
        super().__init__()

        # TODO: connection string params should be passed through args
        self.db_engine = create_engine("postgresql+psycopg2://readonly@/mimiciv")

        with self.db_engine.connect() as conn:
            print(f"Successfully connected: {conn}")

        self.db_metadata = MetaData()

        for schema_name in [
            "mimiciv_hosp",
            "mimiciv_ed",
            "mimiciv_icu",
            "mimiciv_derived",
        ]:
            # NOTE: allows access to tables like self.db_metadata.tables['mimiciv_hosp.patients']
            self.db_metadata.reflect(bind=self.db_engine, schema=schema_name)

        # self.hosp_patients_table = Table(
        #     "patients",
        #     self.db_metadata,
        #     autoload_with=self.db_engine,
        #     schema="mimiciv_hosp",
        # )

        self.root_folder = root_folder
        self.return_labels = return_labels
        self.record_list = (
            pd.read_csv(f"{root_folder}/record_list.csv")
            .groupby("subject_id")
            .agg(list)
        )

        self.freq = freq

        random.seed(42)

    def _get_patient_info(self, subject_id: int, date: datetime.date):
        stmt = select(self.hosp_patients_table).where(
            self.hosp_patients_table.columns.subject_id == subject_id
        )

        with self.db_engine.connect() as conn:
            results = conn.execute(stmt).fetchall()
            assert len(results) == 1  # subject_id should be unique in patients table

        return results[0]

    def __len__(self):
        return len(self.record_list)

    def __getitem__(self, index: int):
        possible_ecgs = self.record_list.iloc[index]["path"]
        path = f"{self.root_folder}/{random.choice(possible_ecgs)}"

        sig, meta = wfdb.rdsamp(path)
        subject_id = int(meta["comments"][0].split(":")[-1])
        date = meta["base_date"]

        patient_info = self._get_patient_info(subject_id, date)

        sig = sig.transpose()

        assert meta["sig_name"] == [
            "I",
            "II",
            "III",
            "aVR",
            "aVF",
            "aVL",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]

        # Sometimes sig contains nans, unfortunately
        # Addressing this with linear interpolation, as it seems to be a small number in most cases
        if np.isnan(sig).any():
            # print(
            #     f"[WARN] Found {np.isnan(sig).sum()} nans in {path}, interpolating..."
            # )
            for i in range(0, len(meta["sig_name"])):
                nans = np.isnan(sig[i, :])
                sig[i, nans] = np.interp(
                    nans.nonzero()[0], (~nans).nonzero()[0], sig[i, ~nans]
                )

        sig_resamp = np.apply_along_axis(
            nk.signal_resample,  # type: ignore
            1,
            sig,
            sampling_rate=meta["fs"],
            desired_sampling_rate=self.freq,
        )  # type: ignore

        assert sig_resamp.shape == (12, 10 * self.freq)

        sig_clean = np.apply_along_axis(
            nk.ecg_clean, 1, sig_resamp, sampling_rate=self.freq  # type: ignore
        )  # type: ignore

        return torch.Tensor(sig_clean).float(), {}


if __name__ == "__main__":
    ds = MimicDS()

    sig, labels = ds[0]

    print(sig.shape)

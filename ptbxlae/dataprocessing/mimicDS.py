import torch
import pandas as pd
import random
import wfdb
import neurokit2 as nk
import numpy as np
from sqlalchemy import (
    create_engine,
    MetaData,
    select,
)
from sqlalchemy.orm import Session

import datetime
from typing import Optional
from abc import ABC, abstractmethod


class MimicConnector(ABC):
    """
    Wanted to abstract this out so that other MIMIC data sources could be implemented

    E.g. will likely need a csv connector for running on OSCAR where postgres not available
    """

    @abstractmethod
    def get_demographics(self, subject_id: int, date_of_interest: datetime.date):
        pass


class MimicSqlConnector(MimicConnector):

    def __init__(self, uri: str):
        # local postgres: "postgresql+psycopg2://readonly@/mimiciv"
        self.db_engine = create_engine(uri)

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

    def get_demographics(self, subject_id: int, date_of_interest: datetime.date):
        """
        For now, just gender and age
        """
        t_patients = self.db_metadata.tables["mimiciv_hosp.patients"]

        stmt = select(
            t_patients.c.gender,
            t_patients.c.anchor_age,
            t_patients.c.anchor_year,
        ).where(t_patients.columns.subject_id == subject_id)

        with Session(self.db_engine) as s:
            result = s.execute(stmt)
            assert result.rowcount == 1  # subject_id should be unique in patients table

            info = result.fetchone()._asdict()

        # NOTE: ignores leap years, but age-resolution in MIMIC is really only down to the year so a few days +/- not relevant
        info["age"] = (
            date_of_interest
            - datetime.date(year=info["anchor_year"], month=1, day=1)
            + datetime.timedelta(days=(365 * info["anchor_age"]))
        ).days / 365

        info.pop("anchor_year")
        info.pop("anchor_age")

        return info


class MimicDS(torch.utils.data.Dataset):
    def __init__(
        self,
        root_folder: str = "./data/mimiciv-ecg",
        mimic_connector: Optional[MimicConnector] = None,
        freq: int = 100,
    ):
        super().__init__()

        self.mimic_connector = mimic_connector

        self.root_folder = root_folder
        self.record_list = (
            pd.read_csv(f"{root_folder}/record_list.csv")
            .groupby("subject_id")
            .agg(list)
        )

        self.freq = freq

        random.seed(42)

    def __len__(self):
        return len(self.record_list)

    def __getitem__(self, index: int):
        possible_ecgs = self.record_list.iloc[index]["path"]
        path = f"{self.root_folder}/{random.choice(possible_ecgs)}"

        sig, meta = wfdb.rdsamp(path)
        subject_id = int(meta["comments"][0].split(":")[-1])
        date = meta["base_date"]

        info = self.mimic_connector.get_demographics(subject_id, date)

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

        return torch.Tensor(sig_clean).float(), info


if __name__ == "__main__":

    ds = MimicDS(
        mimic_connector=MimicSqlConnector(uri="postgresql+psycopg2://readonly@/mimiciv")
    )

    sig, info = ds[0]

    print(sig.shape)

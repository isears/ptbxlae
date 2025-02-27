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

import sqlalchemy as db
from sqlalchemy.orm import Session
import ibis
from ibis import _
import datetime
from typing import Optional
from abc import ABC, abstractmethod
import psycopg2
import psycopg2.extras


class MimicConnector(ABC):
    """
    Wanted to abstract this out so that other MIMIC data sources could be implemented

    E.g. will likely need a csv connector for running on OSCAR where postgres not available
    """

    @abstractmethod
    def get_demographics(
        self, subject_id: int, datetime_of_interest: datetime.datetime
    ) -> dict:
        pass

    @abstractmethod
    def get_labs(
        self, subject_id: int, datetime_of_interest: datetime.datetime
    ) -> dict:
        # chem 7, cardiac
        pass

    @abstractmethod
    def get_cci(self, subject_id: int, datetime_of_interest: datetime.datetime) -> dict:
        pass

    @abstractmethod
    def get_meds(
        self, subject_id: int, datetime_of_interest: datetime.datetime
    ) -> dict:
        pass

    @abstractmethod
    def get_vitals(
        self, subject_id: int, datetime_of_interest: datetime.datetime
    ) -> dict:
        pass


class MimicIbisConnector(MimicConnector):
    def __init__(self, uri: str):
        super(MimicIbisConnector, self).__init__()

        self.con = ibis.postgres.connect(user="readonly", database="mimiciv")

    def get_demographics(
        self, subject_id: int, datetime_of_interest: datetime.datetime
    ) -> dict:
        return {}

    def get_labs(
        self, subject_id: int, datetime_of_interest: datetime.datetime
    ) -> dict:
        # chem 7, cardiac
        raise NotImplementedError

    def get_cci(self, subject_id: int, datetime_of_interest: datetime.datetime) -> dict:
        raise NotImplementedError

    def get_meds(
        self, subject_id: int, datetime_of_interest: datetime.datetime
    ) -> dict:
        raise NotImplementedError

    def get_vitals(
        self, subject_id: int, datetime_of_interest: datetime.datetime
    ) -> dict:
        raise NotImplementedError


class MimicSqlAlchemyConnector(MimicConnector):

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

    def get_demographics(
        self, subject_id: int, datetime_of_interest: datetime.datetime
    ) -> dict:
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
            datetime_of_interest
            - datetime.datetime(year=info["anchor_year"], month=1, day=1)
            + datetime.timedelta(days=(365 * info["anchor_age"]))
        ).days / 365

        info.pop("anchor_year")
        info.pop("anchor_age")

        return info

    def get_labs(self, subject_id, date_of_interest) -> dict:
        t_chem7 = self.db_metadata.tables["mimiciv_derived.chemistry"]
        t_cardiac_marker = self.db_metadata.tables["mimiciv_derived.cardiac_marker"]

        # TODO: could add this as argument
        window = datetime.timedelta(hours=24)

        stmt = select(t_chem7).filter(
            (t_chem7.columns.subject_id == subject_id)
            & (
                t_chem7.columns.charttime.between(
                    date_of_interest - window, date_of_interest + window
                )
            )
        )

        with Session(self.db_engine) as s:
            res = s.query(
                db.func.avg(t_chem7.columns.chloride, t_chem7.columns.potassium)
            ).filter(t_chem7.columns.subject_id == subject_id)

        # TODO: incomplete implementation, going to just try to do something else
        raise NotImplementedError
        return {}

    def get_cci(self, subject_id, date_of_interest):
        raise NotImplementedError

    def get_meds(self, subject_id, date_of_interest):
        raise NotImplementedError

    def get_vitals(self, subject_id, date_of_interest):
        raise NotImplementedError


class MimicSqlConnector(MimicConnector):
    def __init__(
        self,
        database: str,
        user: str,
        password: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        super().__init__()
        self.connection = psycopg2.connect(
            database=database, user=user, password=password, host=host, port=port
        )
        self.cursor = self.connection.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        )

    def get_demographics(self, subject_id, datetime_of_interest) -> dict:
        q = """
        --begin-sql
        SELECT gender, anchor_age, anchor_year FROM mimiciv_hosp.patients WHERE subject_id = %s;
        """, (
            str(subject_id),
        )

        self.cursor.execute(*q)
        records = self.cursor.fetchall()
        assert len(records) == 1  # subject ids should be unique
        info = records[0]

        info["age"] = (
            datetime_of_interest
            - datetime.datetime(year=info["anchor_year"], month=1, day=1)
            + datetime.timedelta(days=(365 * info["anchor_age"]))
        ).days / 365

        info["female"] = info["gender"] == "F"
        info.pop("anchor_year")
        info.pop("anchor_age")
        info.pop("gender")

        return info

    def get_labs(self, subject_id, datetime_of_interest) -> dict:
        q = (
            """
        --begin-sql
        SELECT
            albumin,
            globulin,
            total_protein,
            aniongap,
            bicarbonate,
            bun,
            calcium,
            chloride,
            creatinine,
            glucose,
            sodium,
            potassium,
            ABS(EXTRACT(EPOCH FROM (charttime - TIMESTAMP %(t)s))) AS diff_seconds
        FROM mimiciv_derived.chemistry
        WHERE subject_id = %(id)s
        AND charttime BETWEEN TIMESTAMP %(t)s - INTERVAL '12 hours'
            AND TIMESTAMP %(t)s + INTERVAL '12 hours'
        ORDER BY diff_seconds;
        """,
            {
                "id": str(subject_id),
                "t": str(datetime_of_interest),
            },
        )

        self.cursor.execute(*q)
        records = self.cursor.fetchall()

        df = pd.DataFrame(records)

        return df.bfill().drop(columns="diff_seconds").iloc[0].to_dict()

    def get_cci(self, subject_id, datetime_of_interest):
        raise NotImplementedError

    def get_meds(self, subject_id, datetime_of_interest):
        raise NotImplementedError

    def get_vitals(self, subject_id, datetime_of_interest) -> dict:
        q = (
            """
        --begin-sql
        WITH vitalsign_combined AS (SELECT subject_id,
                                   charttime,
                                   heart_rate,
                                   sbp,
                                   dbp,
                                   resp_rate,
                                   temperature,
                                   spo2
                            FROM mimiciv_derived.vitalsign
                            UNION
                            SELECT subject_id,
                                   charttime,
                                   heartrate,
                                   sbp,
                                   dbp,
                                   resprate,
                                   temperature,
                                   o2sat
                            FROM mimiciv_ed.vitalsign)
        SELECT AVG(heart_rate)  AS heart_rate,
            AVG(sbp)         AS sbp,
            AVG(dbp)         AS dbp,
            AVG(resp_rate)   AS resp_rate,
            AVG(temperature) AS temperature,
            AVG(spo2)        AS spo2
        FROM vitalsign_combined
        WHERE subject_id = %(id)s
        AND charttime BETWEEN TIMESTAMP %(t)s - INTERVAL '12 hours'
            AND TIMESTAMP %(t)s + INTERVAL '12 hours';
        """,
            {
                "id": str(subject_id),
                "t": str(datetime_of_interest),
            },
        )

        self.cursor.execute(*q)
        records = self.cursor.fetchall()
        assert len(records) == 1  # subject ids should be unique
        return records[0]


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
        datetime_of_study = datetime.datetime.combine(
            meta["base_date"], meta["base_time"]
        )

        info = {}
        if self.mimic_connector:
            info.update(
                self.mimic_connector.get_demographics(subject_id, datetime_of_study)
            )
            info.update(self.mimic_connector.get_labs(subject_id, datetime_of_study))
            info.update(self.mimic_connector.get_vitals(subject_id, datetime_of_study))

        # Probably overkill but want to explicitly guarantee consistent order across workers
        info_sorted = {key: value for key, value in sorted(info.items())}

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

        return (
            torch.Tensor(sig_clean).float(),
            torch.Tensor(
                [k if k else float("nan") for k in info_sorted.values()]
            ).float(),
        )


if __name__ == "__main__":

    ds = MimicDS(mimic_connector=MimicSqlConnector(database="mimiciv", user="readonly"))

    sig, info = ds[0]

    print(sig.shape)

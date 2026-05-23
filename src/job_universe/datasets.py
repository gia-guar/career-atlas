"""Custom Kedro datasets for the job-universe project."""

from __future__ import annotations

import pandas as pd
from kedro_datasets.pandas import ParquetDataset

from job_universe.schemas import JOB_POSTING_COLUMNS


class SeededParquetDataset(ParquetDataset):
    """ParquetDataset that returns an empty JobPosting frame on missing file.

    Used for the cumulative store's *input* side so the first pipeline run does
    not have to bootstrap the file manually. The *output* side is a vanilla
    ParquetDataset writing to the same path.
    """

    def load(self) -> pd.DataFrame:
        try:
            return super().load()
        except FileNotFoundError:
            return _empty_frame()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "no such file" in msg or "does not exist" in msg or "not found" in msg:
                return _empty_frame()
            raise


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype=object) for col in JOB_POSTING_COLUMNS})

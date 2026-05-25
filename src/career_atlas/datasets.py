"""Custom Kedro datasets for the career-atlas project."""

from __future__ import annotations

from typing import Any

import pandas as pd
from kedro_datasets.pandas import ParquetDataset

from career_atlas.schemas import JOB_POSTING_COLUMNS


class SeededParquetDataset(ParquetDataset):
    """ParquetDataset that returns an empty frame on missing file.

    Used for cumulative stores' *input* side so the first pipeline run does
    not have to bootstrap the file manually. The *output* side is a vanilla
    ParquetDataset writing to the same path.

    The empty-frame schema defaults to `JobPosting` columns (the original
    use case) but can be overridden via the ``empty_columns`` constructor
    arg — e.g. for the Stage 3 ``posting_skills`` cache, which has its own
    ``(posting_id, name, kind)`` shape.
    """

    def __init__(
        self,
        *args: Any,
        empty_columns: list[str] | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._empty_columns: tuple[str, ...] | None = (
            tuple(empty_columns) if empty_columns is not None else None
        )

    def load(self) -> pd.DataFrame:
        try:
            return super().load()
        except FileNotFoundError:
            return self._make_empty()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "no such file" in msg or "does not exist" in msg or "not found" in msg:
                return self._make_empty()
            raise

    def _make_empty(self) -> pd.DataFrame:
        cols = self._empty_columns or JOB_POSTING_COLUMNS
        return pd.DataFrame({col: pd.Series(dtype=object) for col in cols})


def _empty_frame() -> pd.DataFrame:
    """Back-compat helper; kept so existing imports continue to work."""
    return pd.DataFrame({col: pd.Series(dtype=object) for col in JOB_POSTING_COLUMNS})

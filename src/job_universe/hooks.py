"""Project hooks."""

from __future__ import annotations

from typing import Any

from kedro.framework.hooks import hook_impl
from kedro.io import MemoryDataset


class CredentialsHook:
    """Expose `conf/local/credentials.yml` to nodes via a `credentials` dataset.

    Without this, credentials are only accessible to dataset configs (via the
    `credentials:` key). Pipeline nodes that need raw API keys — like the
    Adzuna fetcher — read them from this MemoryDataset instead of duplicating
    config-loader plumbing.
    """

    @hook_impl
    def after_catalog_created(
        self,
        catalog,
        conf_catalog: dict[str, Any],
        conf_creds: dict[str, Any],
        parameters: dict[str, Any],
        save_version: str,
        load_versions: dict[str, str],
    ) -> None:
        catalog["credentials"] = MemoryDataset(data=conf_creds or {})

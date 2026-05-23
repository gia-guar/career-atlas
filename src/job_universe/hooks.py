"""Project hooks."""

from __future__ import annotations

from typing import Any

from kedro.framework.hooks import hook_impl
from kedro.io import MemoryDataset

from job_universe.llm import OllamaClient


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


class OllamaClientHook:
    """Expose a live `OllamaClient` to nodes via a `ollama_client` dataset.

    Same mechanism as `CredentialsHook`: nodes that need to call the LLM
    declare `ollama_client` as an input and the hook materialises it from
    `parameters["cv_extraction"]["ollama"]`. Tests bypass this hook by
    populating the catalog directly with a stub client.

    `copy_mode="assign"` is load-bearing: the underlying `httpx.Client`
    holds `_thread.RLock` instances, which break Kedro's default
    deepcopy-on-load. Assignment hands nodes the same client object.
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
        cv_params = (parameters or {}).get("cv_extraction") or {}
        ollama_cfg = cv_params.get("ollama") or {}
        client = OllamaClient(
            host=ollama_cfg.get("host", "http://localhost:11434"),
            timeout_s=float(ollama_cfg.get("request_timeout_s", 180)),
        )
        catalog["ollama_client"] = MemoryDataset(data=client, copy_mode="assign")

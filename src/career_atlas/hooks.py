"""Project hooks."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from kedro.framework.hooks import hook_impl
from kedro.io import MemoryDataset

from career_atlas.llm import OllamaClient
from career_atlas.web import progress as web_progress
from career_atlas.web.progress import ProgressEmitter

logger = logging.getLogger(__name__)


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


class SkillEmbedderHook:
    """Expose a `skill_embedder` callable to nodes via the catalog.

    Lazy-instantiates a `sentence_transformers.SentenceTransformer` on the
    first call so the heavy torch import never happens for pipelines /
    tests that don't need it. Subsequent calls reuse the same in-process
    model.

    `trust_remote_code=True` is required for `nomic-ai/nomic-embed-text-v1.5`
    because Nomic ships custom modeling code in their HF repo — do not
    remove this without swapping the model.

    `copy_mode="assign"` is load-bearing: the underlying torch model
    holds non-deepcopy-safe state, same issue as `OllamaClientHook`.
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
        cfg = (parameters or {}).get("skill_map", {}).get("canonicalize") or {}
        model_name = cfg.get("model", "nomic-ai/nomic-embed-text-v1.5")
        batch_size = int(cfg.get("batch_size", 64))

        # Prefix is owned by the canonicalize layer (see build_canonical_map);
        # the embedder just encodes whatever strings it's handed.
        model_state: dict[str, Any] = {"model": None}

        def embed(names: list[str]) -> np.ndarray:
            if not names:
                return np.zeros((0, 1), dtype=np.float32)
            if model_state["model"] is None:
                from sentence_transformers import SentenceTransformer

                logger.info("loading sentence-transformer %s", model_name)
                model_state["model"] = SentenceTransformer(
                    model_name, trust_remote_code=True
                )
            return model_state["model"].encode(
                list(names),
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )

        catalog["skill_embedder"] = MemoryDataset(data=embed, copy_mode="assign")


class ProgressHook:
    """Expose a `progress_emitter` to nodes for live UI updates.

    Reads ``career_atlas.web.progress.CURRENT`` — set by the web runner before
    invoking ``KedroSession.run`` and cleared in ``finally``. When the slot is
    ``None`` (CLI runs, tests) a no-op emitter is registered, so emit-calls
    inside the pipeline do nothing.

    Same ``copy_mode="assign"`` constraint as the other hook-provided clients:
    the emitter holds a reference to an asyncio loop which is not deepcopy-safe.
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
        emitter = web_progress.CURRENT or ProgressEmitter()
        catalog["progress_emitter"] = MemoryDataset(data=emitter, copy_mode="assign")

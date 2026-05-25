"""Skill-name canonicalization via sentence-transformer embeddings + clustering.

Stage 3 collects raw skill/tool/requirement strings from many job postings
plus the user's CV. The same concept appears under many surface forms —
"PyTorch", "pytorch", "Py-Torch", "PyTorch (with CUDA)". This module
clusters semantically-close variants into one canonical name so downstream
frequency, co-occurrence, and CV matching all operate on a consistent
vocabulary.

The embedding model is injected as a callable to keep this module
testable without loading torch. The hook in `hooks.py` materialises a
real `sentence_transformers.SentenceTransformer` and exposes it as
`skill_embedder` in the Kedro catalog; tests inject a stub returning
deterministic vectors.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

Embedder = Callable[[list[str]], np.ndarray]


_WS_RE = re.compile(r"\s+")


def normalize_skill(name: str | None) -> str:
    """Lowercase + collapse whitespace + strip. Bare-minimum normalisation."""
    if name is None:
        return ""
    return _WS_RE.sub(" ", str(name).lower()).strip()


def _cluster_labels(
    embeddings: np.ndarray, distance_threshold: float
) -> np.ndarray:
    """Run AgglomerativeClustering with cosine distance.

    Imported lazily so the rest of the module can be exercised by tests
    without sklearn being installed (sklearn IS a runtime dep, but we
    prefer not to pay its import cost for non-canonicalize code paths).
    """
    from sklearn.cluster import AgglomerativeClustering

    if len(embeddings) <= 1:
        return np.zeros(len(embeddings), dtype=int)

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    return clusterer.fit_predict(embeddings)


def pick_canonical_name(members: list[str], counts: dict[str, int]) -> str:
    """Most-frequent member wins; ties broken by shortest, then alphabetical."""
    if not members:
        raise ValueError("members must be non-empty")
    # max() with a tuple key — sort key reverses for length and alpha so the
    # smallest-length / earliest-alpha wins among count ties.
    best = min(members, key=lambda m: (-counts.get(m, 0), len(m), m))
    return best


def build_canonical_map(
    raw_counts: dict[str, int],
    embedder: Embedder,
    *,
    encode_prefix: str = "clustering: ",
    distance_threshold: float = 0.15,
) -> dict[str, str]:
    """Map each normalized raw name to its canonical cluster representative.

    Parameters
    ----------
    raw_counts:
        ``{normalized_name: occurrence_count}``. Counts are used both for
        canonical-name selection (most-frequent wins) and to skip the
        clustering call entirely when the vocab has zero or one entry.
    embedder:
        Function taking a list of strings → 2D ``np.ndarray`` of
        L2-normalised embeddings (one row per input).
    encode_prefix:
        Prepended to each name before embedding. Nomic's
        ``nomic-embed-text-v1.5`` recommends ``"clustering: "``.
    distance_threshold:
        Cosine-distance cutoff for ``AgglomerativeClustering``. Smaller →
        stricter merging (more clusters, more conservative); larger →
        looser (fewer clusters, more aggressive synonym merging).
    """
    names = list(raw_counts.keys())
    if not names:
        return {}
    if len(names) == 1:
        return {names[0]: names[0]}

    inputs = [f"{encode_prefix}{n}" for n in names]
    embeddings = embedder(inputs)
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings)
    if embeddings.shape[0] != len(names):
        raise ValueError(
            f"embedder returned {embeddings.shape[0]} rows for {len(names)} names"
        )

    labels = _cluster_labels(embeddings, distance_threshold=distance_threshold)

    clusters: dict[int, list[str]] = {}
    for name, label in zip(names, labels):
        clusters.setdefault(int(label), []).append(name)

    mapping: dict[str, str] = {}
    for members in clusters.values():
        canonical = pick_canonical_name(members, raw_counts)
        for m in members:
            mapping[m] = canonical

    logger.info(
        "canonicalize: %d raw names → %d clusters", len(names), len(clusters)
    )
    return mapping


def build_category_map(
    canonical_names: list[str],
    embedder: Embedder,
    *,
    encode_prefix: str = "clustering: ",
    distance_threshold: float = 0.45,
    counts: dict[str, int] | None = None,
) -> dict[str, str]:
    """Second-pass clustering on the already-canonical vocabulary.

    Same agglomerative machinery as ``build_canonical_map`` but at a coarser
    distance threshold so concept families collapse together — e.g.
    ``{ai, machine learning, pytorch, mlops}`` end up in one category while
    ``{react, vue, node.js}`` form another. The label per cluster is picked
    with the same tie-break as canonicalisation (most-frequent wins;
    shortest then alphabetical). If ``counts`` is None all members tie on
    count, so the shortest-then-alpha tie-break drives the choice — usually
    yielding the most general name.
    """
    if not canonical_names:
        return {}
    names = list(dict.fromkeys(canonical_names))  # dedup, preserve order
    if len(names) == 1:
        return {names[0]: names[0]}

    inputs = [f"{encode_prefix}{n}" for n in names]
    embeddings = embedder(inputs)
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings)
    if embeddings.shape[0] != len(names):
        raise ValueError(
            f"embedder returned {embeddings.shape[0]} rows for {len(names)} names"
        )

    labels = _cluster_labels(embeddings, distance_threshold=distance_threshold)

    clusters: dict[int, list[str]] = {}
    for name, label in zip(names, labels):
        clusters.setdefault(int(label), []).append(name)

    counts = counts or {}
    mapping: dict[str, str] = {}
    for members in clusters.values():
        label = pick_canonical_name(members, counts)
        for m in members:
            mapping[m] = label

    logger.info(
        "categorise: %d canonical names → %d categories",
        len(names),
        len(clusters),
    )
    return mapping


def compute_tsne_positions(
    names: list[str],
    embedder: Embedder,
    *,
    encode_prefix: str = "clustering: ",
    perplexity: float = 30.0,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Encode ``names`` via the embedder, t-SNE-project to 2D, and return
    ``{name: {"x": float, "y": float}}`` rescaled so each axis lies in
    roughly [-1, 1]. The front-end multiplies by a viewport spread to get
    pixels — keeps the JSON producer-agnostic.

    Perplexity is the only knob with a sensible default. With fewer than ~30
    names, sklearn requires perplexity < n_samples, so we clamp to
    ``min(perplexity, max(5, n // 4))``. ``init="pca"`` gives a deterministic
    init that combines with ``random_state`` for reproducible maps.
    """
    if not names:
        return {}
    unique = list(dict.fromkeys(names))  # preserve order, dedup
    if len(unique) == 1:
        return {unique[0]: {"x": 0.0, "y": 0.0}}

    from sklearn.manifold import TSNE

    inputs = [f"{encode_prefix}{n}" for n in unique]
    embeddings = embedder(inputs)
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings)
    if embeddings.shape[0] != len(unique):
        raise ValueError(
            f"embedder returned {embeddings.shape[0]} rows for {len(unique)} names"
        )

    effective_perplexity = min(perplexity, max(5, len(unique) // 4))
    # sklearn requires perplexity strictly less than the number of samples.
    effective_perplexity = float(min(effective_perplexity, len(unique) - 1))

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(embeddings)

    # Rescale to [-1, 1] per axis — symmetric, so the JS layer doesn't need
    # to know the original t-SNE range.
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    span = np.maximum(mx - mn, 1e-9)
    centered = coords - (mn + mx) / 2.0
    normalised = 2.0 * centered / span  # → [-1, 1]

    logger.info("tsne: %d names → 2D (perplexity=%.1f)", len(unique), effective_perplexity)
    return {
        name: {"x": float(normalised[i, 0]), "y": float(normalised[i, 1])}
        for i, name in enumerate(unique)
    }


def collect_raw_counts(
    posting_skill_names: list[str], cv_skill_names: list[str]
) -> dict[str, int]:
    """Combine corpora into a single ``{normalized_name: count}`` dict.

    CV names contribute +1 each so they participate in clustering even if a
    given variant never appears in a posting.
    """
    counter: Counter[str] = Counter()
    for raw in posting_skill_names:
        n = normalize_skill(raw)
        if n:
            counter[n] += 1
    for raw in cv_skill_names:
        n = normalize_skill(raw)
        if n:
            counter[n] += 1
    return dict(counter)


def canonicalize_dataframe(
    df: Any,
    name_column: str,
    canonical_map: dict[str, str],
    out_column: str = "canonical",
) -> Any:
    """Add a column with the canonical name; rows whose normalized name is
    not in ``canonical_map`` are dropped.

    Implemented with the pandas namespace lazily to keep the helper testable
    in isolation.
    """
    import pandas as pd

    if df.empty:
        return df.assign(**{out_column: pd.Series(dtype=object)})
    normalized = df[name_column].map(normalize_skill)
    mapped = normalized.map(canonical_map)
    out = df.assign(**{out_column: mapped})
    return out[out[out_column].notna()].reset_index(drop=True)

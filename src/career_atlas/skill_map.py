"""Skill-map computation + rendering helpers.

Provider-agnostic — mirrors the layout of ``scraping.py``. Pipeline nodes
import these functions; tests target them directly.

The map node objects have shape::

    {"id": str, "count": int, "user_has": bool}

The map edge objects have shape::

    {"source": str, "target": str, "weight": float, "cooccurrence": int}

``weight`` is the positive-PMI score; edges with non-positive PMI or
co-occurrence below threshold are filtered out before this is emitted.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from itertools import combinations
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Map construction
# ---------------------------------------------------------------------------


def compute_node_counts(canonical_df: pd.DataFrame) -> dict[str, int]:
    """Count distinct postings each canonical skill appears in."""
    if canonical_df.empty:
        return {}
    deduped = canonical_df.drop_duplicates(subset=["posting_id", "canonical"])
    return deduped["canonical"].value_counts().to_dict()


def compute_cooccurrence(canonical_df: pd.DataFrame) -> Counter[tuple[str, str]]:
    """Count un-ordered pairs of canonical skills co-occurring within a posting."""
    if canonical_df.empty:
        return Counter()
    counter: Counter[tuple[str, str]] = Counter()
    grouped = canonical_df.drop_duplicates(
        subset=["posting_id", "canonical"]
    ).groupby("posting_id")["canonical"]
    for _, skills in grouped:
        unique = sorted(set(skills))
        if len(unique) < 2:
            continue
        for a, b in combinations(unique, 2):
            counter[(a, b)] += 1
    return counter


def pmi(c_ab: int, c_a: int, c_b: int, n_postings: int) -> float:
    """log( N * c_ab / (c_a * c_b) ) — positive ⇒ stronger-than-chance association."""
    if c_ab == 0 or c_a == 0 or c_b == 0 or n_postings == 0:
        return float("-inf")
    return math.log((n_postings * c_ab) / (c_a * c_b))


def determine_user_skills(
    cv_skill_names: list[str], canonical_map: dict[str, str]
) -> set[str]:
    """Resolve CV skill names into the set of canonical skills the user has."""
    from career_atlas.canonicalize import normalize_skill

    user_canonical: set[str] = set()
    for raw in cv_skill_names:
        normalized = normalize_skill(raw)
        canonical = canonical_map.get(normalized)
        if canonical is not None:
            user_canonical.add(canonical)
    return user_canonical


def compute_node_positions(
    node_ids: list[str],
    embedder,
    *,
    perplexity: float = 30.0,
    random_state: int = 42,
    encode_prefix: str = "clustering: ",
) -> dict[str, dict[str, float]]:
    """Return ``{id: {"x": float, "y": float}}`` in [-1, 1], computed by
    t-SNE projection of the embedder's vectors. Delegates to
    ``canonicalize.compute_tsne_positions``."""
    if not node_ids:
        return {}
    from career_atlas.canonicalize import compute_tsne_positions

    return compute_tsne_positions(
        node_ids,
        embedder=embedder,
        encode_prefix=encode_prefix,
        perplexity=perplexity,
        random_state=random_state,
    )


def build_map(
    posting_skills: pd.DataFrame,
    canonical_map: dict[str, str],
    cv_skill_names: list[str],
    category_map: dict[str, str] | None = None,
    embedder=None,
    *,
    min_node_count: int = 3,
    tsne_perplexity: float = 30.0,
    tsne_random_state: int = 42,
    encode_prefix: str = "clustering: ",
) -> dict[str, Any]:
    """Aggregate per-posting skill rows into a filtered node/edge map.

    ``posting_skills`` must have at least the columns ``posting_id`` and
    ``name``. Rows whose normalized name is not in ``canonical_map`` are
    dropped (typically empty / blacklisted).
    """
    from career_atlas.canonicalize import canonicalize_dataframe

    canonical_df = canonicalize_dataframe(
        posting_skills, name_column="name", canonical_map=canonical_map
    )

    node_counts = compute_node_counts(canonical_df)
    n_postings = canonical_df["posting_id"].nunique() if not canonical_df.empty else 0

    user_canonical = determine_user_skills(cv_skill_names, canonical_map)

    # Build filtered node list. We keep nodes the user owns even if they
    # fall under min_node_count — the user's whole point is to see their
    # skills against the market.
    nodes = []
    for name, count in sorted(node_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        user_has = name in user_canonical
        if count < min_node_count and not user_has:
            continue
        node: dict[str, Any] = {
            "id": name,
            "count": int(count),
            "user_has": user_has,
        }
        if category_map is not None:
            node["category"] = category_map.get(name)
        nodes.append(node)

    # Semantic positions via t-SNE on the canonical-name embeddings. Same
    # 2D coordinates are reused by the matplotlib PNG so the two views agree.
    if embedder is not None and nodes:
        positions = compute_node_positions(
            [n["id"] for n in nodes],
            embedder=embedder,
            perplexity=tsne_perplexity,
            random_state=tsne_random_state,
            encode_prefix=encode_prefix,
        )
        for n in nodes:
            n["position"] = positions.get(n["id"], {"x": 0.0, "y": 0.0})

    logger.info(
        "skill_map: %d nodes (kept), %d postings", len(nodes), n_postings
    )
    return {"nodes": nodes, "n_postings": int(n_postings)}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_skill_map_figure(
    map_data: dict[str, Any],
    viz_params: dict[str, Any],
) -> Any:
    """Return a matplotlib Figure suitable for `MatplotlibWriter`.

    Imports of matplotlib / networkx are local so the rest of this module
    stays importable without an X server / Agg backend selected.
    """
    import matplotlib

    # Headless backend chosen at function-call time so tests can override
    # globally with ``matplotlib.use("Agg")`` first.
    if matplotlib.get_backend().lower() not in {"agg", "module://matplotlib_inline.backend_inline"}:
        try:
            matplotlib.use("Agg", force=False)
        except Exception:  # noqa: BLE001
            pass

    import matplotlib.pyplot as plt
    import networkx as nx

    nodes = map_data.get("nodes") or []

    bg = viz_params.get("background_color", "#000000")
    color_default = viz_params.get("node_color_default", "#888888")
    color_user = viz_params.get("node_color_user_has", "#3F704D")
    node_alpha = float(viz_params.get("node_alpha", 0.85))
    label_top_n = int(viz_params.get("label_top_n", 15))
    figsize = tuple(viz_params.get("figsize", [16, 16]))
    node_size_scale = float(viz_params.get("node_size_scale", 40))
    label_color = viz_params.get("label_color", "#FFFFFF")
    label_font_size = int(viz_params.get("label_font_size", 14))

    g = nx.Graph()
    for n in nodes:
        g.add_node(n["id"], count=n["count"], user_has=n["user_has"])

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axis("off")

    if g.number_of_nodes() == 0:
        ax.text(
            0.5,
            0.5,
            "No skills extracted yet — run the scraping + extraction pipelines first.",
            color=label_color,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Positions are baked into the JSON by build_map (t-SNE on embeddings).
    # Fall back to centroid only if a node is missing one (defensive).
    pos = {
        n["id"]: (
            float(n.get("position", {}).get("x", 0.0)),
            float(n.get("position", {}).get("y", 0.0)),
        )
        for n in nodes
    }

    node_list = list(g.nodes())
    counts = [g.nodes[n]["count"] for n in node_list]
    sizes = [max(20.0, math.sqrt(c) * node_size_scale) for c in counts]
    colors = [color_user if g.nodes[n]["user_has"] else color_default for n in node_list]

    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=node_list,
        node_size=sizes,
        node_color=colors,
        alpha=node_alpha,
        linewidths=0,
        ax=ax,
    )

    top_labels = sorted(node_list, key=lambda n: g.nodes[n]["count"], reverse=True)
    label_dict = {n: n for n in top_labels[:label_top_n]}
    if label_dict:
        nx.draw_networkx_labels(
            g,
            pos,
            labels=label_dict,
            font_color=label_color,
            font_size=label_font_size,
            ax=ax,
        )

    fig.tight_layout(pad=0.5)
    return fig


def nodes_dataframe(map_data: dict[str, Any]) -> pd.DataFrame:
    """Flat DataFrame of all canonical nodes (count + user_has) for the CSV sidecar."""
    nodes = map_data.get("nodes") or []
    if not nodes:
        return pd.DataFrame(columns=["id", "count", "user_has"])
    return pd.DataFrame(nodes).sort_values("count", ascending=False).reset_index(drop=True)

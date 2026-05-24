"""Skill canonicalization via injected fake embedder."""

from __future__ import annotations

import numpy as np
import pytest

from job_universe.canonicalize import (
    build_canonical_map,
    build_category_map,
    collect_raw_counts,
    normalize_skill,
    pick_canonical_name,
)


class DeterministicEmbedder:
    """Returns a preset vector based on which substring (post-prefix) matches.

    Each name is matched against ``substrings`` in order; the first hit picks
    its vector. Names that don't match get a unique-per-name fallback so
    they cluster with nothing else. Vectors are L2-normalised.
    """

    def __init__(self, substring_groups: dict[str, np.ndarray]):
        self.substring_groups = substring_groups
        self.calls: list[list[str]] = []

    def __call__(self, names):
        self.calls.append(list(names))
        dim = len(next(iter(self.substring_groups.values())))
        out = np.zeros((len(names), dim), dtype=np.float32)
        for i, raw in enumerate(names):
            # Strip the "clustering: " (or any) prefix the caller may have
            # added so substring tests can target the bare skill name.
            stripped = raw.split(": ", 1)[1] if ": " in raw else raw
            # Match on the no-whitespace version so "py torch" matches "pytorch".
            compact = "".join(stripped.split()).lower()
            for substr, vec in self.substring_groups.items():
                if substr in compact:
                    out[i] = vec
                    break
            else:
                fallback = np.zeros(dim, dtype=np.float32)
                fallback[hash(raw) % dim] = 1.0
                out[i] = fallback
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class TestNormalizeSkill:
    def test_lowercase_and_strip(self):
        assert normalize_skill("  PyTorch  ") == "pytorch"

    def test_collapses_whitespace(self):
        assert normalize_skill("Py\tTorch\n") == "py torch"

    def test_none_returns_empty(self):
        assert normalize_skill(None) == ""


class TestPickCanonicalName:
    def test_most_frequent_wins(self):
        counts = {"pytorch": 10, "py torch": 1}
        assert pick_canonical_name(["pytorch", "py torch"], counts) == "pytorch"

    def test_ties_broken_by_shortest(self):
        counts = {"pytorch": 3, "pytorch lib": 3}
        assert pick_canonical_name(list(counts), counts) == "pytorch"

    def test_alphabetical_final_tiebreak(self):
        counts = {"aws": 1, "gcp": 1}
        assert pick_canonical_name(list(counts), counts) == "aws"


class TestCollectRawCounts:
    def test_combines_posting_and_cv_names(self):
        counts = collect_raw_counts(["PyTorch", "pytorch"], ["PyTorch", "AWS"])
        assert counts["pytorch"] == 3
        assert counts["aws"] == 1

    def test_empties_dropped(self):
        counts = collect_raw_counts(["", None, "  "], ["valid"])  # type: ignore[list-item]
        assert counts == {"valid": 1}


class TestBuildCanonicalMap:
    def test_synonyms_cluster_together(self):
        groups = {
            "pytorch": np.array([1.0, 0.0, 0.0]),  # PyTorch family
            "react": np.array([0.0, 1.0, 0.0]),  # React family
        }
        embedder = DeterministicEmbedder(groups)
        counts = {"pytorch": 5, "py torch": 1, "react": 3, "react.js": 2}
        mapping = build_canonical_map(counts, embedder, distance_threshold=0.2)
        # "pytorch" group all map to the same canonical
        assert mapping["pytorch"] == mapping["py torch"]
        # "react" group all map to the same canonical
        assert mapping["react"] == mapping["react.js"]
        # The two groups are distinct
        assert mapping["pytorch"] != mapping["react"]
        # Most-frequent member is the canonical.
        assert mapping["pytorch"] == "pytorch"
        assert mapping["react"] == "react"

    def test_distant_strings_dont_merge(self):
        groups = {
            "pytorch": np.array([1.0, 0.0]),
            "react": np.array([0.0, 1.0]),
        }
        embedder = DeterministicEmbedder(groups)
        counts = {"pytorch": 1, "react": 1}
        mapping = build_canonical_map(counts, embedder, distance_threshold=0.2)
        assert mapping["pytorch"] != mapping["react"]

    def test_single_name_short_circuit(self):
        embedder = DeterministicEmbedder({"foo": np.array([1.0])})
        mapping = build_canonical_map({"foo": 1}, embedder)
        # Only one name → no clustering call needed.
        assert mapping == {"foo": "foo"}
        assert embedder.calls == []

    def test_empty_vocab_returns_empty(self):
        embedder = DeterministicEmbedder({"foo": np.array([1.0])})
        mapping = build_canonical_map({}, embedder)
        assert mapping == {}
        assert embedder.calls == []

    def test_prefix_applied(self):
        groups = {"pytorch": np.array([1.0, 0.0])}
        embedder = DeterministicEmbedder(groups)
        counts = {"pytorch": 1, "py torch": 1}
        build_canonical_map(counts, embedder, encode_prefix="clustering: ")
        # All embedder inputs should be prefixed with "clustering: ".
        for batch in embedder.calls:
            for s in batch:
                assert s.startswith("clustering: ")

    def test_embedder_row_count_mismatch_raises(self):
        class BadEmbedder:
            def __call__(self, names):
                return np.zeros((len(names) - 1, 3))

        with pytest.raises(ValueError, match="rows"):
            build_canonical_map({"a": 1, "b": 1}, BadEmbedder())


class _FixedVectorEmbedder:
    """Embedder that maps each post-prefix name to an explicit vector. Names
    not in the table get a unique fallback so they don't cluster with anyone.
    Output is L2-normalised."""

    def __init__(self, vectors_by_name: dict[str, np.ndarray]):
        self.vectors_by_name = vectors_by_name
        self.calls: list[list[str]] = []

    def __call__(self, names):
        self.calls.append(list(names))
        dim = len(next(iter(self.vectors_by_name.values())))
        out = np.zeros((len(names), dim), dtype=np.float32)
        for i, raw in enumerate(names):
            stripped = raw.split(": ", 1)[1] if ": " in raw else raw
            if stripped in self.vectors_by_name:
                out[i] = self.vectors_by_name[stripped]
            else:
                fb = np.zeros(dim, dtype=np.float32)
                fb[(hash(stripped) % (dim - 1)) + 1] = 1.0
                out[i] = fb
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class TestBuildCategoryMap:
    def test_coarse_threshold_collapses_concept_family(self):
        # All four AI-family names get vectors close to e0; the two front-end
        # names get vectors close to e1. Distance threshold (0.45) is loose
        # enough to merge each family, tight enough to keep them apart.
        vectors = {
            "ai":               np.array([1.00, 0.05, 0.0, 0.0]),
            "machine learning": np.array([0.99, 0.08, 0.0, 0.0]),
            "pytorch":          np.array([0.98, 0.12, 0.0, 0.0]),
            "mlops":            np.array([0.97, 0.15, 0.0, 0.0]),
            "react":            np.array([0.05, 1.00, 0.0, 0.0]),
            "vue":              np.array([0.08, 0.99, 0.0, 0.0]),
        }
        embedder = _FixedVectorEmbedder(vectors)
        canonical = list(vectors.keys())
        mapping = build_category_map(canonical, embedder, distance_threshold=0.45)
        ai_cat = mapping["ai"]
        assert mapping["machine learning"] == ai_cat
        assert mapping["pytorch"] == ai_cat
        assert mapping["mlops"] == ai_cat
        assert mapping["react"] == mapping["vue"]
        assert mapping["react"] != ai_cat

    def test_empty_vocab_returns_empty(self):
        embedder = _FixedVectorEmbedder({"foo": np.array([1.0, 0.0])})
        assert build_category_map([], embedder) == {}
        assert embedder.calls == []

    def test_single_name_short_circuit(self):
        embedder = _FixedVectorEmbedder({"foo": np.array([1.0, 0.0])})
        assert build_category_map(["pytorch"], embedder) == {"pytorch": "pytorch"}
        assert embedder.calls == []

    def test_label_picked_by_counts_when_provided(self):
        # When two names cluster, the highest-count wins.
        vectors = {
            "ai":                      np.array([1.0, 0.05]),
            "artificial intelligence": np.array([0.98, 0.10]),
        }
        embedder = _FixedVectorEmbedder(vectors)
        counts = {"ai": 50, "artificial intelligence": 5}
        mapping = build_category_map(
            ["ai", "artificial intelligence"],
            embedder,
            distance_threshold=0.45,
            counts=counts,
        )
        assert set(mapping.values()) == {"ai"}

    def test_prefix_is_applied_to_embedder_input(self):
        vectors = {"foo": np.array([1.0, 0.0]), "bar": np.array([0.0, 1.0])}
        embedder = _FixedVectorEmbedder(vectors)
        build_category_map(["foo", "bar"], embedder, encode_prefix="clustering: ")
        for batch in embedder.calls:
            for s in batch:
                assert s.startswith("clustering: ")

"""Frequency / co-occurrence / PMI / user_has math for the skill map."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from career_atlas import skill_map as sg
from career_atlas.pipelines.skill_map import nodes as sg_nodes
from career_atlas.schemas import POSTING_SKILLS_COLUMNS


@pytest.fixture
def posting_skills() -> pd.DataFrame:
    """4 postings × varying skill membership.

    p1: pytorch, cuda, mlops
    p2: pytorch, cuda
    p3: pytorch, mlops
    p4: react
    """
    rows = [
        ("p1", "pytorch", "tool"),
        ("p1", "cuda", "tool"),
        ("p1", "mlops", "skill"),
        ("p2", "pytorch", "tool"),
        ("p2", "cuda", "tool"),
        ("p3", "pytorch", "tool"),
        ("p3", "mlops", "skill"),
        ("p4", "react", "tool"),
    ]
    return pd.DataFrame(rows, columns=list(POSTING_SKILLS_COLUMNS))


@pytest.fixture
def canonical_map() -> dict[str, str]:
    # Identity map for these tests — normalised names ARE canonical.
    return {"pytorch": "pytorch", "cuda": "cuda", "mlops": "mlops", "react": "react"}


@pytest.fixture
def cv_profile() -> dict:
    return {
        "skills": [
            {"name": "PyTorch", "kind": "tool"},
            {"name": "MLOps", "kind": "skill"},
        ],
        "role_titles": ["ML Engineer"],
        "seniority": "senior",
        "years_experience": 5.0,
        "locations_preferred": ["Berlin"],
        "summary": "x",
    }


@pytest.fixture
def skill_map_params() -> dict:
    return {
        "map": {
            "min_node_count": 1,
        },
        "viz": {
            "tsne_perplexity": 2.0,  # small for tiny test fixtures
            "tsne_random_state": 42,
        },
    }


class _StubEmbedder:
    """Returns deterministic 8-dim vectors per name. Just enough for t-SNE
    not to choke on the tiny test vocabularies; we don't assert on the
    resulting (x, y) values."""

    def __call__(self, names):
        import numpy as _np

        rng = _np.random.default_rng(0)
        vecs = rng.normal(size=(len(names), 8))
        norms = _np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / _np.maximum(norms, 1e-9)


class TestNodeCounts:
    def test_each_skill_counted_once_per_posting(self, posting_skills, canonical_map):
        canonical_df = posting_skills.assign(canonical=posting_skills["name"])
        counts = sg.compute_node_counts(canonical_df)
        # pytorch in p1+p2+p3 = 3; cuda in p1+p2 = 2; mlops in p1+p3 = 2; react in p4 = 1
        assert counts == {"pytorch": 3, "cuda": 2, "mlops": 2, "react": 1}


class TestCooccurrence:
    def test_unordered_pairs_within_posting(self, posting_skills):
        canonical_df = posting_skills.assign(canonical=posting_skills["name"])
        co = sg.compute_cooccurrence(canonical_df)
        # p1 has 3 skills → C(3,2) = 3 pairs all once
        # p2 has 2 → 1 pair (pytorch,cuda) +1
        # p3 has 2 → 1 pair (pytorch,mlops) +1
        # p4 has 1 → 0 pairs
        assert co[("cuda", "pytorch")] == 2  # p1, p2
        assert co[("mlops", "pytorch")] == 2  # p1, p3
        assert co[("cuda", "mlops")] == 1  # p1 only


class TestPMI:
    def test_positive_when_above_chance(self):
        # 4 postings; pytorch in 3, cuda in 2, together in 2.
        # P(pytorch)=3/4, P(cuda)=2/4, P(both)=2/4
        # PMI = log( (2/4) / (3/4 * 2/4) ) = log( (1/2) / (3/8) ) = log(4/3)
        assert sg.pmi(2, 3, 2, 4) == pytest.approx(math.log(4 / 3))

    def test_zero_handling(self):
        assert sg.pmi(0, 5, 5, 100) == float("-inf")


class TestBuildGraph:
    def test_node_counts_match_handworked(
        self, posting_skills, canonical_map, skill_map_params
    ):
        # No CV, no embedder — just verify per-canonical-skill posting counts.
        graph = sg.build_map(
            posting_skills,
            canonical_map=canonical_map,
            cv_skill_names=[],
            **skill_map_params["map"],
        )
        ids = {n["id"]: n for n in graph["nodes"]}
        assert ids["pytorch"]["count"] == 3
        assert ids["cuda"]["count"] == 2
        assert ids["mlops"]["count"] == 2
        assert ids["react"]["count"] == 1
        # Semantic map drops the network metaphor; no edges expected.
        assert "edges" not in graph

    def test_user_has_marked_via_canonical_map(
        self, posting_skills, canonical_map, cv_profile, skill_map_params
    ):
        # Use cv_profile skills via the node so the canonical map participates.
        out = sg_nodes.build_skill_map(
            posting_skills, canonical_map, {}, cv_profile, skill_map_params
        )
        ids = {n["id"]: n for n in out["nodes"]}
        assert ids["pytorch"]["user_has"] is True
        assert ids["mlops"]["user_has"] is True
        assert ids["cuda"]["user_has"] is False
        assert ids["react"]["user_has"] is False

    def test_min_node_count_filter_keeps_user_skills(
        self, posting_skills, canonical_map, cv_profile
    ):
        # react only appears once. With min_node_count=2 it should be dropped.
        # User does NOT have react, so it actually IS dropped.
        params = {"map": {"min_node_count": 2}}
        out = sg_nodes.build_skill_map(
            posting_skills, canonical_map, {}, cv_profile, params
        )
        ids = {n["id"] for n in out["nodes"]}
        assert "react" not in ids

    def test_user_skill_below_threshold_is_kept(
        self, posting_skills, canonical_map
    ):
        # If the user owns react (only 1 posting), it should be kept even at
        # min_node_count=5 because user_has overrides the filter.
        cv = {
            "skills": [{"name": "react", "kind": "tool"}],
            "role_titles": ["x"],
        }
        params = {"map": {"min_node_count": 5}}
        out = sg_nodes.build_skill_map(
            posting_skills, canonical_map, {}, cv, params
        )
        ids = {n["id"]: n for n in out["nodes"]}
        assert ids.get("react", {}).get("user_has") is True

    def test_nodes_carry_position_when_embedder_provided(
        self, posting_skills, canonical_map, cv_profile, skill_map_params
    ):
        out = sg_nodes.build_skill_map(
            posting_skills,
            canonical_map,
            {},
            cv_profile,
            skill_map_params,
            skill_embedder=_StubEmbedder(),
        )
        for n in out["nodes"]:
            assert "position" in n
            assert isinstance(n["position"]["x"], float)
            assert isinstance(n["position"]["y"], float)

    def test_no_positions_when_embedder_missing(
        self, posting_skills, canonical_map, cv_profile, skill_map_params
    ):
        # build_skill_map(skill_embedder=None) is the CLI default before the
        # hook fires — t-SNE is skipped entirely, no position keys appear.
        out = sg_nodes.build_skill_map(
            posting_skills, canonical_map, {}, cv_profile, skill_map_params
        )
        for n in out["nodes"]:
            assert "position" not in n

    def test_category_map_threaded_into_nodes(
        self, posting_skills, canonical_map, cv_profile, skill_map_params
    ):
        category_map = {
            "pytorch": "ai",
            "cuda": "ai",
            "mlops": "ai",
            "react": "frontend",
        }
        out = sg_nodes.build_skill_map(
            posting_skills, canonical_map, category_map, cv_profile, skill_map_params
        )
        ids = {n["id"]: n for n in out["nodes"]}
        assert ids["pytorch"]["category"] == "ai"
        assert ids["mlops"]["category"] == "ai"
        assert ids["react"]["category"] == "frontend"

    def test_empty_category_map_omits_field(
        self, posting_skills, canonical_map, cv_profile, skill_map_params
    ):
        # `{}` is treated as "no categorisation" by build_skill_map (it passes
        # None to build_map). Nodes should not carry a `category` key.
        out = sg_nodes.build_skill_map(
            posting_skills, canonical_map, {}, cv_profile, skill_map_params
        )
        for n in out["nodes"]:
            assert "category" not in n

    def test_empty_posting_skills_safe(self, canonical_map, cv_profile, skill_map_params):
        empty = pd.DataFrame(columns=list(POSTING_SKILLS_COLUMNS))
        out = sg_nodes.build_skill_map(
            empty, canonical_map, {}, cv_profile, skill_map_params
        )
        assert out["nodes"] == []
        assert "edges" not in out
        assert out["n_postings"] == 0

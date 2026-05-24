"""Frequency / co-occurrence / PMI / user_has math for the skill graph."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from job_universe import skill_graph as sg
from job_universe.pipelines.skill_graph import nodes as sg_nodes
from job_universe.schemas import POSTING_SKILLS_COLUMNS


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
def skill_graph_params() -> dict:
    return {
        "graph": {
            "min_node_count": 1,
            "min_cooccurrence": 1,
            "min_pmi": -10.0,  # keep everything for math assertions
        }
    }


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
    def test_node_counts_and_edges_match_handworked(
        self, posting_skills, canonical_map, skill_graph_params
    ):
        # No CV influence here; pass empty CV skills.
        graph = sg.build_graph(
            posting_skills,
            canonical_map=canonical_map,
            cv_skill_names=[],
            **skill_graph_params["graph"],
        )
        ids = {n["id"]: n for n in graph["nodes"]}
        assert ids["pytorch"]["count"] == 3
        assert ids["cuda"]["count"] == 2
        assert ids["mlops"]["count"] == 2
        assert ids["react"]["count"] == 1
        edges = {tuple(sorted([e["source"], e["target"]])): e for e in graph["edges"]}
        assert edges[("cuda", "pytorch")]["cooccurrence"] == 2
        assert edges[("mlops", "pytorch")]["cooccurrence"] == 2

    def test_user_has_marked_via_canonical_map(
        self, posting_skills, canonical_map, cv_profile, skill_graph_params
    ):
        # Use cv_profile skills via the node so the canonical map participates.
        out = sg_nodes.build_skill_graph(
            posting_skills, canonical_map, {}, cv_profile, skill_graph_params
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
        # But user does NOT have react, so it actually IS dropped.
        params = {"graph": {"min_node_count": 2, "min_cooccurrence": 1, "min_pmi": -10.0}}
        out = sg_nodes.build_skill_graph(
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
        params = {"graph": {"min_node_count": 5, "min_cooccurrence": 999, "min_pmi": 0}}
        out = sg_nodes.build_skill_graph(
            posting_skills, canonical_map, {}, cv, params
        )
        ids = {n["id"]: n for n in out["nodes"]}
        assert ids.get("react", {}).get("user_has") is True

    def test_edge_filter_min_pmi(
        self, posting_skills, canonical_map, cv_profile
    ):
        # Set a punishingly high PMI cutoff — no edges should survive.
        params = {"graph": {"min_node_count": 1, "min_cooccurrence": 1, "min_pmi": 99.0}}
        out = sg_nodes.build_skill_graph(
            posting_skills, canonical_map, {}, cv_profile, params
        )
        assert out["edges"] == []

    def test_nodes_carry_position_after_build(
        self, posting_skills, canonical_map, cv_profile, skill_graph_params
    ):
        out = sg_nodes.build_skill_graph(
            posting_skills, canonical_map, {}, cv_profile, skill_graph_params
        )
        for n in out["nodes"]:
            assert "position" in n
            assert isinstance(n["position"]["x"], float)
            assert isinstance(n["position"]["y"], float)

    def test_category_map_threaded_into_nodes(
        self, posting_skills, canonical_map, cv_profile, skill_graph_params
    ):
        category_map = {
            "pytorch": "ai",
            "cuda": "ai",
            "mlops": "ai",
            "react": "frontend",
        }
        out = sg_nodes.build_skill_graph(
            posting_skills, canonical_map, category_map, cv_profile, skill_graph_params
        )
        ids = {n["id"]: n for n in out["nodes"]}
        assert ids["pytorch"]["category"] == "ai"
        assert ids["mlops"]["category"] == "ai"
        assert ids["react"]["category"] == "frontend"

    def test_empty_category_map_omits_field(
        self, posting_skills, canonical_map, cv_profile, skill_graph_params
    ):
        # `{}` is treated as "no categorisation" by build_skill_graph (it passes
        # None to build_graph). Nodes should not carry a `category` key.
        out = sg_nodes.build_skill_graph(
            posting_skills, canonical_map, {}, cv_profile, skill_graph_params
        )
        for n in out["nodes"]:
            assert "category" not in n

    def test_empty_posting_skills_safe(self, canonical_map, cv_profile, skill_graph_params):
        empty = pd.DataFrame(columns=list(POSTING_SKILLS_COLUMNS))
        out = sg_nodes.build_skill_graph(
            empty, canonical_map, {}, cv_profile, skill_graph_params
        )
        assert out["nodes"] == []
        assert out["edges"] == []
        assert out["n_postings"] == 0

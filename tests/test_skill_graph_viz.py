"""Smoke tests for the matplotlib rendering node."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must come before pyplot import

import matplotlib.pyplot as plt
import pytest

from job_universe import skill_graph as sg


@pytest.fixture
def small_graph() -> dict:
    return {
        "nodes": [
            {"id": "pytorch", "count": 10, "user_has": True},
            {"id": "cuda", "count": 8, "user_has": False},
            {"id": "mlops", "count": 6, "user_has": True},
            {"id": "react", "count": 5, "user_has": False},
            {"id": "kubernetes", "count": 4, "user_has": False},
        ],
        "edges": [
            {"source": "pytorch", "target": "cuda", "weight": 2.1, "cooccurrence": 8},
            {"source": "pytorch", "target": "mlops", "weight": 1.8, "cooccurrence": 5},
            {"source": "cuda", "target": "mlops", "weight": 0.9, "cooccurrence": 3},
        ],
        "n_postings": 20,
    }


@pytest.fixture
def viz_params() -> dict:
    return {
        "background_color": "#000000",
        "node_color_default": "#888888",
        "node_color_user_has": "#3F704D",
        "edge_color": "#333333",
        "edge_alpha": 0.3,
        "node_alpha": 0.85,
        "label_top_n": 3,
        "figsize": [6, 6],
        "layout_seed": 42,
        "spring_k": 0.7,
        "spring_iterations": 50,
        "node_size_scale": 40,
        "label_font_size": 8,
        "label_color": "#FFFFFF",
    }


class TestRenderSkillGraphFigure:
    def test_returns_a_figure(self, small_graph, viz_params):
        fig = sg.render_skill_graph_figure(small_graph, viz_params)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_background_color_applied(self, small_graph, viz_params):
        fig = sg.render_skill_graph_figure(small_graph, viz_params)
        try:
            assert fig.patch.get_facecolor()[:3] == (0.0, 0.0, 0.0)
            ax = fig.axes[0]
            assert ax.get_facecolor()[:3] == (0.0, 0.0, 0.0)
        finally:
            plt.close(fig)

    def test_axis_is_hidden(self, small_graph, viz_params):
        fig = sg.render_skill_graph_figure(small_graph, viz_params)
        try:
            ax = fig.axes[0]
            # ax.axis("off") sets axison=False; ticks may still be cached.
            assert ax.axison is False
        finally:
            plt.close(fig)

    def test_empty_graph_renders_placeholder(self, viz_params):
        fig = sg.render_skill_graph_figure({"nodes": [], "edges": []}, viz_params)
        try:
            assert isinstance(fig, plt.Figure)
            ax = fig.axes[0]
            assert any("No skills" in t.get_text() for t in ax.texts)
        finally:
            plt.close(fig)


class TestNodesDataframe:
    def test_sorted_by_count_descending(self, small_graph):
        df = sg.nodes_dataframe(small_graph)
        assert list(df["count"]) == sorted(df["count"], reverse=True)
        # First row is the highest-count node.
        assert df.iloc[0]["id"] == "pytorch"

    def test_empty_graph(self):
        df = sg.nodes_dataframe({"nodes": [], "edges": []})
        assert df.empty
        assert list(df.columns) == ["id", "count", "user_has"]

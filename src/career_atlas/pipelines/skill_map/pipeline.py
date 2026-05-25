"""Stage 3 pipeline wiring: extract → canonicalize → build → render."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from . import nodes


def create_pipeline(**_kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=nodes.extract_posting_skills,
                inputs=[
                    "postings_cumulative_existing",
                    "posting_skills_existing",
                    "params:cv_extraction",
                    "params:skill_map",
                    "ollama_client",
                    "progress_emitter",
                ],
                outputs="posting_skills",
                name="extract_posting_skills",
            ),
            node(
                func=nodes.canonicalize_skills,
                inputs=[
                    "posting_skills",
                    "cv_profile",
                    "params:skill_map",
                    "skill_embedder",
                ],
                outputs="canonical_skill_map",
                name="canonicalize_skills",
            ),
            node(
                func=nodes.assign_skill_categories,
                inputs=[
                    "canonical_skill_map",
                    "params:skill_map",
                    "skill_embedder",
                ],
                outputs="skill_category_map",
                name="assign_skill_categories",
            ),
            node(
                func=nodes.build_skill_map,
                inputs=[
                    "posting_skills",
                    "canonical_skill_map",
                    "skill_category_map",
                    "cv_profile",
                    "params:skill_map",
                    "skill_embedder",
                ],
                outputs="skill_map_data",
                name="build_skill_map",
            ),
            node(
                func=nodes.render_skill_map_png,
                inputs=["skill_map_data", "params:skill_map"],
                outputs=["skill_map_png", "skill_map_nodes_csv"],
                name="render_skill_map_png",
            ),
        ]
    )

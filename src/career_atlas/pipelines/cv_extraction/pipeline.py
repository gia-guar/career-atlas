"""CV extraction pipeline wiring."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from . import nodes


def create_pipeline(**_kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=nodes.extract_cv_profile,
                inputs=["cv_raw_text", "params:cv_extraction", "ollama_client"],
                outputs="cv_profile",
                name="extract_cv_profile",
            ),
            node(
                func=nodes.derive_targeted_scraping_params,
                inputs=[
                    "cv_profile",
                    "params:cv_extraction",
                    "params:scraping",
                    "ollama_client",
                ],
                outputs="cv_derived_scraping_params",
                name="derive_targeted_scraping_params",
            ),
        ]
    )

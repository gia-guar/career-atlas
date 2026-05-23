"""Scraping pipeline wiring + identity of node functions with the shared module."""

from __future__ import annotations

from job_universe import scraping
from job_universe.pipelines.cv_extraction import create_pipeline as cv_pipeline
from job_universe.pipelines.scraping import create_pipeline as scraping_pipeline


def _node_by_name(pipeline, name):
    for n in pipeline.nodes:
        if n.name == name:
            return n
    raise AssertionError(f"node {name!r} not found in pipeline")


class TestPipelineWiring:
    def test_scraping_pipeline_has_six_nodes(self):
        p = scraping_pipeline()
        assert len(p.nodes) == 6

    def test_consumes_cv_derived_scraping_params_only(self):
        p = scraping_pipeline()
        fetch_adz = _node_by_name(p, "fetch_adzuna")
        assert "cv_derived_scraping_params" in fetch_adz.inputs
        # There is no `params:scraping.queries` injection — all queries / countries
        # / locations come from the CV-derived params.
        all_inputs = set()
        for n in p.nodes:
            all_inputs.update(n.inputs)
        assert "params:scraping" not in all_inputs

    def test_writes_to_cumulative_store(self):
        p = scraping_pipeline()
        update = _node_by_name(p, "update_cumulative")
        assert update.outputs == ["postings_cumulative"]
        assert "postings_cumulative_existing" in update.inputs

    def test_node_funcs_come_from_shared_module(self):
        p = scraping_pipeline()
        funcs = {n.name: n.func for n in p.nodes}
        assert funcs["fetch_adzuna"] is scraping.fetch_adzuna
        assert funcs["fetch_jobspy"] is scraping.fetch_jobspy
        assert funcs["normalize_adzuna"] is scraping.normalize_adzuna
        assert funcs["normalize_jobspy"] is scraping.normalize_jobspy
        assert funcs["merge_and_dedupe"] is scraping.merge_and_dedupe
        assert funcs["update_cumulative"] is scraping.update_cumulative

    def test_cv_pipeline_handoff_to_scraping(self):
        cv_p = cv_pipeline()
        s_p = scraping_pipeline()
        cv_outputs = set()
        for n in cv_p.nodes:
            cv_outputs.update(n.outputs)
        s_inputs = set()
        for n in s_p.nodes:
            s_inputs.update(n.inputs)
        # The handoff dataset is produced by cv_extraction and consumed by scraping.
        assert "cv_derived_scraping_params" in cv_outputs
        assert "cv_derived_scraping_params" in s_inputs

"""CV extraction pipeline: CV text → CVProfile → targeted scraping params."""

from job_universe.pipelines.cv_extraction.pipeline import create_pipeline

__all__ = ["create_pipeline"]

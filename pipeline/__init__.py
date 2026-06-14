"""Runnable pipeline — config-driven assembly of the detection cycle.

    from pipeline import run_pipeline, load_config
    run_pipeline(load_config("pipeline.yaml"))
"""
from .runner import build_detector, build_engine, load_config, run_pipeline

__all__ = ["run_pipeline", "load_config", "build_detector", "build_engine"]

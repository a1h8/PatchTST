"""Beam runner selection — build ``PipelineOptions`` for a target runner.

Keeps runner choice out of the engine and the connectors: the pipeline config
names a runner by alias (``direct`` / ``dataflow`` / ``flink``) and passes a
small dict of runner-specific settings (``project``, ``region``,
``temp_location`` for Dataflow; ``flink_master`` for Flink; ...). ``apache_beam``
is imported lazily so the agnostic core never needs it.

M6 order (decision): **Dataflow first** (managed, least ops), then
**Flink-on-K8s** (portable runner + job server on the existing k3s cluster). Both
run the *same* ``BeamEngine`` pipeline — only these options change.
"""
from __future__ import annotations

from typing import Any, Optional

# Alias -> Beam runner class. Kept small and explicit; unknown aliases fail fast
# rather than being forwarded to Beam as an opaque string.
_RUNNER_ALIASES = {
    "direct": "DirectRunner",
    "dataflow": "DataflowRunner",
    # FlinkRunner boots its own job server from `flink_master` (dev / embedded).
    "flink": "FlinkRunner",
    # PortableRunner talks to a standalone Beam job server via `job_endpoint` —
    # the k8s-native path (job server + Flink cluster deployed separately).
    "portable": "PortableRunner",
}


def beam_pipeline_options(
    runner: str = "direct",
    *,
    streaming: bool = False,
    options: Optional[dict[str, Any]] = None,
):
    """Translate a runner alias + settings dict into Beam ``PipelineOptions``.

    Runner-specific keys are flattened into Beam's ``--flag=value`` argv form so
    each is parsed by whichever ``PipelineOptions`` view owns it (e.g.
    ``project``/``region``/``temp_location`` land on ``GoogleCloudOptions``). The
    runner alias and ``streaming`` flag are then set explicitly on
    ``StandardOptions``.
    """
    from apache_beam.options.pipeline_options import (
        PipelineOptions,
        StandardOptions,
    )

    try:
        runner_name = _RUNNER_ALIASES[runner]
    except KeyError:
        raise KeyError(
            f"unknown runner {runner!r}; available: {sorted(_RUNNER_ALIASES)}"
        ) from None

    # Flatten settings into --flag=value. A list value emits the flag once per
    # item, matching Beam's repeatable flags (e.g. experiments, sdk_harness
    # extra packages). A bool emits --flag=true/false.
    argv: list[str] = []
    for key, value in (options or {}).items():
        items = value if isinstance(value, (list, tuple)) else [value]
        for item in items:
            argv.append(f"--{key}={item}")
    opts = PipelineOptions(argv)
    opts.view_as(StandardOptions).runner = runner_name
    opts.view_as(StandardOptions).streaming = streaming
    return opts
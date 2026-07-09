"""Pre-flight validation for a real remote submit (Dataflow-first, M6).

A live Dataflow (or portable/Flink) submit is slow and billable; the failures
that bite on the *first* run are boring and catchable offline:

- the launch-time ``apache-beam`` version drifts from the SDK worker image tag
  (``requirements-connectors.txt`` pins a floor ``>=2.70``, but the image is an
  exact ``:2.74.0``) → the runner rejects the workers;
- a required Dataflow option is missing, or ``temp_location`` / a sink ``root``
  points at a worker-local path the distributed workers cannot reach;
- a streaming graph with no window, or a streaming Dataflow submit that would
  block the driver forever (see ``BeamEngine(block=...)``).

``preflight(config)`` runs these checks with **no infra access** — it never
contacts GCP, never reads a metric, never submits — and returns a list of
``Issue``. ``python -m pipeline --check <config>`` prints them and exits
non-zero on any ``error`` so CI / a human can gate the submit; a real submit
runs the same checks first and aborts on error.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# Runners whose workers run remotely from a custom SDK image, so the launch-time
# beam version and the object-store paths actually matter. DirectRunner is
# in-process and local paths are fine, so it is exempt from most checks.
_REMOTE_RUNNERS = {"dataflow", "flink", "portable"}
# Options Dataflow needs to schedule the job; missing any is a hard error.
_DATAFLOW_REQUIRED = (
    "project",
    "region",
    "temp_location",
    "staging_location",
    "sdk_container_image",
)
# Object-store URI schemes a distributed worker can actually reach.
_OBJECT_STORE_SCHEMES = ("gs://", "s3://")
# Where the worker image's beam version is pinned (base tag of the SDK image).
_DOCKERFILE = Path("deploy/dataflow/Dockerfile")

Level = Literal["error", "warn"]


@dataclass(frozen=True)
class Issue:
    """A single pre-flight finding. ``error`` blocks a submit; ``warn`` informs."""

    level: Level
    code: str
    message: str

    def __str__(self) -> str:  # rendered by the CLI
        mark = "✗" if self.level == "error" else "!"
        return f"  {mark} [{self.code}] {self.message}"


def _tag(image: str) -> str | None:
    """Return the tag of a container ref (``repo/name:TAG``), or None.

    Splits on the last ``:`` but ignores a ``:`` that belongs to a registry
    ``host:port`` — a real tag has no ``/`` after it.
    """
    _, sep, tail = image.rpartition(":")
    if not sep or "/" in tail:
        return None
    return tail or None


def _dockerfile_base_tag(root: Path) -> str | None:
    """Parse the ``FROM apache/beam_python...:TAG`` base tag from the Dockerfile."""
    path = root / _DOCKERFILE
    try:
        text = path.read_text()
    except OSError:
        return None
    m = re.search(r"^FROM\s+\S*beam\S*:(\S+)", text, re.MULTILINE)
    return m.group(1) if m else None


def _installed_beam_version() -> str | None:
    try:
        import apache_beam

        return apache_beam.__version__
    except Exception:  # pragma: no cover - beam always present where this runs
        return None


def _check_version_alignment(
    options: dict[str, Any], issues: list[Issue], *, root: Path
) -> None:
    """Installed beam == sdk_container_image tag == Dockerfile base tag.

    A mismatch between the launch-time SDK and the worker image is the classic
    first-submit failure: Dataflow refuses workers whose SDK differs from the
    submitting client.
    """
    installed = _installed_beam_version()
    image = options.get("sdk_container_image")
    image_tag = _tag(image) if isinstance(image, str) else None
    base_tag = _dockerfile_base_tag(root)

    if installed and image_tag and installed != image_tag:
        issues.append(
            Issue(
                "error",
                "beam-version-mismatch",
                f"installed apache-beam {installed} != sdk_container_image tag "
                f"{image_tag!r}; rebuild/point the worker image at {installed} "
                f"(Dataflow rejects workers whose SDK differs from the launcher)",
            )
        )
    if installed and base_tag and installed != base_tag:
        issues.append(
            Issue(
                "warn",
                "beam-dockerfile-drift",
                f"installed apache-beam {installed} != {_DOCKERFILE} base tag "
                f"{base_tag!r}; rebuild the worker image from a matching base "
                f"and bump requirements-connectors.txt off the '>=' floor",
            )
        )


def _check_dataflow_options(options: dict[str, Any], issues: list[Issue]) -> None:
    for key in _DATAFLOW_REQUIRED:
        if not options.get(key):
            issues.append(
                Issue(
                    "error",
                    "missing-option",
                    f"engine.options.{key} is required for the dataflow runner",
                )
            )
    for key in ("temp_location", "staging_location"):
        val = options.get(key)
        if isinstance(val, str) and not val.startswith("gs://"):
            issues.append(
                Issue(
                    "error",
                    "non-gcs-location",
                    f"engine.options.{key}={val!r} must be a gs:// path "
                    f"(Dataflow workers cannot reach a local path)",
                )
            )
    experiments = options.get("experiments") or []
    if isinstance(experiments, str):
        experiments = [experiments]
    if options.get("sdk_container_image") and "use_runner_v2" not in experiments:
        issues.append(
            Issue(
                "warn",
                "runner-v2-recommended",
                "a custom sdk_container_image usually needs experiments: "
                "[use_runner_v2]; add it unless you know the legacy path applies",
            )
        )


def _check_sinks_remote_reachable(config: dict, issues: list[Issue]) -> None:
    """A distributed worker must write to an object store, not a local path."""
    for sink in config.get("sinks", []):
        params = sink.get("params", {}) or {}
        root = params.get("root")
        if isinstance(root, str) and not root.startswith(_OBJECT_STORE_SCHEMES):
            issues.append(
                Issue(
                    "error",
                    "local-sink-root",
                    f"sink {sink.get('type')!r} root={root!r} is not an object "
                    f"store ({' / '.join(_OBJECT_STORE_SCHEMES)}); remote workers "
                    f"cannot share a local filesystem",
                )
            )


def _check_streaming(engine: dict, options: dict, issues: list[Issue]) -> None:
    streaming = bool(engine.get("streaming", False))
    if streaming and not engine.get("window"):
        issues.append(
            Issue(
                "error",
                "streaming-no-window",
                "engine.streaming is true but no engine.window is set; an "
                "unbounded source needs a WindowInto to ever emit",
            )
        )
    # A streaming Dataflow job runs until drained; if the driver blocks on it the
    # submit never returns. block must be explicitly false to detach.
    if streaming and engine.get("block", True):
        issues.append(
            Issue(
                "warn",
                "streaming-submit-blocks",
                "engine.block is not false for a streaming submit; the driver "
                "will wait_until_finish() and hang on the long-running job. Set "
                "engine.block: false to submit-and-detach",
            )
        )


def _check_construction(config: dict, issues: list[Issue]) -> None:
    """Dry-build source/detector/sinks/engine — validates registry names and
    params without reading a source or submitting a graph."""
    from connectors import build

    from .runner import build_detector, build_engine

    try:
        build(config["source"]["type"], **config["source"].get("params", {}))
    except Exception as exc:  # noqa: BLE001 - surfaced as an issue, not a crash
        issues.append(Issue("error", "source-build", f"source: {exc}"))
    for sink in config.get("sinks", []):
        try:
            build(sink["type"], **sink.get("params", {}))
        except Exception as exc:  # noqa: BLE001
            issues.append(Issue("error", "sink-build", f"sink {sink.get('type')!r}: {exc}"))
    try:
        build_detector(config["detector"])
    except Exception as exc:  # noqa: BLE001
        issues.append(Issue("error", "detector-build", f"detector: {exc}"))
    try:
        build_engine(config.get("engine"))
    except Exception as exc:  # noqa: BLE001
        issues.append(Issue("error", "engine-build", f"engine: {exc}"))


def preflight(config: dict, *, root: Path | None = None) -> list[Issue]:
    """Return pre-flight issues for ``config`` without touching any infra.

    ``root`` is the repo root used to locate ``deploy/dataflow/Dockerfile`` for
    the version cross-check (defaults to the current working directory).
    """
    root = root or Path.cwd()
    issues: list[Issue] = []
    engine = config.get("engine") or {}
    runner = engine.get("runner", "direct") if engine.get("type") == "beam" else "local"
    options = engine.get("options") or {}

    _check_construction(config, issues)

    if runner not in _REMOTE_RUNNERS:
        return issues  # local / DirectRunner: nothing infra-specific to gate

    _check_streaming(engine, options, issues)
    _check_sinks_remote_reachable(config, issues)
    if runner == "dataflow":
        _check_dataflow_options(options, issues)
        _check_version_alignment(options, issues, root=root)
    return issues


def has_errors(issues: list[Issue]) -> bool:
    return any(i.level == "error" for i in issues)


def format_report(issues: list[Issue]) -> str:
    if not issues:
        return "preflight: OK — no issues"
    errors = sum(i.level == "error" for i in issues)
    warns = len(issues) - errors
    lines = [f"preflight: {errors} error(s), {warns} warning(s)"]
    # Errors first, warnings after; stable within each level.
    ordered = sorted(issues, key=lambda i: 0 if i.level == "error" else 1)
    lines += [str(i) for i in ordered]
    return "\n".join(lines)
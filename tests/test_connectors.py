"""Connector SPI tests — pure-Python, no Beam install required.

Covers: pivot validation, internal registry, the conformance contract for the
built-in connectors, and the multivariate alignment that backs decision D4.
"""
import pytest

import connectors  # noqa: F401  (triggers built-in registration)
from connectors.alignment import align_group
from connectors.conformance import (
    assert_buildable,
    assert_registered,
    assert_sink_contract,
    assert_source_contract,
)
from connectors.pivot import PivotRow
from connectors.registry import available
from connectors.sources.mimir import to_pivot_rows


# --- pivot schema ---------------------------------------------------------

def test_pivot_valid():
    row = PivotRow("pod-a", 1000, (1.0, 2.0), ("cpu", "mem"))
    assert row.width == 2


def test_pivot_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        PivotRow("pod-a", 1000, (1.0,), ("cpu", "mem"))


def test_pivot_rejects_float_ts():
    with pytest.raises(ValueError, match="ts must be int"):
        PivotRow("pod-a", 1000.0, (1.0,), ("cpu",))  # type: ignore[arg-type]


def test_pivot_rejects_duplicate_channels():
    with pytest.raises(ValueError, match="duplicate channels"):
        PivotRow("pod-a", 1000, (1.0, 2.0), ("cpu", "cpu"))


# --- registry -------------------------------------------------------------

def test_builtins_registered():
    reg = available()
    assert reg.get("mimir") == "source"
    assert reg.get("parquet") == "sink"


def test_build_unknown_raises():
    with pytest.raises(KeyError, match="unknown connector"):
        assert_buildable("does-not-exist")


# --- conformance ----------------------------------------------------------

def test_mimir_conforms_to_source_contract():
    assert_registered("mimir")
    src = assert_buildable(
        "mimir", endpoint="http://mimir:9009", promql="up", start=0, end=10
    )
    assert_source_contract(src)


def test_parquet_conforms_to_sink_contract():
    assert_registered("parquet")
    sink = assert_buildable("parquet", path="/tmp/out")
    assert_sink_contract(sink)


# --- multivariate alignment (D4) -----------------------------------------

def test_align_group_ffill_across_cadences():
    # cpu @ 15s, mem @ 30s on a 15s grid -> mem forward-filled into the gap.
    rows = align_group(
        "pod-a",
        {
            "cpu": [(0, 1.0), (15_000, 1.1), (30_000, 1.2)],
            "mem": [(0, 5.0), (30_000, 5.5)],
        },
        step_ms=15_000,
        fill="ffill",
    )
    assert [r.ts for r in rows] == [0, 15_000, 30_000]
    assert rows[0].channels == ("cpu", "mem")
    assert rows[1].values == (1.1, 5.0)   # mem carried forward
    assert rows[2].values == (1.2, 5.5)


def test_align_group_drop_incomplete():
    rows = align_group(
        "pod-a",
        {"cpu": [(0, 1.0), (15_000, 1.1)], "mem": [(15_000, 5.0)]},
        step_ms=15_000,
        fill="drop",
    )
    # grid bucket 0 has no mem -> dropped; only bucket 15000 is complete.
    assert [r.ts for r in rows] == [15_000]


def test_mimir_to_pivot_rows_groups_and_aligns():
    result = [
        {
            "metric": {"__name__": "cpu", "instance": "pod-a"},
            "values": [[0, "1.0"], [15, "1.1"]],
        },
        {
            "metric": {"__name__": "mem", "instance": "pod-a"},
            "values": [[0, "5.0"], [15, "5.5"]],
        },
    ]
    rows = to_pivot_rows(
        result,
        group_by=["instance"],
        channel_label="__name__",
        step_ms=15_000,
        fill="ffill",
    )
    assert {r.group_id for r in rows} == {"pod-a"}
    assert rows[0].channels == ("cpu", "mem")
    assert rows[0].values == (1.0, 5.0)
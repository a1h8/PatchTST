"""Kafka source (connector C7) tests.

``kafka-python`` is an optional dependency (only needed for a live ``read()``),
so it's stubbed via ``sys.modules`` the same way ``test_ingest.py`` stubs
``snappy`` — no real broker involved. The Beam cross-language path
(``native_beam_read``) is exercised only where ``apache_beam`` is actually
installed (CI's ``requirements-connectors.txt``); constructing ``ReadFromKafka``
does not start the expansion service (that happens in ``expand()``, never
called here), so it's safe to build without a JVM/Docker available.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

import connectors  # noqa: F401  (triggers built-in registration)
from connectors.conformance import assert_registered
from connectors.pivot import PivotRow
from connectors.registry import available, build
from connectors.sources.kafka import KafkaSource, _row_from_json

# --- registry ---------------------------------------------------------------

def test_kafka_registered():
    assert available().get("kafka") == "source"


def test_kafka_buildable():
    # Full conformance (incl. native_beam_read) is checked separately, gated on
    # apache_beam being installed; here we just confirm registry wiring.
    assert_registered("kafka")
    src = build("kafka", bootstrap_servers="b1:9092,b2:9092", topic="t")
    assert isinstance(src, KafkaSource)
    assert src.bootstrap_servers == ["b1:9092", "b2:9092"]


def test_kafka_accepts_list_bootstrap_servers():
    src = build("kafka", bootstrap_servers=["b1:9092", "b2:9092"], topic="t")
    assert src.bootstrap_servers == ["b1:9092", "b2:9092"]


def test_kafka_describe():
    src = KafkaSource("b1:9092", "t", group_id="g1")
    d = src.describe()
    assert d["kind"] == "source"
    assert d["topic"] == "t" and d["group_id"] == "g1"
    assert d["bootstrap_servers"] == ["b1:9092"]


# --- wire format --------------------------------------------------------------

def test_row_from_json_roundtrip():
    row = PivotRow("pod-a", 1000, (1.0, 2.0), ("cpu", "mem"), {"ns": "default"})
    payload = json.dumps(
        {
            "group_id": row.group_id,
            "ts": row.ts,
            "values": list(row.values),
            "channels": list(row.channels),
            "labels": row.labels,
        }
    )
    assert _row_from_json(payload) == row


def test_row_from_json_defaults_missing_labels():
    payload = json.dumps({"group_id": "g", "ts": 0, "values": [1.0], "channels": ["a"]})
    row = _row_from_json(payload)
    assert row.labels == {}


# --- read() (bounded poll, kafka-python stubbed) -----------------------------

class _FakeMessage:
    def __init__(self, value: str) -> None:
        self.value = value


def _install_fake_kafka(monkeypatch, messages, captured):
    class _FakeKafkaConsumer:
        def __init__(self, topic, **kwargs):
            captured["topic"] = topic
            captured["kwargs"] = kwargs
            self._messages = messages
            self.closed = False

        def __iter__(self):
            return iter(self._messages)

        def close(self):
            self.closed = True
            captured["closed"] = True

    fake_kafka = types.ModuleType("kafka")
    fake_kafka.KafkaConsumer = _FakeKafkaConsumer
    monkeypatch.setitem(sys.modules, "kafka", fake_kafka)
    return _FakeKafkaConsumer


def test_read_drains_available_messages(monkeypatch):
    captured: dict = {}
    row = PivotRow("pod-a", 1000, (1.0,), ("cpu",))
    payload = json.dumps(
        {"group_id": row.group_id, "ts": row.ts, "values": list(row.values),
         "channels": list(row.channels)}
    )
    _install_fake_kafka(monkeypatch, [_FakeMessage(payload)], captured)

    src = KafkaSource("b1:9092", "t", group_id="g1", auto_offset_reset="earliest")
    rows = src.read()

    assert rows == [row]
    assert captured["topic"] == "t"
    assert captured["kwargs"]["bootstrap_servers"] == ["b1:9092"]
    assert captured["kwargs"]["group_id"] == "g1"
    assert captured["kwargs"]["auto_offset_reset"] == "earliest"
    assert captured["closed"] is True


def test_read_empty_topic_returns_empty_list(monkeypatch):
    _install_fake_kafka(monkeypatch, [], {})
    src = KafkaSource("b1:9092", "t")
    assert src.read() == []


# --- native_beam_read (requires apache_beam; construction only) -------------

def test_native_beam_read_returns_composite_transform():
    beam = pytest.importorskip("apache_beam")

    src = KafkaSource(["b1:9092"], "t", group_id="g1")
    transform = src.native_beam_read()

    assert hasattr(transform, "expand")
    assert isinstance(transform, beam.PTransform)


def test_kafka_conforms_to_source_contract():
    pytest.importorskip("apache_beam")
    from connectors.conformance import assert_source_contract

    src = KafkaSource("b1:9092", "t")
    assert_source_contract(src)

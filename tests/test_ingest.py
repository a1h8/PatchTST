"""CSV -> Mimir/Prometheus remote-write ingestion tests.

The protobuf encoder is checked with an *independent* minimal decoder (below), so
the round-trip gives real confidence rather than asserting opaque bytes. Network
and snappy are stubbed: dry-run does neither, and the live path is exercised with
a fake urlopen + a fake snappy module.
"""
import struct
import sys
import types

import pytest

from connectors.ingest import (
    RemoteWriter,
    TimeSeries,
    csv_to_timeseries,
    encode_write_request,
)
from connectors.ingest import remote_write as rw
from connectors.ingest.__main__ import _parse_labels, main


# --- an independent protobuf decoder, just for the WriteRequest schema -------

def _read_varint(buf, i):
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, i
        shift += 7


def _read_fields(buf):
    """Yield (field_num, wire_type, value) over a protobuf message."""
    i = 0
    while i < len(buf):
        tag, i = _read_varint(buf, i)
        field_num, wire = tag >> 3, tag & 7
        if wire == 0:
            val, i = _read_varint(buf, i)
        elif wire == 1:
            val, i = struct.unpack_from("<d", buf, i)[0], i + 8
        elif wire == 2:
            ln, i = _read_varint(buf, i)
            val, i = buf[i : i + ln], i + ln
        else:  # pragma: no cover - schema has no other wire types
            raise AssertionError(wire)
        yield field_num, wire, val


def _decode_write_request(buf):
    """WriteRequest bytes -> [{labels: {...}, samples: [(ts, val)]}]."""
    series = []
    for fn, _w, val in _read_fields(buf):
        assert fn == 1
        labels, samples = {}, []
        for sfn, _sw, sval in _read_fields(val):
            if sfn == 1:  # Label
                parts = {p: v for p, _, v in _read_fields(sval)}
                labels[parts[1].decode()] = parts[2].decode()
            elif sfn == 2:  # Sample
                fields = {p: v for p, _, v in _read_fields(sval)}
                samples.append((fields[2], fields[1]))  # (ts_ms, value)
        series.append({"labels": labels, "samples": samples})
    return series


# --- encoding ---------------------------------------------------------------

def test_encode_decode_round_trip():
    ts = TimeSeries({"__name__": "cpu", "instance": "node1"}, [(1000, 1.5), (2000, 2.5)])
    decoded = _decode_write_request(encode_write_request([ts]))
    assert len(decoded) == 1
    assert decoded[0]["labels"] == {"__name__": "cpu", "instance": "node1"}
    assert decoded[0]["samples"] == [(1000, 1.5), (2000, 2.5)]


def test_labels_are_sorted_by_name():
    ts = TimeSeries({"zzz": "1", "__name__": "m", "aaa": "2"}, [(1, 1.0)])
    decoded = _decode_write_request(encode_write_request([ts]))
    # dict preserves insertion order; sorted encoding => decode order is sorted
    assert list(decoded[0]["labels"]) == ["__name__", "aaa", "zzz"]


def test_encode_skips_empty_series():
    assert encode_write_request([TimeSeries({"__name__": "m"}, [])]) == b""


def test_encode_requires_name_label():
    with pytest.raises(ValueError, match="__name__"):
        encode_write_request([TimeSeries({"instance": "n"}, [(1, 1.0)])])


# --- CSV adapter ------------------------------------------------------------

def _write_csv(tmp_path, text):
    p = tmp_path / "d.csv"
    p.write_text(text)
    return str(p)


def test_csv_to_timeseries_basic(tmp_path):
    path = _write_csv(tmp_path, "date,a,b\n2016-07-01 00:00:00,1.0,2.0\n2016-07-01 01:00:00,3.0,4.0\n")
    series = csv_to_timeseries(path, instance="etth1", metric_prefix="ett_")
    by_name = {s.labels["__name__"]: s for s in series}
    assert set(by_name) == {"ett_a", "ett_b"}
    assert by_name["ett_a"].labels["instance"] == "etth1"
    assert by_name["ett_a"].samples[0][1] == 1.0
    # 2016-07-01 00:00:00 UTC = 1467331200000 ms
    assert by_name["ett_a"].samples[0][0] == 1467331200000


def test_csv_epoch_seconds_and_ms(tmp_path):
    path = _write_csv(tmp_path, "ts,v\n1467331200,5.0\n1467331200000,6.0\n")
    s = csv_to_timeseries(path, instance="i")[0]
    assert s.samples[0][0] == 1467331200000  # seconds -> ms
    assert s.samples[1][0] == 1467331200000  # ms kept


def test_csv_skips_blank_cells(tmp_path):
    path = _write_csv(tmp_path, "ts,v\n1000,\n2000,7.0\n")
    s = csv_to_timeseries(path, instance="i")[0]
    assert s.samples == [(2000000, 7.0)]


def test_csv_timestamp_column_by_name(tmp_path):
    path = _write_csv(tmp_path, "v,when\n1.0,1000\n")
    s = csv_to_timeseries(path, instance="i", timestamp_column="when")[0]
    assert s.labels["__name__"] == "v" and s.samples == [(1000000, 1.0)]


def test_csv_skips_blank_rows(tmp_path):
    path = _write_csv(tmp_path, "ts,v\n1000,1.0\n\n2000,2.0\n")
    s = csv_to_timeseries(path, instance="i")[0]
    assert s.samples == [(1000000, 1.0), (2000000, 2.0)]


def test_end_at_now_no_samples_is_noop(tmp_path):
    # all cells blank -> no series with samples -> _shift_to_now early-returns
    path = _write_csv(tmp_path, "ts,v\n1000,\n2000,\n")
    assert csv_to_timeseries(path, instance="i", end_at_now=True) == []


def test_end_at_now_shifts_series(tmp_path):
    import time

    path = _write_csv(tmp_path, "ts,v\n1000,1.0\n2000,2.0\n")
    s = csv_to_timeseries(path, instance="i", end_at_now=True)[0]
    now_ms = int(time.time() * 1000)
    assert abs(s.samples[-1][0] - now_ms) < 5000
    # spacing preserved: 1000s and 2000s epoch -> 1_000_000 ms apart
    assert s.samples[1][0] - s.samples[0][0] == 1_000_000


# --- writer -----------------------------------------------------------------

def test_dry_run_counts_samples_without_network(monkeypatch):
    def boom(*a, **k):  # any network use would fail the test
        raise AssertionError("network called during dry-run")

    monkeypatch.setattr(rw.urllib.request, "urlopen", boom)
    series = [TimeSeries({"__name__": "m"}, [(1, 1.0), (2, 2.0), (3, 3.0)])]
    assert RemoteWriter("http://x").push(series, dry_run=True) == 3


def test_push_posts_compressed_with_headers(monkeypatch):
    captured = {}

    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["data"] = req.data
        captured["headers"] = {k.lower(): v for k, v in req.headers.items()}
        return _Resp()

    fake_snappy = types.ModuleType("snappy")
    fake_snappy.compress = lambda b: b"SNAPPY" + b
    monkeypatch.setitem(sys.modules, "snappy", fake_snappy)
    monkeypatch.setattr(rw.urllib.request, "urlopen", fake_urlopen)

    series = [TimeSeries({"__name__": "m", "instance": "n"}, [(1, 1.0)])]
    sent = RemoteWriter("http://mimir:9009/", tenant="t1").push(series)

    assert sent == 1
    assert captured["url"] == "http://mimir:9009/api/v1/push"
    assert captured["data"].startswith(b"SNAPPY")
    assert captured["headers"]["content-encoding"] == "snappy"
    assert captured["headers"]["content-type"] == "application/x-protobuf"
    assert captured["headers"]["x-scope-orgid"] == "t1"


def test_push_chunks_by_max_samples(monkeypatch):
    posts = []
    monkeypatch.setattr(RemoteWriter, "_post", lambda self, body: posts.append(body))
    series = [TimeSeries({"__name__": "m"}, [(i, float(i)) for i in range(10)])]
    sent = RemoteWriter("http://x", max_samples_per_request=4).push(series)
    assert sent == 10 and len(posts) == 3  # 4 + 4 + 2


def test_snappy_missing_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "snappy", None)   # force ImportError
    monkeypatch.setitem(sys.modules, "cramjam", None)
    with pytest.raises(RuntimeError, match="snappy"):
        rw._snappy_compress(b"data")


def test_snappy_cramjam_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "snappy", None)
    fake = types.ModuleType("cramjam")
    fake.snappy = types.SimpleNamespace(compress_raw=lambda b: b"CJ" + b)
    monkeypatch.setitem(sys.modules, "cramjam", fake)
    assert rw._snappy_compress(b"x") == b"CJx"


# --- CLI --------------------------------------------------------------------

def test_parse_labels():
    assert _parse_labels(["a=1", "b=2"]) == {"a": "1", "b": "2"}
    assert _parse_labels(None) == {}


def test_cli_dry_run(tmp_path, caplog):
    path = _write_csv(tmp_path, "date,a\n2016-07-01 00:00:00,1.0\n")
    with caplog.at_level("INFO"):
        rc = main(["--csv", path, "--endpoint", "http://x", "--instance", "i",
                   "--label", "env=test", "--dry-run"])
    assert rc == 0
    assert "dry-run" in caplog.text and "1 series" in caplog.text

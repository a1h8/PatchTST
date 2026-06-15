"""Prometheus remote-write client — push samples into Mimir / Prometheus.

The write-side counterpart of ``connectors.sources.mimir`` (which reads back via
``query_range``). Encodes a Prometheus ``WriteRequest`` and POSTs it snappy-
compressed to ``/api/v1/push`` — the path a real Grafana Mimir (or a Prometheus
started with ``--web.enable-remote-write-receiver``) ingests.

The protobuf is hand-encoded with the stdlib (the ``WriteRequest`` schema is tiny
and stable), so encoding has no third-party dependency. Only the actual push
needs snappy + network: ``snappy`` is imported lazily, and ``dry_run`` skips both
— so ``encode_write_request`` and conversion are usable anywhere.

Wire schema (prometheus/prompb):
    WriteRequest { repeated TimeSeries timeseries = 1 }
    TimeSeries   { repeated Label labels = 1; repeated Sample samples = 2 }
    Label        { string name = 1; string value = 2 }
    Sample       { double value = 1; int64 timestamp = 2 }   # timestamp in ms
"""
from __future__ import annotations

import struct
import urllib.request
from dataclasses import dataclass, field

PUSH_PATH = "/api/v1/push"


@dataclass
class TimeSeries:
    """One labelled series: ``labels`` (incl. ``__name__``) + ``(ts_ms, value)``."""

    labels: dict[str, str]
    samples: list[tuple[int, float]] = field(default_factory=list)


# --- protobuf wire encoding (stdlib only) -----------------------------------

def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _tag(field_num: int, wire_type: int) -> bytes:
    return _varint((field_num << 3) | wire_type)


def _len_delim(field_num: int, payload: bytes) -> bytes:
    return _tag(field_num, 2) + _varint(len(payload)) + payload


def _string_field(field_num: int, value: str) -> bytes:
    return _len_delim(field_num, value.encode("utf-8"))


def _double_field(field_num: int, value: float) -> bytes:
    return _tag(field_num, 1) + struct.pack("<d", value)


def _int64_field(field_num: int, value: int) -> bytes:
    return _tag(field_num, 0) + _varint(value)


def _encode_label(name: str, value: str) -> bytes:
    return _string_field(1, name) + _string_field(2, value)


def _encode_sample(ts_ms: int, value: float) -> bytes:
    return _double_field(1, value) + _int64_field(2, ts_ms)


def _encode_series(ts: TimeSeries) -> bytes:
    body = bytearray()
    # Prometheus requires labels sorted by name, with __name__ present.
    for name, value in sorted(ts.labels.items()):
        body += _len_delim(1, _encode_label(name, value))
    for ts_ms, value in ts.samples:
        body += _len_delim(2, _encode_sample(int(ts_ms), float(value)))
    return bytes(body)


def encode_write_request(series: list[TimeSeries]) -> bytes:
    """Encode a list of series into a Prometheus ``WriteRequest`` protobuf."""
    body = bytearray()
    for ts in series:
        if not ts.samples:
            continue
        if "__name__" not in ts.labels:
            raise ValueError(f"series missing __name__ label: {ts.labels}")
        body += _len_delim(1, _encode_series(ts))
    return bytes(body)


# --- snappy (lazy) ----------------------------------------------------------

def _snappy_compress(payload: bytes) -> bytes:
    """Block-format snappy via python-snappy or cramjam (whichever is present)."""
    try:
        import snappy  # python-snappy

        return snappy.compress(payload)
    except ImportError:
        pass
    try:
        from cramjam import snappy as cj  # pure-wheel alternative

        return bytes(cj.compress_raw(payload))
    except ImportError as exc:  # pragma: no cover - exercised via tests w/ fakes
        raise RuntimeError(
            "remote-write needs snappy: pip install python-snappy (or cramjam)"
        ) from exc


# --- writer -----------------------------------------------------------------

@dataclass
class RemoteWriter:
    """POST ``WriteRequest``s to a Mimir/Prometheus remote-write endpoint."""

    endpoint: str
    tenant: str | None = None
    timeout: float = 30.0
    path: str = PUSH_PATH
    max_samples_per_request: int = 5000

    def _url(self) -> str:
        return f"{self.endpoint.rstrip('/')}{self.path}"

    def push(self, series: list[TimeSeries], *, dry_run: bool = False) -> int:
        """Push all series in chunks. Returns the number of samples sent.

        ``dry_run`` encodes but neither compresses nor opens the network — useful
        to validate conversion against a real series without a running server.
        """
        sent = 0
        for chunk in _chunk(series, self.max_samples_per_request):
            body = encode_write_request(chunk)
            n = sum(len(s.samples) for s in chunk)
            if not dry_run:
                self._post(body)
            sent += n
        return sent

    def _post(self, body: bytes) -> None:
        compressed = _snappy_compress(body)
        req = urllib.request.Request(self._url(), data=compressed, method="POST")
        req.add_header("Content-Encoding", "snappy")
        req.add_header("Content-Type", "application/x-protobuf")
        req.add_header("X-Prometheus-Remote-Write-Version", "0.1.0")
        if self.tenant:
            req.add_header("X-Scope-OrgID", self.tenant)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            if resp.status >= 300:  # pragma: no cover - server-dependent
                raise RuntimeError(f"remote-write failed: HTTP {resp.status}")


def _chunk(series: list[TimeSeries], max_samples: int):
    """Yield sub-lists of series whose total sample count stays under the cap.

    A single series larger than the cap is split across chunks.
    """
    batch: list[TimeSeries] = []
    count = 0
    for ts in series:
        for i in range(0, len(ts.samples), max_samples):
            piece = TimeSeries(ts.labels, ts.samples[i : i + max_samples])
            if count and count + len(piece.samples) > max_samples:
                yield batch
                batch, count = [], 0
            batch.append(piece)
            count += len(piece.samples)
    if batch:
        yield batch

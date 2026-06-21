"""SignalStore — the structured knowledge base.

Aggregated ``SignalRecord``s are written as Parquet (the datalake) and queried
by entity/metric/time-window via DuckDB. This is the read path kube-verdict's
``rca/context_builder`` uses as historical evidence: "what is the signal history
of this entity?"

Parquet + DuckDB keeps the POC dependency-light and S3-ready (DuckDB reads
``s3://`` and Iceberg too); a ClickHouse backend can replace it at scale behind
the same ``write`` / ``query`` interface.
"""
from __future__ import annotations

import glob
import json
import os
import uuid
from typing import Iterable

from .signal import SignalRecord

_COLUMNS = [
    "entity_uid", "metric_name", "ts", "severity",
    "score", "method", "horizon", "n_points", "labels", "text",
]


class SignalStore:
    def __init__(self, root: str) -> None:
        self.root = root

    def _schema(self):
        import pyarrow as pa

        return pa.schema(
            [
                ("entity_uid", pa.string()),
                ("metric_name", pa.string()),
                ("ts", pa.int64()),
                ("severity", pa.string()),
                ("score", pa.float64()),
                ("method", pa.string()),
                ("horizon", pa.string()),
                ("n_points", pa.int64()),
                ("labels", pa.string()),   # JSON
                ("text", pa.string()),
            ]
        )

    @staticmethod
    def _to_dict(r: SignalRecord) -> dict:
        return {
            "entity_uid": r.entity_uid,
            "metric_name": r.metric_name,
            "ts": int(r.ts),
            "severity": r.severity,
            "score": float(r.score),
            "method": r.method,
            "horizon": r.horizon,
            "n_points": int(r.n_points),
            "labels": json.dumps(dict(r.labels), sort_keys=True),
            "text": r.to_text(),
        }

    def write(self, records: Iterable[SignalRecord]) -> str | None:
        """Append a Parquet partition of signals; return its path (or None if empty)."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = [self._to_dict(r) for r in records]
        if not rows:
            return None
        os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, f"signals-{uuid.uuid4().hex}.parquet")
        pq.write_table(pa.Table.from_pylist(rows, schema=self._schema()), path)
        return path

    def query(
        self,
        entity_uid: str,
        metric: str | None = None,
        since: int | None = None,
        until: int | None = None,
        limit: int | None = None,
    ) -> list[SignalRecord]:
        """Signal history for an entity, optionally filtered by metric/time window.

        This is the contract kube-verdict's context_builder calls.
        """
        import duckdb

        files = sorted(glob.glob(os.path.join(self.root, "*.parquet")))
        if not files:
            return []

        conds = ["entity_uid = ?"]
        params: list = [entity_uid]
        if metric is not None:
            conds.append("metric_name = ?")
            params.append(metric)
        if since is not None:
            conds.append("ts >= ?")
            params.append(int(since))
        if until is not None:
            conds.append("ts <= ?")
            params.append(int(until))

        sql = (
            f"SELECT {', '.join(_COLUMNS)} FROM read_parquet(?) "
            f"WHERE {' AND '.join(conds)} ORDER BY ts"
        )
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        con = duckdb.connect()
        try:
            rows = con.execute(sql, [files, *params]).fetchall()
        finally:
            con.close()

        return [self._row_to_record(r) for r in rows]

    def latest(self, entity_uid: str, metric: str | None = None) -> SignalRecord | None:
        """The most recent signal for an entity (optionally a metric), or None.

        Used to seed cross-batch state (e.g. the regime state machine) from the
        last persisted assessment — see ``detection.KBSeededRegimeState``.
        """
        import duckdb

        files = sorted(glob.glob(os.path.join(self.root, "*.parquet")))
        if not files:
            return None

        conds = ["entity_uid = ?"]
        params: list = [entity_uid]
        if metric is not None:
            conds.append("metric_name = ?")
            params.append(metric)

        sql = (
            f"SELECT {', '.join(_COLUMNS)} FROM read_parquet(?) "
            f"WHERE {' AND '.join(conds)} ORDER BY ts DESC LIMIT 1"
        )
        con = duckdb.connect()
        try:
            rows = con.execute(sql, [files, *params]).fetchall()
        finally:
            con.close()

        return self._row_to_record(rows[0]) if rows else None

    @staticmethod
    def _row_to_record(r) -> SignalRecord:
        return SignalRecord(
            entity_uid=r[0],
            metric_name=r[1],
            ts=int(r[2]),
            severity=r[3],
            score=float(r[4]),
            method=r[5],
            horizon=r[6] or "",
            n_points=int(r[7]),
            labels=json.loads(r[8] or "{}"),
            text=r[9] or "",
        )

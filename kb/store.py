"""SignalStore — the structured knowledge base.

Aggregated ``SignalRecord``s are written as Parquet (the datalake) and queried
by entity/metric/time-window via DuckDB. This is the read path kube-verdict's
``rca/context_builder`` uses as historical evidence: "what is the signal history
of this entity?"

Parquet + DuckDB keeps the POC dependency-light; a ClickHouse backend can
replace it at scale behind the same ``write`` / ``query`` interface.

``root`` is a local path or any URI ``pyarrow.fs`` understands — ``s3://bucket/kb``
(MinIO via ``?endpoint_override=host:9000&scheme=http``, credentials from
``AWS_*``), ``gs://bucket/kb`` (Application Default Credentials), ``file:///...``.
Distributed runners (Flink, Dataflow) need this: their workers share no local
filesystem, so a local root would land signals on each worker's own disk.
"""
from __future__ import annotations

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

    def _fs(self):
        """(filesystem, base path) for ``root`` — local path or a pyarrow.fs URI."""
        import pyarrow.fs as pafs

        if "://" not in self.root:
            return pafs.LocalFileSystem(), os.path.abspath(self.root)
        return pafs.FileSystem.from_uri(self.root)

    def _files(self) -> list[str]:
        """Parquet files under the root (paths valid for ``_fs()``), sorted."""
        import pyarrow.fs as pafs

        fs, base = self._fs()
        infos = fs.get_file_info(pafs.FileSelector(base, allow_not_found=True))
        return sorted(
            i.path for i in infos
            if i.type == pafs.FileType.File and i.path.endswith(".parquet")
        )

    def _read(self):
        """All signals as one Arrow table, or None when the root is empty."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        files = self._files()
        if not files:
            return None
        fs, _ = self._fs()
        return pa.concat_tables(
            [pq.read_table(f, filesystem=fs, schema=self._schema()) for f in files]
        )

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
        fs, base = self._fs()
        fs.create_dir(base, recursive=True)
        path = f"{base.rstrip('/')}/signals-{uuid.uuid4().hex}.parquet"
        pq.write_table(
            pa.Table.from_pylist(rows, schema=self._schema()), path, filesystem=fs
        )
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

        table = self._read()
        if table is None:
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
            f"SELECT {', '.join(_COLUMNS)} FROM signals "
            f"WHERE {' AND '.join(conds)} ORDER BY ts"
        )
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        con = duckdb.connect()
        try:
            con.register("signals", table)
            rows = con.execute(sql, params).fetchall()
        finally:
            con.close()

        return [self._row_to_record(r) for r in rows]

    def latest(self, entity_uid: str, metric: str | None = None) -> SignalRecord | None:
        """The most recent signal for an entity (optionally a metric), or None.

        Used to seed cross-batch state (e.g. the regime state machine) from the
        last persisted assessment — see ``detection.KBSeededRegimeState``.
        """
        import duckdb

        table = self._read()
        if table is None:
            return None

        conds = ["entity_uid = ?"]
        params: list = [entity_uid]
        if metric is not None:
            conds.append("metric_name = ?")
            params.append(metric)

        sql = (
            f"SELECT {', '.join(_COLUMNS)} FROM signals "
            f"WHERE {' AND '.join(conds)} ORDER BY ts DESC LIMIT 1"
        )
        con = duckdb.connect()
        try:
            con.register("signals", table)
            rows = con.execute(sql, params).fetchall()
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

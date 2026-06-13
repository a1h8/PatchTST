"""Signal knowledge-base HTTP service.

Exposes the read contract kube-verdict's ``rca/context_builder`` calls during
RCA — "give me the signal history of this entity":

    GET /api/v1/signals/history?entity=Pod/prod/api&metric=cpu_usage&since=..&until=..

Returns aggregated SignalRecords as historical evidence. Read-only; the write
path is the aggregation pipeline (connectors + engines) feeding the store.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, Query

from .store import SignalStore


def create_app(store: SignalStore) -> FastAPI:
    app = FastAPI(title="Signal Knowledge Base", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/v1/signals/history")
    def signal_history(
        entity: str = Query(..., description="entity_uid, e.g. 'Pod/prod/api-7c9'"),
        metric: Optional[str] = Query(None, description="filter by metric name"),
        since: Optional[int] = Query(None, description="epoch-ms lower bound"),
        until: Optional[int] = Query(None, description="epoch-ms upper bound"),
        limit: Optional[int] = Query(None, ge=1, description="max records"),
    ) -> dict:
        records = store.query(entity, metric=metric, since=since, until=until, limit=limit)
        return {
            "entity": entity,
            "metric": metric,
            "count": len(records),
            "signals": [asdict(r) for r in records],
        }

    return app

"""Knowledge base — aggregated signal store + query service (D7).

The longitudinal evidence layer that kube-verdict queries during RCA. See
docs/ARCHITECTURE.md. Public surface: the signal schema, the store, and the
HTTP app factory.
"""
from .signal import SignalRecord
from .store import SignalStore

# Register the SPI sink façade (write path goes through the connector cycle).
from .sink import SignalStoreSink  # noqa: E402

__all__ = ["SignalRecord", "SignalStore", "SignalStoreSink", "create_app"]


def create_app(store: SignalStore):
    # lazy import so the schema/store are usable without FastAPI installed
    from .api import create_app as _create_app

    return _create_app(store)

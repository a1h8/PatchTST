"""sys.path shim for the vendored ``PatchTST_self_supervised`` package.

The package now lives inside the ``vendor/patchtst-upstream`` git submodule
(yuqinie98/PatchTST) rather than directly in this repo. Call
``ensure_vendor_path()`` before importing anything from it.
"""
from __future__ import annotations

import sys
from pathlib import Path

_VENDOR_ROOT = Path(__file__).resolve().parent.parent / "vendor" / "patchtst-upstream"


def ensure_vendor_path() -> None:
    path = str(_VENDOR_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)

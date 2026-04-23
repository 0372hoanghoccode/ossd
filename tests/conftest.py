from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def clear_embedder_cache() -> None:
    import src.pipeline as pipeline_module

    pipeline_module._EMBEDDER_CACHE.clear()
    yield
    pipeline_module._EMBEDDER_CACHE.clear()

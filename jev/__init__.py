"""jev-torch: train JEV-style calibrated decision models in PyTorch.

Names are imported lazily, so `python -m jev.<module>` runs without importing the whole package first.
"""

from importlib import import_module

_EXPORTS = {
    "JEVConfig": "config", "load_config": "config",
    "JEVExample": "schema", "load_jsonl": "schema", "save_jsonl": "schema",
    "JEVModel": "model",
    "JEVPredictor": "predict",
    "SOURCES": "sources", "load_split": "sources", "register_source": "sources",
    "Trainer": "trainer",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module 'jev' has no attribute {name!r}")

"""Data sources: anything that can produce a split as a list of JEVExample.

A source is a function `(cfg: DataConfig, split: str) -> list[JEVExample]` registered by name.
New datasets plug in by registering a source; the trainer never needs to change:

    @register_source("my_dataset")
    def my_dataset(cfg, split):
        return [JEVExample(...) for row in ...]

Your own data in the JEV format needs no code: use `source: jsonl` with `path: <dir>`.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

from .config import DataConfig
from .schema import JEVExample, load_jsonl

SourceFn = Callable[[DataConfig, str], list[JEVExample]]
SOURCES: dict[str, SourceFn] = {}
SOURCE_DATASETS: dict[str, str] = {}  # source name -> Hugging Face dataset id, when it reads one


def register_source(name: str, hf_dataset: str | None = None):
    def deco(fn: SourceFn) -> SourceFn:
        SOURCES[name] = fn
        if hf_dataset:
            SOURCE_DATASETS[name] = hf_dataset
        return fn

    return deco


def describe_source(cfg: DataConfig) -> str:
    """Human-readable name of the data a config trains on, e.g. for logs and model cards."""
    if cfg.source == "jsonl":
        return f"local JSONL files in {cfg.path}"
    hf = SOURCE_DATASETS.get(cfg.source)
    return f"{cfg.source} (https://huggingface.co/datasets/{hf})" if hf else cfg.source


def load_split(cfg: DataConfig, split: str, limit: int | None = None, seed: int = 0) -> list[JEVExample]:
    """Load one split, apply include/exclude filters, then a deterministic random subset if `limit`."""
    if cfg.source not in SOURCES:
        raise KeyError(f"unknown data source {cfg.source!r}; registered: {sorted(SOURCES)}")
    examples = SOURCES[cfg.source](cfg, split)
    for key, allowed in cfg.include.items():
        allowed = set(allowed)
        examples = [ex for ex in examples if ex.get(key) in allowed]
    for key, banned in cfg.exclude.items():
        banned = set(banned)
        examples = [ex for ex in examples if ex.get(key) not in banned]
    if limit is not None and len(examples) > limit:
        examples = random.Random(seed).sample(examples, limit)
    return examples


@register_source("jsonl")
def jsonl(cfg: DataConfig, split: str) -> list[JEVExample]:
    if not cfg.path:
        raise ValueError("source `jsonl` needs data.path (a directory with <split>.jsonl files)")
    return load_jsonl(Path(cfg.path) / f"{split}.jsonl")


@register_source("jev_distill", hf_dataset="SargeDev/jev-distill-corpus-v3")
def jev_distill(cfg: DataConfig, split: str) -> list[JEVExample]:
    """SargeDev/jev-distill-corpus-v3: 741k rows of TypeSafe Jev 1.13 output distributions.

    Splits: train (656k) / validation / calibration / test / ood / test_set_30k (the held-out
    benchmark, stratified by family and kind). Rows carry `family` and `source` in meta.
    """
    from datasets import load_dataset

    ds = load_dataset("SargeDev/jev-distill-corpus-v3", split=split)
    return [
        JEVExample(
            id=r["id"], kind=r["kind"], state=r["state"], question=r["question"],
            options=r["options"], target=r["target"], domain=r["domain"],
            meta={"family": r["family"], "source": r["source"]},
        )
        for r in ds
    ]


@register_source("commonsense_qa", hf_dataset="tau/commonsense_qa")
def commonsense_qa(cfg: DataConfig, split: str) -> list[JEVExample]:
    """tau/commonsense_qa: hard labels, 5 options. A template for adapting plain multiple-choice data.

    Only train / validation have labels.
    """
    from datasets import load_dataset

    ds = load_dataset("tau/commonsense_qa", split=split)
    return [
        JEVExample(
            id=f"csqa-{r['id']}", kind="choice", state="", question=r["question"],
            options=r["choices"]["text"], label=r["choices"]["label"].index(r["answerKey"]),
            domain="commonsense",
        )
        for r in ds
    ]

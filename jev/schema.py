"""The JEV example schema and JSONL I/O.

One example per line, following SargeDev/jev-distill-corpus-v3:

    {
      "id": "v3_ee950df8c330cd92_c",
      "kind": "choice",                   # noul (yes/no) | choice (2-24 options) | score (0-5 ordinal)
      "state": "A distribution center shows a count variance of 2.3%; ...",
      "question": "Supplier response for this scenario.",
      "options": ["issue_warning", "renegotiate", "dual_source", "maintain"],
      "target": [0.6, 0.07, 0.32, 0.01],  # teacher probability per option (soft label)
      "label": 0,                         # optional hard label; derived as argmax(target) if absent
      "domain": "inventory_supply",       # optional
      "meta": {"family": "business"}      # optional; unknown top-level keys are folded in here
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KINDS = ("noul", "choice", "score")


@dataclass
class JEVExample:
    state: str
    question: str
    options: list[str]
    target: list[float] | None = None
    label: int | None = None
    kind: str = "choice"
    domain: str | None = None
    id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        k = len(self.options)
        if k < 2:
            raise ValueError(f"[{self.id}] need at least 2 options, got {k}")
        if self.kind not in KINDS:
            raise ValueError(f"[{self.id}] kind must be one of {KINDS}, got {self.kind!r}")
        if self.label is None and self.target is None:
            raise ValueError(f"[{self.id}] need `target` or `label`")
        if self.label is not None and not 0 <= self.label < k:
            raise ValueError(f"[{self.id}] label {self.label} out of range for {k} options")
        if self.target is not None:
            if len(self.target) != k:
                raise ValueError(f"[{self.id}] target has {len(self.target)} entries, expected {k}")
            s = float(sum(self.target))
            if s <= 0:
                raise ValueError(f"[{self.id}] target must have positive mass")
            self.target = [float(t) / s for t in self.target]
            if self.label is None:
                self.label = max(range(k), key=lambda i: self.target[i])

    @property
    def ordinal(self) -> bool:
        """Options are ordered (score 0-5): near-misses should cost less (RPS), and order is kept."""
        return self.kind == "score"

    def target_dist(self) -> list[float]:
        if self.target is not None:
            return self.target
        return [1.0 if i == self.label else 0.0 for i in range(len(self.options))]

    def get(self, key: str, default=None):
        """Field lookup for filtering: top-level attribute first, then meta."""
        if key in self.__dataclass_fields__ and key != "meta":
            return getattr(self, key)
        return self.meta.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        d = {"id": self.id, "kind": self.kind, "state": self.state, "question": self.question,
             "options": self.options, "label": self.label}
        if self.target is not None:
            d["target"] = self.target
        if self.domain:
            d["domain"] = self.domain
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JEVExample":
        known = set(cls.__dataclass_fields__)
        extra = {k: v for k, v in d.items() if k not in known}
        d = {k: v for k, v in d.items() if k in known}
        if extra:
            d["meta"] = {**d.get("meta", {}), **extra}
        return cls(**d)


def load_jsonl(path: str | Path) -> list[JEVExample]:
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if line.strip():
                d = json.loads(line)
                d.setdefault("id", f"{Path(path).stem}-{i}")
                out.append(JEVExample.from_dict(d))
    return out


def save_jsonl(examples: list[JEVExample], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")

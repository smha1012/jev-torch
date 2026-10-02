"""How close a model is to its teacher (TypeSafe Jev 1.13 for the JEV corpus), and how that compares
with the published autotrust/JEV numbers.

Only rows labelled by the teacher count (`data.teacher_sources`, `yuri_v3` in the corpus): the other
streams carry programmatic or placeholder labels. In `test_set_30k` these are exactly the 25,376
"Jev-labelled" rows autotrust evaluates on, so the numbers are directly comparable.
"""

from __future__ import annotations

import torch

from .losses import compute_metrics, metrics_by
from .schema import KINDS

# Published on the autotrust/JEV model cards (test_set_30k, Jev-labelled rows unless noted).
AUTOTRUST_REFERENCE = {
    "JEV-9B": {"kl": 0.019, "choice_acc": 0.902, "choice_acc_all_rows": 0.898, "ece": 0.0007},
    "JEV-27B": {"kl": 0.017, "choice_acc_all_rows": 0.903, "ece": 0.0009},
}


def teacher_block(res: dict, examples: list, teacher_sources: list[str]) -> dict | None:
    """Metrics on teacher-labelled rows, overall and per kind, plus choice agreement on all rows.

    `res` holds padded tensors (logits already calibrated if desired) in the order of `examples`.
    """
    if not teacher_sources:
        return None
    sources = set(teacher_sources)
    rows = [i for i, ex in enumerate(examples) if ex.get("source") in sources]
    if not rows:
        return None
    ix = torch.tensor(rows)
    sub = {k: v[ix] for k, v in res.items()}
    block = compute_metrics(**sub)
    block["by_kind"] = metrics_by(sub, [KINDS[k] for k in sub["kind"].tolist()])
    choice_all = [i for i, ex in enumerate(examples) if ex.kind == "choice"]
    if choice_all:
        cix = torch.tensor(choice_all)
        block["choice_acc_all_rows"] = compute_metrics(**{k: v[cix] for k, v in res.items()})["acc"]
    block["teacher_sources"] = sorted(sources)
    return block


def format_comparison(block: dict, split: str) -> str:
    """A small table: this model vs the autotrust/JEV references."""
    choice = block["by_kind"].get("choice", {})
    ours = {"kl": block["kl"], "choice_acc": choice.get("acc"),
            "choice_acc_all_rows": block.get("choice_acc_all_rows"), "ece": block["ece"]}
    cols = [("mean KL to teacher (lower is better)", "kl", "{:.4f}"),
            ("choice top-1 agreement, teacher rows", "choice_acc", "{:.1%}"),
            ("choice top-1 agreement, all choice rows", "choice_acc_all_rows", "{:.1%}"),
            ("ECE vs teacher probabilities", "ece", "{:.4f}")]
    fmt = lambda v, f: f.format(v) if v is not None else "-"
    lines = [f"vs teacher on {split} ({block['n']:,} teacher-labelled rows):",
             f"  {'metric':42s} {'this model':>11s} " + " ".join(f"{name:>9s}" for name in AUTOTRUST_REFERENCE)]
    for label, key, f in cols:
        refs = " ".join(f"{fmt(r.get(key), f):>9s}" for r in AUTOTRUST_REFERENCE.values())
        lines.append(f"  {label:42s} {fmt(ours[key], f):>11s} {refs}")
    lines.append("  per kind: " + "  ".join(f"{k} acc={m['acc']:.1%} kl={m['kl']:.4f}"
                                            for k, m in block["by_kind"].items()))
    return "\n".join(lines)

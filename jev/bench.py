"""JevBench: compare a checkpoint with TypeSafe Jev 1.13.0 on ground-truth labels.

    python3 -m jev.bench --model runs/jev-9b/best
    python3 -m jev.bench --model your-name/jev-9b --out bench.json
    python3 -m jev.bench --model runs/jev-9b/best --limit 50          # quick look

Data: the public JevBench release (https://huggingface.co/datasets/Leanmcp/jevbench, pinned below):
cases built from public datasets with gold answers, plus the predictions TypeSafe's hosted Jev 1.13.0
returned for the very same cases. Both systems are scored here with the same code, case by case.

Skipped slices: banking77 (77 options; the 24-slot head scores at most 16) and sst5 (its text is
withheld for licensing). Each case keeps its source dataset's licence; nothing is redistributed here.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from .env import load_env
from .predict import JEVPredictor
from .schema import JEVExample

REPO = "Leanmcp/jevbench"
REVISION = "ed50fe0381714d1c0273cbe03cece755b25d586b"
TEACHER_RUN = "jev-1.13.0"
SLICES = ["medqa_usmle", "medmcqa", "pubmedqa", "mmlu_pro", "scienceqa_text",
          "aegis2", "aegis2_response", "jailbreak_classification", "prompt_injections", "atbench500"]
SKIPPED = {"banking77": "77 options; the head scores at most 16", "sst5": "text withheld by the release"}
MAX_CHOICE = 16


def _download(path: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO, path, repo_type="dataset", revision=REVISION))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def render_state(state) -> str:
    """The case state is a dict of named fields; render it as labelled text blocks."""
    if not isinstance(state, dict):
        return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    parts = []
    for key, value in state.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        parts.append(f"{key.replace('_', ' ')}:\n{text}")
    return "\n\n".join(parts)


def case_to_example(case: dict) -> tuple[JEVExample, list] | None:
    """JEVExample plus the gold-comparable option keys, or None when the case cannot be scored."""
    q, kind = case["question"], case["task_type"]
    criteria = q.get("criteria") or {}
    state = render_state(case["state"])
    if kind == "choice":
        keys = list(criteria)
        if len(keys) > MAX_CHOICE or case["gold"] not in keys:
            return None
        ex = JEVExample(kind="choice", state=state, question=q["instructions"],
                        options=[str(criteria[k]) for k in keys], label=keys.index(case["gold"]),
                        id=case["case_id"])
        return ex, keys
    if kind == "noul":
        hint = ""
        if "true" in criteria and "false" in criteria:
            hint = f"\n(true: {criteria['true']} / false: {criteria['false']})"
        ex = JEVExample(kind="noul", state=state, question=q["instructions"] + hint,
                        options=["false", "true"], label=int(bool(case["gold"])), id=case["case_id"])
        return ex, [False, True]
    return None


def _ece(conf: torch.Tensor, correct: torch.Tensor, n_bins: int = 10) -> float:
    edges, ece = torch.linspace(0, 1, n_bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(correct[m].mean().item() - conf[m].mean().item())
    return ece


def score(probs: list[list[float]], labels: list[int], kind: str = "choice") -> dict:
    """Accuracy, top-label ECE (10 bins) and Brier for hard gold labels.

    noul is decided as p_true >= 0.5 (the release's accuracy_at_0.5), so an exact 0.5 counts as true.
    """
    if not probs:
        return {"n": 0}
    acc, conf, brier = [], [], []
    for p, y in zip(probs, labels):
        p = torch.tensor(p, dtype=torch.float32)
        p = p / p.sum() if p.sum() > 0 else torch.full_like(p, 1 / len(p))
        top = int(p[1] >= 0.5) if kind == "noul" else int(p.argmax())
        acc.append(float(top == y))
        conf.append(float(p[top]))
        onehot = torch.zeros_like(p)
        onehot[y] = 1
        brier.append(float(((p - onehot) ** 2).sum()))
    acc_t, conf_t = torch.tensor(acc), torch.tensor(conf)
    return {"n": len(acc), "acc": acc_t.mean().item(), "ece": _ece(conf_t, acc_t),
            "brier": sum(brier) / len(brier)}


def teacher_probs(pred: dict, keys: list) -> list[float] | None:
    """TypeSafe's probability vector in our option order."""
    if pred.get("error"):
        return None
    if pred["answer_type"] == "noul":
        p_true = float(pred["p_true"])
        return [1 - p_true, p_true]
    dist = pred.get("probabilities") or {}
    return [float(dist.get(k, 0.0)) for k in keys]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="checkpoint directory or Hub repo id")
    p.add_argument("--slices", nargs="+", default=SLICES)
    p.add_argument("--limit", type=int, help="cases per slice, for a quick look")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--out", help="write per-slice results as JSON")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    load_env()
    teacher = {r["case_id"]: r for r in _read_jsonl(_download(f"predictions/{TEACHER_RUN}/predictions.jsonl"))}
    jev = JEVPredictor(args.model, device=args.device, batch_size=args.batch_size, max_length=args.max_length)
    print(f"JevBench {REPO}@{REVISION[:8]} vs {TEACHER_RUN} | skipped: "
          + ", ".join(f"{k} ({v})" for k, v in SKIPPED.items()))

    results, all_ours, all_teacher, all_labels = {}, [], [], []
    for slice_name in args.slices:
        cases = _read_jsonl(_download(f"cases/{slice_name}.jsonl"))
        if args.limit:
            cases = cases[: args.limit]
        items = []
        for case in cases:
            converted = case_to_example(case)
            t = teacher.get(case["case_id"])
            if converted is None or t is None:
                continue
            ex, keys = converted
            tp = teacher_probs(t, keys)
            if tp is not None:
                items.append((ex, tp))
        if not items:
            print(f"  {slice_name}: no scorable cases")
            continue
        examples = [ex for ex, _ in items]
        order = sorted(range(len(examples)), key=lambda i: len(examples[i].state))
        probs_sorted = jev.predict_batch([examples[i] for i in order])
        ours = [None] * len(examples)
        for pos, i in enumerate(order):
            ours[i] = list(probs_sorted[pos].values())
        labels = [ex.label for ex in examples]
        theirs = [tp for _, tp in items]
        kind = examples[0].kind
        results[slice_name] = {"ours": score(ours, labels, kind), "teacher": score(theirs, labels, kind),
                               "kind": kind, "skipped_cases": len(cases) - len(items)}
        all_ours += [(p, kind) for p in ours]
        all_teacher += [(p, kind) for p in theirs]
        all_labels += labels
        r = results[slice_name]
        print(f"  {slice_name:26s} n={r['ours']['n']:5d}  acc {r['ours']['acc']:.3f} vs {r['teacher']['acc']:.3f}"
              f"  ece {r['ours']['ece']:.3f} vs {r['teacher']['ece']:.3f}")

    def pooled(rows):  # score each case with its own kind's rule, then pool
        per = [score([p], [y], k) for (p, k), y in zip(rows, all_labels)]
        acc = torch.tensor([r["acc"] for r in per])
        conf = torch.tensor([max(p) if k == "choice" else (p[1] if p[1] >= 0.5 else p[0]) for p, k in rows])
        return {"n": len(per), "acc": acc.mean().item(), "ece": _ece(conf, acc),
                "brier": sum(r["brier"] for r in per) / len(per)}

    overall = {"ours": pooled(all_ours), "teacher": pooled(all_teacher)}
    macro = {who: sum(r[who]["acc"] for r in results.values()) / len(results) for who in ("ours", "teacher")}
    print("\n" + format_table(results, overall, macro))
    if args.out:
        Path(args.out).write_text(json.dumps({"model": args.model, "data": f"{REPO}@{REVISION}",
                                              "teacher_run": TEACHER_RUN, "skipped_slices": SKIPPED,
                                              "slices": results, "overall": overall, "macro_acc": macro},
                                             indent=2))
        print(f"wrote {args.out}")


def format_table(results: dict, overall: dict, macro: dict) -> str:
    lines = [f"{'slice':26s} {'kind':6s} {'n':>5s}  {'acc ours':>8s} {'acc Jev':>8s} {'Δ':>6s}  "
             f"{'ECE ours':>8s} {'ECE Jev':>8s}"]
    for name, r in results.items():
        o, t = r["ours"], r["teacher"]
        lines.append(f"{name:26s} {r['kind']:6s} {o['n']:5d}  {o['acc']:8.3f} {t['acc']:8.3f} "
                     f"{o['acc'] - t['acc']:+6.3f}  {o['ece']:8.3f} {t['ece']:8.3f}")
    o, t = overall["ours"], overall["teacher"]
    lines.append(f"{'all cases':26s} {'':6s} {o['n']:5d}  {o['acc']:8.3f} {t['acc']:8.3f} "
                 f"{o['acc'] - t['acc']:+6.3f}  {o['ece']:8.3f} {t['ece']:8.3f}")
    lines.append(f"{'mean over slices':26s} {'':6s} {'':5s}  {macro['ours']:8.3f} {macro['teacher']:8.3f} "
                 f"{macro['ours'] - macro['teacher']:+6.3f}")
    lines.append(f"relative to TypeSafe Jev (all cases): {o['acc'] / t['acc']:.1%} of its accuracy")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

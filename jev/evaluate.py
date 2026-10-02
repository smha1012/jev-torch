"""Evaluate a checkpoint and compare it with the teacher and the autotrust/JEV references.

    python3 -m jev.evaluate --model runs/jev-9b/best
    python3 -m jev.evaluate --model your-name/jev-9b --split test_set_30k --out eval.json
    python3 -m jev.evaluate --model runs/jev-9b/best --limit 2000          # quick look
    python3 -m jev.evaluate --model runs/jev-9b/best --split test_set_30k ood \
        --update_report runs/jev-9b/report.json                          # refresh a run's report

The data settings (dataset, pinned revision, teacher_sources) come from --config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .collate import KIND_SLOTS
from .config import load_config
from .env import env_overrides, load_env
from .losses import apply_temperatures, compute_metrics, metrics_by
from .predict import JEVPredictor
from .report import format_comparison, teacher_block
from .schema import KINDS
from .sources import describe_source, load_split

MAX_K = max(n for _, n in KIND_SLOTS.values())


def pad_logits(examples, logits_list) -> dict[str, torch.Tensor]:
    n = len(examples)
    out = {"logits": torch.full((n, MAX_K), float("-inf")), "target": torch.zeros(n, MAX_K),
           "option_mask": torch.zeros(n, MAX_K, dtype=torch.bool)}
    for i, (ex, lg) in enumerate(zip(examples, logits_list)):
        k = len(lg)
        out["logits"][i, :k], out["target"][i, :k], out["option_mask"][i, :k] = lg, torch.tensor(ex.target_dist()), True
    out["label"] = torch.tensor([ex.label for ex in examples])
    out["kind"] = torch.tensor([KINDS.index(ex.kind) for ex in examples])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="checkpoint directory or Hub repo id")
    p.add_argument("--config", default="configs/jev-9b.yaml", help="where the data settings come from")
    p.add_argument("--split", nargs="+", default=["test_set_30k"], help="one or more splits")
    p.add_argument("--limit", type=int, help="random subset for a quick look")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out", help="also write the results as JSON")
    p.add_argument("--update_report", metavar="REPORT_JSON",
                   help="write the new metrics into a run's report.json (e.g. before refreshing the model card)")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    load_env()
    cfg = load_config(args.config if Path(args.config).exists() else None, env_overrides())
    jev = JEVPredictor(args.model, device=args.device, batch_size=args.batch_size,
                       max_length=cfg.data.max_length)
    temps = jev.model.temperature.cpu()
    results = {}
    for split in args.split:
        examples = load_split(cfg.data, split, args.limit, cfg.train.seed)
        print(f"data={describe_source(cfg.data)} split={split} rows={len(examples)}")
        order = sorted(range(len(examples)), key=lambda i: len(examples[i].state))  # less padding
        sorted_logits = jev.logits([examples[i] for i in order])
        logits = [None] * len(examples)
        for pos, i in enumerate(order):
            logits[i] = sorted_logits[pos]

        res = pad_logits(examples, logits)
        cal = {**res, "logits": apply_temperatures(res["logits"], res["kind"], temps)}
        result = {"uncalibrated": compute_metrics(**res), "calibrated": compute_metrics(**cal)}
        result["calibrated"]["by_kind"] = metrics_by(cal, [KINDS[k] for k in cal["kind"].tolist()])
        vs = teacher_block(cal, examples, cfg.data.teacher_sources)
        if vs:
            result["vs_teacher"] = vs
        results[split] = result

        c = result["calibrated"]
        print(f"{split} all rows (calibrated): acc={c['acc']:.4f} kl={c['kl']:.4f} ece={c['ece']:.4f} "
              f"ece_top1={c['ece_top1']:.4f} n={c['n']}")
        if vs:
            print(format_comparison(vs, split))

    if args.out:
        Path(args.out).write_text(json.dumps({"model": args.model, "temperature": dict(zip(KINDS, temps.tolist())),
                                              **results}, indent=2))
        print(f"wrote {args.out}")
    if args.update_report:
        if args.limit:
            raise SystemExit("--update_report needs the full splits; drop --limit")
        path = Path(args.update_report)
        report = json.loads(path.read_text()) if path.exists() else {}
        for split, result in results.items():
            report[split] = {**report.get(split, {}), **result}
        path.write_text(json.dumps(report, indent=2))
        print(f"updated {path}")


if __name__ == "__main__":
    main()

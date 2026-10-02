"""One command for a full JevBench report: the trained model, its untrained backbone, and the charts.

    python3 -m jev.bench_report --model runs/jev-9b/best
    python3 -m jev.bench_report --model your-name/jev-9b --out_dir bench/jev-9b
    python3 -m jev.bench_report --model runs/jev-9b/best --limit 50        # quick look

Steps:
  1. JevBench on the trained model                     -> bench.json
  2. JevBench on its base model, untrained (zero-shot)  -> bench-zeroshot.json   (base read from the checkpoint)
  3. charts with all three systems                      -> bench.png, bench-dark.png
  4. a summary table: what training added, and the remaining gap to TypeSafe Jev -> summary.md

Models run one after another, so a single GPU is enough. Re-running reuses finished steps unless --rerun.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from .bench import SLICES, run
from .env import load_env
from .model import resolve_checkpoint
from .plot_bench import plot
from .predict import JEVPredictor


def base_model_of(model: str) -> str:
    """The backbone a checkpoint was trained from, read from its jev_config.json."""
    meta = json.loads((resolve_checkpoint(model) / "jev_config.json").read_text())
    return meta["model"]["name"]


def summary_table(trained: dict, zero: dict | None) -> str:
    """Markdown: per slice, zero-shot vs trained vs Jev accuracy, the training gain and the gap to Jev."""
    head = "| Slice | Kind | n | Zero-shot | Trained | TypeSafe Jev | Training gain | Gap to Jev |"
    lines = [head, "|---|---|---:|---:|---:|---:|---:|---:|"]
    rows = list(trained["slices"].items()) + [("**all cases**", {"kind": "", **trained["overall"]})]
    for name, r in rows:
        z = (zero["overall"] if name == "**all cases**" else zero["slices"].get(name, {})).get("ours") if zero else None
        t, j = r["ours"]["acc"], r["teacher"]["acc"]
        zs = f"{z['acc']:.1%}" if z else "–"
        gain = f"{t - z['acc']:+.1%}" if z else "–"
        lines.append(f"| {name} | {r['kind']} | {r['ours']['n']:,} | {zs} | **{t:.1%}** | {j:.1%} | {gain} | {t - j:+.1%} |")
    return "\n".join(lines)


def _free(predictor):
    del predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="trained checkpoint directory or Hub repo id")
    p.add_argument("--base", help="backbone for the zero-shot baseline (default: read from the checkpoint)")
    p.add_argument("--no_zero_shot", action="store_true", help="skip the untrained baseline")
    p.add_argument("--out_dir", help="default: bench/<model name>")
    p.add_argument("--name", help="label for the trained model in charts (default: its name)")
    p.add_argument("--slices", nargs="+", default=SLICES)
    p.add_argument("--limit", type=int, help="cases per slice, for a quick look")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--device", default="auto")
    p.add_argument("--rerun", action="store_true", help="recompute steps whose JSON already exists")
    args = p.parse_args()

    load_env()
    parts = Path(args.model.rstrip("/")).parts  # runs/jev-9b/best -> jev-9b ; you/jev-9b -> jev-9b
    name = args.name or (parts[-2] if parts[-1] == "best" and len(parts) > 1 else parts[-1])
    out = Path(args.out_dir or f"bench/{name}")
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(device=args.device, batch_size=args.batch_size, max_length=args.max_length)

    def step(path: Path, label: str, make):
        if path.exists() and not args.rerun:
            print(f"== reusing {path} (pass --rerun to recompute)")
            return json.loads(path.read_text())
        print(f"\n== {label}")
        predictor = make()
        result = run(predictor, label if "zero-shot" in label else args.model, args.slices, args.limit)
        _free(predictor)
        path.write_text(json.dumps(result, indent=2))
        print(f"wrote {path}")
        return result

    trained = step(out / "bench.json", f"trained model: {args.model}", lambda: JEVPredictor(args.model, **kw))
    zero = None
    if not args.no_zero_shot:
        base = args.base or base_model_of(args.model)
        zero = step(out / "bench-zeroshot.json", f"zero-shot {base}", lambda: JEVPredictor.zero_shot(base, **kw))

    for theme, file in (("light", "bench.png"), ("dark", "bench-dark.png")):
        plot(trained, out / file, theme, name, zero, "zero-shot")
    table = summary_table(trained, zero)
    (out / "summary.md").write_text(f"# JevBench: {args.model}\n\n{table}\n")

    print("\n" + table)
    print(f"\nreport in {out}/: bench.json" + (", bench-zeroshot.json" if zero else "")
          + ", bench.png, bench-dark.png, summary.md")


if __name__ == "__main__":
    main()

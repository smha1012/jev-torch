"""Plot JevBench results (the JSON written by `python -m jev.bench --out`).

    python3 -m jev.plot_bench bench.json                    # -> bench.png
    python3 -m jev.plot_bench bench.json --out bench.svg --theme dark
    python3 -m jev.plot_bench bench.json --baseline bench-zeroshot.json   # + the untrained backbone

Two panels on the same slices: accuracy (higher is better) and ECE (lower is better), each comparing
this model with TypeSafe Jev 1.13.0. Needs seaborn (pip install seaborn).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Reference data-viz palette: categorical slots 1-2, validated for colour-vision deficiency in both modes.
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "ours": "#2a78d6", "teacher": "#eb6834", "baseline": "#1baf7a"},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#3a3a37",
             "ours": "#3987e5", "teacher": "#d95926", "baseline": "#199e70"},
}
KIND_ORDER = {"choice": 0, "noul": 1, "score": 2}


def plot(data: dict, out: Path, theme: str = "light", model_name: str | None = None,
         baseline: dict | None = None, baseline_name: str = "zero-shot"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    c = THEMES[theme]
    slices = data["slices"]
    # Group by question kind, then strongest-to-weakest for the teacher, so gaps read top to bottom.
    names = sorted(slices, key=lambda s: (KIND_ORDER.get(slices[s]["kind"], 9), -slices[s]["teacher"]["acc"]))
    order = [f"{s}  ·  {slices[s]['kind']}" for s in names]
    ours_label = model_name or Path(str(data.get("model", "this model"))).name or "this model"
    teacher_label = f"TypeSafe {data.get('teacher_run', 'Jev')}"
    sources = [(ours_label, slices, "ours"), (teacher_label, slices, "teacher")]
    if baseline:  # e.g. the untrained backbone from `jev.bench --zero_shot`, drawn first
        sources.insert(0, (baseline_name, baseline["slices"], "ours"))
    systems = [s for s, _, _ in sources]

    # Long-form table: one row per (slice, system).
    df = pd.DataFrame([
        {"slice": label, "system": system, "acc": src[name][who]["acc"], "ece": src[name][who]["ece"]}
        for name, label in zip(names, order)
        for system, src, who in sources if name in src
    ])

    sns.set_theme(style="whitegrid", font="DejaVu Sans", font_scale=0.95, rc={
        "figure.facecolor": c["surface"], "axes.facecolor": c["surface"], "savefig.facecolor": c["surface"],
        "grid.color": c["grid"], "axes.edgecolor": c["grid"], "text.color": c["text"],
        "axes.labelcolor": c["muted"], "xtick.color": c["muted"], "ytick.color": c["text"],
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, (0.52 + 0.24 * bool(baseline)) * len(names) + 2.4), sharey=True,
                             gridspec_kw={"width_ratios": [1.6, 1]})
    palette = {ours_label: c["ours"], teacher_label: c["teacher"]}
    if baseline:
        palette[baseline_name] = c["baseline"]
    panels = [("acc", "Accuracy (higher is better)", axes[0], "{:.0%}"),
              ("ece", "Calibration error, ECE (lower is better)", axes[1], "{:.3f}")]
    for key, title, ax, fmt in panels:
        sns.barplot(data=df, x=key, y="slice", hue="system", order=order, hue_order=systems, palette=palette,
                    orient="h", width=0.76, gap=0.1, legend=False, ax=ax,
                    saturation=1, linewidth=0)  # keep the validated colours; no outlines
        for bars in ax.containers:
            ax.bar_label(bars, labels=[fmt.format(v) for v in bars.datavalues], padding=4, fontsize=8.5,
                         color=c["muted"])
        top = df[key].max()
        ax.set_xlim(0, 1.09 if key == "acc" else top * 1.2)  # room for the value labels
        if key == "acc":
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(title, loc="left", fontsize=11, color=c["text"], pad=10)
        ax.set(xlabel="", ylabel="")
        ax.grid(axis="y", visible=False)
        sns.despine(ax=ax, left=True)
        ax.tick_params(length=0)

    o, t = data["overall"]["ours"], data["overall"]["teacher"]
    height_in = fig.get_figheight()
    line = 0.3 / height_in  # one text line, in figure-fraction units
    top_y = 1 - 0.18 / height_in
    bench = "Leanmcp JevBench v0.1" if "Leanmcp/jevbench" in str(data.get("data", "")) else "JevBench"
    fig.text(0.01, top_y, f"{bench}: {ours_label} vs {teacher_label}", ha="left", va="top", fontsize=13,
             color=c["text"], fontweight="bold")
    summary = (f"All {o['n']:,} cases: accuracy {o['acc']:.1%} vs {t['acc']:.1%} "
               f"({o['acc'] / t['acc']:.1%} of Jev's) · ECE {o['ece']:.3f} vs {t['ece']:.3f}")
    if baseline:  # over the slices shown here, not the baseline file's own overall (it may cover more slices)
        shown = [baseline["slices"][n]["ours"] for n in names if n in baseline["slices"]]
        if shown:
            acc = sum(r["acc"] * r.get("n", 1) for r in shown) / sum(r.get("n", 1) for r in shown)
            summary += f" · {baseline_name}: {acc:.1%}"
    fig.text(0.01, top_y - line, summary, ha="left", va="top", fontsize=10, color=c["muted"])
    handles = [matplotlib.patches.Patch(color=palette[s], label=s) for s in systems]
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False, bbox_to_anchor=(0.003, top_y - 2 * line),
               labelcolor=c["text"], handlelength=1.2, handleheight=0.8)
    fig.tight_layout(rect=(0, 0, 1, top_y - 3.2 * line))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", help="JSON from python -m jev.bench --out")
    p.add_argument("--out", help="image path (.png or .svg); default: next to the JSON")
    p.add_argument("--theme", choices=sorted(THEMES), default="light")
    p.add_argument("--name", help="label for this model (default: from the JSON)")
    p.add_argument("--baseline", metavar="JSON", help="a second jev.bench result to draw as a baseline, "
                   "e.g. from --zero_shot, to show what training added")
    p.add_argument("--baseline_name", default="zero-shot")
    args = p.parse_args()

    data = json.loads(Path(args.results).read_text())
    out = Path(args.out) if args.out else Path(args.results).with_suffix(".png")
    baseline = json.loads(Path(args.baseline).read_text()) if args.baseline else None
    plot(data, out, args.theme, args.name, baseline, args.baseline_name)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

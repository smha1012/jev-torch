"""Plot JevBench results (the JSON written by `python -m jev.bench --out`).

    python3 -m jev.plot_bench bench.json                    # -> bench.png
    python3 -m jev.plot_bench bench.json --out bench.svg --theme dark

Two panels on the same slices: accuracy (higher is better) and ECE (lower is better), each comparing
this model with TypeSafe Jev 1.13.0. Needs matplotlib (pip install matplotlib).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Reference data-viz palette: categorical slots 1-2, validated for colour-vision deficiency in both modes.
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "ours": "#2a78d6", "teacher": "#eb6834"},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#3a3a37",
             "ours": "#3987e5", "teacher": "#d95926"},
}
KIND_ORDER = {"choice": 0, "noul": 1, "score": 2}


def plot(data: dict, out: Path, theme: str = "light", model_name: str | None = None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = THEMES[theme]
    slices = data["slices"]
    # Group by question kind, then strongest-to-weakest for the teacher, so gaps read top to bottom.
    names = sorted(slices, key=lambda s: (KIND_ORDER.get(slices[s]["kind"], 9), -slices[s]["teacher"]["acc"]))
    labels = [f"{s}  ·  {slices[s]['kind']}" for s in names]
    ours_label = model_name or Path(str(data.get("model", "this model"))).name or "this model"
    teacher_label = f"TypeSafe {data.get('teacher_run', 'Jev')}"

    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.52 * len(names) + 2.4), sharey=True,
                             gridspec_kw={"width_ratios": [1.6, 1]})
    fig.patch.set_facecolor(c["surface"])

    h, gap = 0.36, 0.04  # thin bars, a small surface gap between the pair
    y = list(range(len(names)))
    panels = [("acc", "Accuracy (higher is better)", axes[0], "{:.0%}"),
              ("ece", "Calibration error, ECE (lower is better)", axes[1], "{:.3f}")]
    for key, title, ax, fmt in panels:
        ax.set_facecolor(c["surface"])
        ours = [slices[s]["ours"][key] for s in names]
        theirs = [slices[s]["teacher"][key] for s in names]
        top = max(ours + theirs)
        b1 = ax.barh([v - (h + gap) / 2 for v in y], ours, height=h, color=c["ours"], label=ours_label)
        b2 = ax.barh([v + (h + gap) / 2 for v in y], theirs, height=h, color=c["teacher"], label=teacher_label)
        for bars, vals in ((b1, ours), (b2, theirs)):
            for bar, val in zip(bars, vals):
                ax.text(bar.get_width() + top * 0.012, bar.get_y() + bar.get_height() / 2, fmt.format(val),
                        va="center", ha="left", fontsize=8.5, color=c["muted"])
        ax.set_xlim(0, 1.09 if key == "acc" else top * 1.2)  # room for the value labels
        ax.set_title(title, loc="left", fontsize=11, color=c["text"], pad=10)
        ax.grid(axis="x", color=c["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(c["grid"])
        ax.tick_params(colors=c["muted"], length=0)
        if key == "acc":
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, color=c["text"])
    axes[0].invert_yaxis()

    o, t = data["overall"]["ours"], data["overall"]["teacher"]
    height_in = fig.get_figheight()
    line = 0.3 / height_in  # one text line, in figure-fraction units
    top_y = 1 - 0.18 / height_in
    fig.text(0.01, top_y, f"JevBench: {ours_label} vs {teacher_label}", ha="left", va="top", fontsize=13,
             color=c["text"], fontweight="bold")
    fig.text(0.01, top_y - line, f"All {o['n']:,} cases: accuracy {o['acc']:.1%} vs {t['acc']:.1%} "
             f"({o['acc'] / t['acc']:.1%} of Jev's) · ECE {o['ece']:.3f} vs {t['ece']:.3f}",
             ha="left", va="top", fontsize=10, color=c["muted"])
    fig.legend(handles=[b1, b2], loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.003, top_y - 2 * line),
               labelcolor=c["text"], handlelength=1.2, handleheight=0.8)
    fig.tight_layout(rect=(0, 0, 1, top_y - 3.2 * line))
    fig.savefig(out, dpi=160, facecolor=c["surface"])
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", help="JSON from python -m jev.bench --out")
    p.add_argument("--out", help="image path (.png or .svg); default: next to the JSON")
    p.add_argument("--theme", choices=sorted(THEMES), default="light")
    p.add_argument("--name", help="label for this model (default: from the JSON)")
    args = p.parse_args()

    data = json.loads(Path(args.results).read_text())
    out = Path(args.out) if args.out else Path(args.results).with_suffix(".png")
    plot(data, out, args.theme, args.name)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

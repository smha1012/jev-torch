"""Upload a trained checkpoint to the Hugging Face Hub with a generated model card.

    python -m jev.push_to_hub --ckpt runs/jev-9b/best --repo your-name/jev-9b            # private
    python -m jev.push_to_hub --ckpt runs/jev-9b/best --repo your-name/jev-9b --public
    python -m jev.push_to_hub --ckpt runs/jev-9b/best --repo your-name/jev-9b --dry_run out/   # write card only

Training already pushes automatically (train.hf_push); this command is for manual uploads.
The token comes from HF_TOKEN, .env.local, or `hf auth login`. The card pulls training settings from
<run>/config.yaml and metrics from <run>/report.json when they exist next to the checkpoint.
Afterwards anyone can run:  JEVPredictor("your-name/jev-9b")
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .env import load_env
from .hub import HubUploader, stage_checkpoint
from .losses import JEVLoss


def _metrics_table(report: dict) -> str:
    rows = []
    for split in ("test_set_30k", "ood"):
        cal = report.get(split, {}).get("calibrated")
        if not cal:
            continue
        rows.append((f"{split} (all)", cal))
        rows += [(f"{split} / {k}", m) for k, m in cal.get("by_kind", {}).items()]
    if not rows:
        return ""
    lines = ["| split | n | top-1 agreement | KL to teacher | ECE |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {name} | {m['n']:,} | {m['acc']:.3f} | {m['kl']:.4f} | {m['ece']:.4f} |" for name, m in rows]
    return "\n".join(lines)


def build_model_card(ckpt: Path, repo_id: str, status: str | None = None) -> str:
    """Model card for a checkpoint. `status` marks an intermediate (e.g. per-epoch) upload."""
    meta = json.loads((ckpt / "jev_config.json").read_text())
    base = meta["model"]["name"]
    run_dir = ckpt.parent
    config = yaml.safe_load((run_dir / "config.yaml").read_text()) if (run_dir / "config.yaml").exists() else {}
    report = json.loads((run_dir / "report.json").read_text()) if (run_dir / "report.json").exists() else {}
    train = config.get("train", {})
    data = config.get("data", {})
    temps = meta.get("temperature", {})

    front = {
        "license": "apache-2.0",
        "base_model": base,
        "library_name": "peft",
        "pipeline_tag": "text-classification",
        "tags": ["jev", "jev-torch", "decision-model", "calibration", "lora", "distillation"],
    }
    from .config import DataConfig
    from .sources import hub_dataset

    data_cfg = DataConfig(**{k: v for k, v in data.items() if k in DataConfig.__dataclass_fields__})
    if hub_dataset(data_cfg):
        front["datasets"] = [hub_dataset(data_cfg)]

    metrics = _metrics_table(report)
    from .sources import describe_source

    settings = [
        f"- **Training data:** {describe_source(data_cfg)}",
        f"- **Base model:** [`{base}`](https://huggingface.co/{base}) (frozen) + LoRA r={meta['model']['lora_r']}, "
        f"α={meta['model']['lora_alpha']} + 24-slot fp32 decision head",
        f"- **Loss:** `{JEVLoss(train.get('loss')).describe()}`",
    ]
    if train:
        settings.append(f"- **Optimization:** {train.get('max_steps') or '1 epoch'} steps × "
                        f"{train.get('global_batch_size')} rows, LoRA lr {train.get('lr')}, head lr {train.get('head_lr')}")
    if temps:
        settings.append("- **Temperatures:** " + ", ".join(f"{k} {v:.3f}" for k, v in temps.items()))

    status_block = f"\n> [!NOTE]\n> {status}\n" if status else ""
    body = f"""
# {repo_id.split('/')[-1]}
{status_block}
An **unofficial JEV-style decision model** trained with [jev-torch](https://github.com/smha1012/jev-torch).
It reads a *state*, a *question* and a list of *options*, runs one forward pass, and returns a
calibrated probability for every option.

## Usage

```bash
pip install git+https://github.com/smha1012/jev-torch.git
```

```python
from jev import JEVPredictor

jev = JEVPredictor("{repo_id}")
jev.predict(
    kind="noul",
    state="The canary shows p99 latency up 40% after the deploy.",
    question="Should the rollout be paused?",
    options=["false", "true"],
)
```

Question kinds: `noul` (exactly 2 options, false-like then true-like), `choice` (2–16 options),
`score` (ordered levels 0–5).

## Training

{chr(10).join(settings)}

## Evaluation

{metrics or "_No evaluation report was found next to this checkpoint._"}

`top-1 agreement` = how often the model's top option matches the teacher's top option.

## Limitations

- Probabilities mirror the teacher model's judgments; they are not ground truth.
- Trained mostly on synthetic English scenarios; validate on your own domain before relying on it.

## License and attribution

Weights: Apache-2.0. The training corpus is Apache-2.0, but part of its labels are outputs of the closed
TypeSafe Jev 1.13 model; check that model's terms before commercial use.
This is an independent project, not affiliated with TypeSafe AI or autotrust.
"""
    return "---\n" + yaml.safe_dump(front, sort_keys=False) + "---\n" + body


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint directory, e.g. runs/jev-9b/best")
    p.add_argument("--repo", required=True, help="Hub repo id, e.g. your-name/jev-9b")
    p.add_argument("--public", action="store_true", help="create the repo public (default: private)")
    p.add_argument("--tag", help="also tag this upload, e.g. v1")
    p.add_argument("--dry_run", metavar="DIR", help="write the staged upload to DIR instead of uploading")
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    if not (ckpt / "jev_config.json").exists():
        raise SystemExit(f"{ckpt} is not a jev-torch checkpoint")
    card = build_model_card(ckpt, args.repo)
    if args.dry_run:
        stage_checkpoint(ckpt, Path(args.dry_run), card)
        print(f"staged upload in {args.dry_run}")
        return

    load_env()
    hub = HubUploader.setup(args.repo, private=not args.public, run_name=ckpt.parent.name)
    if not hub.push(ckpt, card, "Upload jev-torch checkpoint", tag=args.tag):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

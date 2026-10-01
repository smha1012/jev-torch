"""Score options with a trained JEV checkpoint.

CLI:
    python -m jev.predict --ckpt runs/jev-9b/best --input examples/example.jsonl
    python -m jev.predict --ckpt runs/jev-9b/best --input data.jsonl --metrics

Python:
    from jev import JEVPredictor
    jev = JEVPredictor("runs/jev-9b/best")
    jev.predict(kind="choice", state="...", question="Root cause?", options=["a", "b", "c"])
    # -> {"a": 0.71, "b": 0.22, "c": 0.07}
"""

from __future__ import annotations

import argparse
import json

import torch

from .collate import JEVCollator, JEVDataset
from .distributed import pick_device, pick_dtype
from .losses import apply_temperatures, compute_metrics
from .model import JEVModel
from .schema import JEVExample


class JEVPredictor:
    def __init__(self, ckpt: str, device: str = "auto", max_length: int = 1024, batch_size: int = 16):
        self.device = pick_device(device)
        dtype = pick_dtype(self.device)
        device_map = {"": self.device.index or 0} if self.device.type == "cuda" else None
        self.model, self.tokenizer = JEVModel.load(ckpt, dtype=dtype, device_map=device_map)
        self.model.to(self.device).eval()
        self.autocast = self.device.type == "cuda" and dtype == torch.bfloat16
        self.collator = JEVCollator(self.tokenizer, max_length=max_length)
        self.batch_size = batch_size

    @torch.no_grad()
    def logits(self, examples: list[JEVExample]) -> list[torch.Tensor]:
        """Raw (uncalibrated) logits per example, one entry per option."""
        ds, out = JEVDataset(examples), []
        for s in range(0, len(examples), self.batch_size):
            items = [ds[i] for i in range(s, min(s + self.batch_size, len(examples)))]
            batch = {k: v.to(self.device) for k, v in self.collator(items).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
                lg = self.model(**batch).float().cpu()
            out.extend(lg[i, : len(it["options"])] for i, it in enumerate(items))
        return out

    def predict_batch(self, examples: list[JEVExample]) -> list[dict[str, float]]:
        temps = self.model.temperature.cpu()
        from .schema import KINDS

        res = []
        for ex, lg in zip(examples, self.logits(examples)):
            p = torch.softmax(lg / temps[KINDS.index(ex.kind)], dim=-1)
            res.append(dict(zip(ex.options, p.tolist())))
        return res

    def predict(self, state: str, question: str, options: list[str], kind: str = "choice") -> dict[str, float]:
        ex = JEVExample(state=state, question=question, options=options, kind=kind, label=0)
        return self.predict_batch([ex])[0]


def _read(path: str) -> list[JEVExample]:
    out = []
    for i, line in enumerate(l for l in open(path) if l.strip()):
        d = json.loads(line)
        d.setdefault("id", f"ex-{i}")
        if d.get("label") is None and d.get("target") is None:
            d["label"] = 0  # unlabeled input: dummy label, only used by --metrics
        out.append(JEVExample.from_dict(d))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input", required=True, help="JSONL in the JEV format")
    p.add_argument("--metrics", action="store_true", help="aggregate metrics (needs target/label)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    jev = JEVPredictor(args.ckpt, device=args.device, batch_size=args.batch_size)
    examples = _read(args.input)
    if args.metrics:
        from .collate import KIND_SLOTS
        from .schema import KINDS

        K = max(n for _, n in KIND_SLOTS.values())
        lg = jev.logits(examples)
        logits = torch.full((len(lg), K), float("-inf"))
        target, mask = torch.zeros(len(lg), K), torch.zeros(len(lg), K, dtype=torch.bool)
        for i, (ex, l) in enumerate(zip(examples, lg)):
            logits[i, : len(l)], target[i, : len(l)], mask[i, : len(l)] = l, torch.tensor(ex.target_dist()), True
        kind = torch.tensor([KINDS.index(ex.kind) for ex in examples])
        label = torch.tensor([ex.label for ex in examples])
        base = dict(target=target, label=label, option_mask=mask)
        print("uncalibrated", compute_metrics(logits=logits, **base))
        print("calibrated  ", compute_metrics(logits=apply_temperatures(logits, kind, jev.model.temperature.cpu()), **base))
        return

    for ex, probs in zip(examples, jev.predict_batch(examples)):
        print(f"\n[{ex.id}] ({ex.kind}) {ex.state[:120]}{'...' if len(ex.state) > 120 else ''}")
        print(f"  Q: {ex.question}")
        for opt, pr in sorted(probs.items(), key=lambda x: -x[1]):
            print(f"  {pr:6.3f}  {opt}")


if __name__ == "__main__":
    main()

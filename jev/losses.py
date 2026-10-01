"""Losses, temperature calibration and metrics over option distributions.

Every function takes padded tensors: logits/target [B, K], option_mask [B, K] (bool).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .schema import KINDS


def kl_per_example(logits, target, option_mask):
    """KL(target || softmax(logits)) per row. Equals cross-entropy when target is one-hot."""
    log_p = F.log_softmax(logits, dim=-1).masked_fill(~option_mask, 0.0)
    return (torch.xlogy(target, target) - target * log_p).sum(-1)


def rps_per_example(logits, target, option_mask):
    """Ranked Probability Score: squared distance between predicted and target CDFs, per row.

    For ordinal options (score 0-5): predicting 4 when the teacher says 5 costs less than 0.
    """
    p = torch.softmax(logits, dim=-1).masked_fill(~option_mask, 0.0)
    k = option_mask.sum(-1).clamp(min=2).float()
    return ((p.cumsum(-1) - target.cumsum(-1)) ** 2).sum(-1) / (k - 1)


def ce_per_example(logits, label, **_):
    """Cross-entropy against the hard label only (ignores the soft target)."""
    return -F.log_softmax(logits, dim=-1).gather(1, label.unsqueeze(1)).squeeze(1)


def brier_per_example(logits, target, option_mask):
    """Squared error between the predicted and target probability vectors (Open-Jev uses NLL + Brier)."""
    p = torch.softmax(logits, dim=-1).masked_fill(~option_mask, 0.0)
    return ((p - target) ** 2).sum(-1)


# ---------------------------------------------------------------------------
# Loss registry: train.loss picks a weighted combination by name
# ---------------------------------------------------------------------------

LOSSES: dict[str, dict] = {}


def register_loss(name: str, default_kinds: tuple[str, ...] | None = None):
    """Register a per-example loss `fn(logits, target, option_mask, label, kind) -> [B]`.

    `default_kinds` restricts where it applies unless the config says otherwise (RPS only makes sense on
    ordinal `score` rows). Example of a custom loss:

        @register_loss("js")
        def js(logits, target, option_mask, **_):
            ...
    """
    def deco(fn):
        LOSSES[name] = {"fn": fn, "kinds": default_kinds}
        return fn

    return deco


register_loss("kl")(lambda logits, target, option_mask, **_: kl_per_example(logits, target, option_mask))
register_loss("ce")(lambda logits, label, **_: ce_per_example(logits, label))
register_loss("brier")(lambda logits, target, option_mask, **_: brier_per_example(logits, target, option_mask))
register_loss("rps", default_kinds=("score",))(
    lambda logits, target, option_mask, **_: rps_per_example(logits, target, option_mask))

DEFAULT_LOSS = {"kl": 1.0, "rps": {"weight": 0.5, "kinds": ["score"]}}  # JEV-9B / 27B recipe


class JEVLoss:
    """Weighted sum of registered losses, configured as `train.loss`:

        loss:
          kl: 1.0                                  # name: weight
          rps: {weight: 0.5, kinds: [score]}       # name: {weight, kinds}
          # brier: 0.25                            # e.g. the Open-Jev style NLL + Brier

    Each term is averaged per row, restricted to its kinds, multiplied by the row weight
    (data.loss_weights) and summed. The batch mean is over all rows, so down-weighted or excluded rows
    contribute less instead of inflating the others.
    """

    def __init__(self, spec: dict | None = None):
        spec = DEFAULT_LOSS if spec is None else spec
        if not spec:
            raise ValueError("train.loss is empty")
        self.terms = []
        for name, value in spec.items():
            if name not in LOSSES:
                raise ValueError(f"unknown loss {name!r}; registered: {sorted(LOSSES)}")
            if isinstance(value, dict):
                bad = set(value) - {"weight", "kinds"}
                if bad:
                    raise ValueError(f"loss {name!r}: unknown keys {sorted(bad)} (use weight, kinds)")
                weight, kinds = float(value.get("weight", 1.0)), value.get("kinds", LOSSES[name]["kinds"])
            else:
                weight, kinds = float(value), LOSSES[name]["kinds"]
            if kinds is not None:
                unknown = set(kinds) - set(KINDS)
                if unknown:
                    raise ValueError(f"loss {name!r}: unknown kinds {sorted(unknown)}; use {list(KINDS)}")
                kinds = tuple(kinds)
            if weight != 0:
                self.terms.append((name, weight, kinds))
        if not self.terms:
            raise ValueError("train.loss has no term with a non-zero weight")

    def __call__(self, logits, target, option_mask, kind, label, weight=None):
        """Returns (scalar loss, {term name: detached mean}) for logging."""
        total = torch.zeros(logits.size(0), device=logits.device)
        parts = {}
        for name, w, kinds in self.terms:
            per_row = LOSSES[name]["fn"](logits=logits, target=target, option_mask=option_mask,
                                         label=label, kind=kind)
            if kinds is not None:
                mask = torch.zeros_like(per_row)
                for k in kinds:
                    mask = mask + (kind == KINDS.index(k)).float()
                per_row = per_row * mask
            parts[name] = per_row.detach().mean()
            total = total + w * per_row
        if weight is not None:
            total = total * weight
        return total.mean(), parts

    def describe(self) -> str:
        return " + ".join(f"{w:g}·{n}" + (f"[{','.join(k)}]" if k else "") for n, w, k in self.terms)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def fit_temperature(logits, target, option_mask, t_min=0.05, t_max=20.0, n=400) -> float:
    """Temperature scaling: T minimizing mean KL of softmax(logits / T) to the target.

    Dense log-spaced grid (a 1-D problem, so it cannot diverge); T=1.0 is always a candidate, so the
    calibration split can never get worse. Changes confidence only, never the argmax.
    """
    logits, target, option_mask = logits.float(), target.float(), option_mask
    grid = torch.cat([torch.logspace(torch.log10(torch.tensor(t_min)), torch.log10(torch.tensor(t_max)), n),
                      torch.ones(1)])
    with torch.no_grad():
        losses = torch.stack([kl_per_example(logits / t, target, option_mask).mean() for t in grid])
    return float(grid[losses.argmin()])


def fit_temperatures(logits, target, option_mask, kind) -> torch.Tensor:
    """One temperature per kind (noul, choice, score); kinds absent from the data keep T=1."""
    temps = torch.ones(len(KINDS))
    for k in range(len(KINDS)):
        rows = kind == k
        if rows.any():
            temps[k] = fit_temperature(logits[rows], target[rows], option_mask[rows])
    return temps


def apply_temperatures(logits, kind, temps):
    return logits / temps.to(logits.device)[kind].unsqueeze(-1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_metrics(logits, target, label, option_mask, n_bins: int = 15, **_):
    """acc: top-1 agrees with label (= teacher's argmax); kl: to the teacher distribution;
    ece: confidence vs. accuracy gap; tv: total-variation distance to the teacher distribution."""
    logits, target = logits.float(), target.float()
    p = torch.softmax(logits, dim=-1).masked_fill(~option_mask, 0.0)
    conf, pred = p.max(-1)
    correct = (pred == label).float()

    ece = torch.zeros(())
    edges = torch.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean() * (correct[m].mean() - conf[m].mean()).abs()

    return {
        "acc": correct.mean().item(),
        "kl": kl_per_example(logits, target, option_mask).mean().item(),
        "brier": ((p - target) ** 2).sum(-1).mean().item(),
        "tv": 0.5 * (p - target).abs().sum(-1).mean().item(),
        "ece": ece.item(),
        "conf": conf.mean().item(),
        "n": int(len(label)),
    }


def metrics_by(res: dict, groups: list, **kw) -> dict:
    """compute_metrics per group value (e.g. per kind or per family)."""
    out = {}
    for g in sorted(set(groups), key=str):
        ix = torch.tensor([i for i, x in enumerate(groups) if x == g])
        out[str(g)] = compute_metrics(**{k: v[ix] for k, v in res.items()}, **kw)
    return out

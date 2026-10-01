"""Losses, temperature calibration and metrics over option distributions.

Every function takes padded tensors: logits/target [B, K], option_mask [B, K] (bool).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .schema import KINDS

SCORE = KINDS.index("score")


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


def jev_loss(logits, target, option_mask, kind, rps_weight: float = 0.5, weight=None):
    """KL over active slots on every row + rps_weight * RPS on score rows (JEV-9B recipe).

    `weight` [B] down-weights rows (e.g. placeholder labels); the batch mean is still over B rows,
    so down-weighted rows simply contribute less instead of inflating the others.
    """
    loss = kl_per_example(logits, target, option_mask)
    if rps_weight > 0:
        is_score = (kind == SCORE).float()
        loss = loss + rps_weight * is_score * rps_per_example(logits, target, option_mask)
    if weight is not None:
        loss = loss * weight
    return loss.mean()


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

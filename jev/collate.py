"""Prompting, slot assignment, tokenization and batching.

Every option is scored by one of 24 fixed slots, each tied to a verbalizer token:

    slots 0-1   noul    false, true
    slots 2-7   score   0 1 2 3 4 5
    slots 8-23  choice  A ... P      (so a choice question has at most 16 options)

The model reads one prompt per example and its final hidden state is mapped to all 24 slot logits;
only the slots belonging to the example's kind are "active" and enter the softmax.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from .schema import KINDS, JEVExample

CHOICE_LETTERS = "ABCDEFGHIJKLMNOP"
SLOT_VERBALIZERS = ["false", "true", "0", "1", "2", "3", "4", "5", *CHOICE_LETTERS]
KIND_SLOTS = {"noul": (0, 2), "score": (2, 6), "choice": (8, 16)}  # kind -> (first slot, n slots)
NUM_SLOTS = len(SLOT_VERBALIZERS)


def check_example(ex: JEVExample) -> None:
    n_slots = KIND_SLOTS[ex.kind][1]
    if len(ex.options) > n_slots:
        raise ValueError(f"[{ex.id}] {ex.kind} supports at most {n_slots} options, got {len(ex.options)}")
    if ex.kind == "noul" and len(ex.options) != 2:
        raise ValueError(f"[{ex.id}] noul needs exactly 2 options (false-like first, true-like second)")


def format_options(kind: str, options: list[str]) -> str:
    if kind == "choice":
        return "\n".join(f"{CHOICE_LETTERS[j]}. {o}" for j, o in enumerate(options))
    return "\n".join(f"- {o}" for o in options)


def prompt_parts(kind: str, state: str, question: str, options: list[str]) -> tuple[str, str, str]:
    """(prefix, state, suffix). The state is kept separate so it alone gets truncated."""
    prefix = f"[kind] {kind}\n[state] "
    suffix = f"\n[question] {question}\n[options]\n{format_options(kind, options)}\n[decision]:"
    return prefix, state, suffix


class JEVDataset(Dataset):
    def __init__(self, examples: list[JEVExample], shuffle_prob: float = 0.0,
                 loss_weights: dict[str, dict[str, float]] | None = None):
        for ex in examples:
            check_example(ex)
        self.examples = examples
        self.shuffle_prob = shuffle_prob
        self.loss_weights = loss_weights or {}

    def weight(self, ex: JEVExample) -> float:
        w = 1.0
        for key, table in self.loss_weights.items():
            w *= table.get(ex.get(key), 1.0)
        return w

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        options, target, label = ex.options, ex.target_dist(), ex.label
        # Only `choice` is permuted: noul/score options map to fixed verbalizer slots by position.
        if ex.kind == "choice" and self.shuffle_prob > 0 and random.random() < self.shuffle_prob:
            perm = list(range(len(options)))
            random.shuffle(perm)
            options = [options[p] for p in perm]
            target = [target[p] for p in perm]
            label = perm.index(label)
        return {"kind": ex.kind, "state": ex.state, "question": ex.question,
                "options": options, "target": target, "label": label, "weight": self.weight(ex)}


@dataclass
class JEVCollator:
    """Batch of dataset items -> tensors.

    input_ids / attention_mask [B, L]  right-padded; the model pools at attention_mask.sum()-1
    slot_index [B, K]   which of the 24 slots scores option j (0 for padding, masked out)
    option_mask [B, K]  bool, real options
    target [B, K]       teacher distribution
    label [B]           argmax / hard label
    kind [B]            index into KINDS
    weight [B]          per-row loss weight (data.loss_weights)
    """

    tokenizer: Any
    max_length: int = 1024
    head_frac: float = 0.6  # when the state is too long, keep 60% of its budget from the start, 40% from the end

    def __call__(self, batch):
        B = len(batch)
        K = max(len(b["options"]) for b in batch)
        slot_index = torch.zeros(B, K, dtype=torch.long)
        option_mask = torch.zeros(B, K, dtype=torch.bool)
        target = torch.zeros(B, K)
        ids = []
        for i, b in enumerate(batch):
            k = len(b["options"])
            first, _ = KIND_SLOTS[b["kind"]]
            slot_index[i, :k] = torch.arange(first, first + k)
            option_mask[i, :k] = True
            target[i, :k] = torch.tensor(b["target"])
            ids.append(self._encode(*prompt_parts(b["kind"], b["state"], b["question"], b["options"])))

        L = max(len(x) for x in ids)
        input_ids = torch.full((B, L), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(B, L, dtype=torch.long)
        for r, x in enumerate(ids):
            input_ids[r, : len(x)] = torch.tensor(x)
            attention_mask[r, : len(x)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "slot_index": slot_index,
            "option_mask": option_mask,
            "target": target,
            "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
            "kind": torch.tensor([KINDS.index(b["kind"]) for b in batch], dtype=torch.long),
            "weight": torch.tensor([b.get("weight", 1.0) for b in batch]),
        }

    def _encode(self, prefix: str, state: str, suffix: str) -> list[int]:
        enc = lambda s: self.tokenizer(s, add_special_tokens=False)["input_ids"]
        p, s, q = enc(prefix), enc(state), enc(suffix)
        budget = self.max_length - len(p) - len(q)
        if budget <= 0:  # question + options alone overflow: keep their end, which holds [decision]
            return (p + q)[-self.max_length :]
        if len(s) > budget:
            head = int(budget * self.head_frac)
            s = s[:head] + s[len(s) - (budget - head) :]
        return p + s + q


class LengthGroupedBatchSampler:
    """Deterministic, resumable, DDP-aware batch sampler.

    Each epoch: shuffle -> split into mega-batches of `group` micro-batches -> sort each mega-batch
    by length (less padding) -> cut into micro-batches -> shuffle micro-batch order. Rank r takes
    micro-batches r, r+W, r+2W, ... The count is trimmed so every rank does the same number of
    optimizer steps. `start` skips already-consumed micro-batches when resuming mid-epoch.
    """

    def __init__(self, lengths, batch_size, *, shuffle=True, group_by_length=True, seed=0,
                 rank=0, world_size=1, multiple_of=1, group=64):
        self.lengths, self.batch_size = lengths, batch_size
        self.shuffle, self.group_by_length, self.seed = shuffle, group_by_length, seed
        self.rank, self.world_size, self.multiple_of, self.group = rank, world_size, multiple_of, group
        self.set_epoch(0)

    def set_epoch(self, epoch: int, start: int = 0):
        self.epoch, self.start = epoch, start
        self._batches = self._my_batches()

    def _all_batches(self) -> list[list[int]]:
        idx = list(range(len(self.lengths)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(idx)
        if self.group_by_length:
            mega = self.batch_size * self.group
            idx = [i for s in range(0, len(idx), mega)
                   for i in sorted(idx[s : s + mega], key=lambda i: self.lengths[i])]
        batches = [idx[s : s + self.batch_size] for s in range(0, len(idx), self.batch_size)]
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def _my_batches(self) -> list[list[int]]:
        batches = self._all_batches()
        per_rank = len(batches) // self.world_size
        per_rank -= per_rank % self.multiple_of
        return batches[self.rank :: self.world_size][:per_rank]

    def __len__(self):  # full epoch length for this rank, including already-consumed batches
        return len(self._batches)

    def __iter__(self):
        yield from self._batches[self.start :]

"""Run configuration: YAML file + dotted command-line overrides.

    cfg = load_config("configs/jev-0.5b.yaml", ["train.lr=1e-4", "data.max_train_examples=20000"])
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3.5-9B"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # JEV-9B targets: Qwen3.5 linear attention (in_proj_qkv, in_proj_z, out_proj), full attention
    # (q/k/v/o_proj) and MLP. Names absent from the backbone are skipped.
    lora_targets: list[str] = field(default_factory=lambda: [
        "in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    gradient_checkpointing: bool = True
    load_in_4bit: bool = False  # QLoRA: frozen backbone in NF4, e.g. 27B on a single 48 GB GPU
    attn_implementation: str | None = None  # None = transformers default (sdpa)


@dataclass
class DataConfig:
    source: str = "jev_distill"  # a name registered in jev.sources
    path: str | None = None  # used by the `jsonl` source: directory holding <split>.jsonl
    hf_dataset: str | None = None  # override the Hub dataset a source reads (e.g. a mirror/snapshot)
    hf_revision: str | None = None  # pin a Hub dataset commit / tag / branch for reproducibility
    train_split: str = "train"
    val_split: str = "validation"
    calib_split: str = "calibration"
    eval_splits: list[str] = field(default_factory=lambda: ["test_set_30k", "ood"])
    include: dict[str, list[str]] = field(default_factory=dict)  # keep rows whose field is in list
    exclude: dict[str, list[str]] = field(default_factory=dict)  # drop rows whose field is in list
    max_train_examples: int | None = None
    max_eval_examples: int | None = None  # cap for calib and eval splits
    max_length: int = 1024  # an over-long state is cut to its first 60% + last 40% of the budget
    shuffle_prob: float = 0.3  # chance of permuting a `choice` row's options (targets follow)
    # Per-row loss weight by field value, e.g. {source: {yuri_v1: 0.1}}. Rows not matched weigh 1.0.
    loss_weights: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class TrainConfig:
    output_dir: str = "runs/jev"
    num_gpus: int = 1  # `python -m jev.launch` starts one process per GPU via torchrun
    epochs: int = 1
    max_steps: int | None = None
    global_batch_size: int = 128  # rows per optimizer step, across all GPUs and accumulation
    micro_batch_size: int = 16  # rows per forward pass per GPU
    lr: float = 1e-4  # LoRA
    head_lr: float = 2e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.98])
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.0  # cosine floor as a fraction of peak lr (JEV-27B: 0.02)
    max_grad_norm: float = 1.0
    # Weighted loss terms by name (jev.losses.LOSSES: kl, ce, brier, rps, or your own register_loss).
    # A term is `name: weight` or `name: {weight, kinds}`. Default = the JEV-9B / 27B recipe.
    loss: dict = field(default_factory=lambda: {"kl": 1.0, "rps": {"weight": 0.5, "kinds": ["score"]}})
    eval_every: int = 500
    val_max_examples: int | None = 4000  # val subset used for checkpoint selection during training
    save_every: int = 250  # resumable state in <output_dir>/last (pods can be preempted)
    log_every: int = 10
    eval_batch_size: int = 32
    num_workers: int = 2
    group_by_length: bool = True
    device: str = "auto"
    dtype: str = "auto"  # auto: bf16 weights + autocast on CUDA, fp32 elsewhere
    seed: int = 42
    wandb_project: str | None = None  # set to enable Weights & Biases logging (key: WANDB_API_KEY)
    wandb_entity: str | None = None  # team / user; None = your default entity
    wandb_run_name: str | None = None  # None = output_dir name
    wandb_tags: list[str] = field(default_factory=list)
    # Hugging Face Hub: "auto" = push to <token account>/<output_dir name> when an HF token exists,
    # "owner/name" = push there (token required), null = never. Pushes every epoch end + final.
    hf_push: str | None = "auto"
    hf_private: bool = True


@dataclass
class JEVConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> JEVConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {} if path else {}
    for item in overrides or []:
        key, _, value = item.partition("=")
        section, _, name = key.partition(".")
        if not name:
            raise ValueError(f"override must look like section.field=value, got {item!r}")
        raw.setdefault(section, {})[name] = yaml.safe_load(value)

    sections = {"model": ModelConfig, "data": DataConfig, "train": TrainConfig}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    built = {}
    for name, cls in sections.items():
        values = raw.get(name) or {}
        valid = {f.name for f in dataclasses.fields(cls)}
        bad = set(values) - valid
        if bad:
            raise ValueError(f"unknown {name} fields: {sorted(bad)}")
        built[name] = cls(**values)
    return JEVConfig(**built)

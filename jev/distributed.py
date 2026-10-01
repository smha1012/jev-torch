"""Minimal torch.distributed helpers. Single-process runs work unchanged (world_size = 1).

Launch multi-GPU with torchrun, which sets RANK / WORLD_SIZE / LOCAL_RANK:
    torchrun --nproc_per_node 8 -m jev.train --config configs/jev-9b.yaml
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass
class DistContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: torch.device = torch.device("cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    def barrier(self):
        if self.enabled:
            dist.barrier()

    def all_gather_object(self, obj) -> list:
        if not self.enabled:
            return [obj]
        out = [None] * self.world_size
        dist.all_gather_object(out, obj)
        return out

    def mean(self, value: float) -> float:
        if not self.enabled:
            return value
        t = torch.tensor([value], device=self.device, dtype=torch.float64)
        dist.all_reduce(t)
        return t.item() / self.world_size

    def sum(self, value: float) -> float:
        if not self.enabled:
            return value
        t = torch.tensor([value], device=self.device, dtype=torch.float64)
        dist.all_reduce(t)
        return t.item()

    def print(self, *args, **kwargs):
        if self.is_main:
            print(*args, **kwargs, flush=True)


def setup(device: str = "auto") -> DistContext:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        # Long timeout: rank 0 may be busy saving a 27B checkpoint or downloading data while others wait.
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
        return DistContext(dist.get_rank(), world_size, local_rank, torch.device("cuda", local_rank))
    return DistContext(device=pick_device(device))


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device, name: str = "auto") -> torch.dtype:
    if name != "auto":
        return getattr(torch, name)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32

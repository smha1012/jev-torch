"""Launch training with the GPU count taken from the config (train.num_gpus).

    python -m jev.launch --config configs/jev-27b.yaml [--set train.num_gpus=4 ...] [--resume]

num_gpus == 1 runs in this process; num_gpus > 1 re-executes under torchrun, one process per GPU.
"""

from __future__ import annotations

import os
import sys

from .config import load_config
from .train import main as train_main
from .train import parse_args


def main():
    argv = sys.argv[1:]
    args = parse_args(argv)
    cfg = load_config(args.config, args.set)
    n = cfg.train.num_gpus
    if n <= 1 or "LOCAL_RANK" in os.environ:
        train_main(argv)
        return
    try:
        import torch

        available = torch.cuda.device_count()
    except Exception:
        available = 0
    if available < n:
        sys.exit(f"config asks for train.num_gpus={n} but only {available} CUDA device(s) are visible")
    cmd = ["torchrun", "--standalone", f"--nproc_per_node={n}", "-m", "jev.train", *argv]
    print("exec:", " ".join(cmd), flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()

"""Train a general JEV model. Usually started through the launcher, which reads train.num_gpus:

    python -m jev.launch --config configs/jev-9b.yaml
    python -m jev.launch --config configs/jev-9b.yaml --set train.max_steps=100 --resume

Direct single-process / manual torchrun use also works:

    python -m jev.train --config configs/debug.yaml
    torchrun --nproc_per_node 4 -m jev.train --config configs/jev-9b.yaml
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import torch

from .config import load_config
from .distributed import cleanup, pick_dtype, setup
from .env import load_env
from .hub import HubUploader
from .model import JEVModel
from .sources import load_split
from .trainer import Trainer


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="SECTION.FIELD=VALUE", help="config overrides")
    p.add_argument("--resume", action="store_true", help="continue from <output_dir>/last")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config, args.set)
    load_env()  # HF_TOKEN / WANDB_API_KEY from .env.local; real environment variables win
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ctx = setup(cfg.train.device)
    dtype = pick_dtype(ctx.device, cfg.train.dtype)
    random.seed(cfg.train.seed + ctx.rank)
    torch.manual_seed(cfg.train.seed)
    ctx.print(f"model={cfg.model.name} device={ctx.device} dtype={dtype} world_size={ctx.world_size}")

    # Check Hub access before spending time on data and weights: a bad token or missing write
    # permission should stop the run now, not after hours of training. Every rank exits together.
    hub, hub_error = None, None
    if ctx.is_main:
        try:
            hub = HubUploader.setup(cfg.train.hf_push, cfg.train.hf_private, Path(cfg.train.output_dir).name)
        except SystemExit as e:
            hub_error = str(e)
    hub_error = ctx.all_gather_object(hub_error)[0]
    if hub_error:
        cleanup()
        raise SystemExit(hub_error)

    # Rank 0 downloads/caches datasets first; the others then read from the cache.
    def load_all():
        d, seed = cfg.data, cfg.train.seed
        train = load_split(d, d.train_split, d.max_train_examples, seed)
        val = load_split(d, d.val_split, cfg.train.val_max_examples, seed)
        calib = load_split(d, d.calib_split, d.max_eval_examples, seed) if d.calib_split else None
        evals = {s: load_split(d, s, d.max_eval_examples, seed) for s in d.eval_splits}
        return train, val, calib, evals

    if not ctx.is_main:
        ctx.barrier()
    train, val, calib, evals = load_all()
    if ctx.is_main:
        ctx.barrier()
    ctx.print(f"train={len(train)} val={len(val)} calib={len(calib or [])} "
              + " ".join(f"{k}={len(v)}" for k, v in evals.items()))

    # CUDA: load weights straight onto this rank's GPU (a 27B model must not be staged in CPU RAM x N ranks).
    device_map = {"": ctx.device.index or 0} if ctx.device.type == "cuda" else None
    if not ctx.is_main:
        ctx.barrier()
    model, tokenizer = JEVModel.build(cfg.model, dtype=dtype, device_map=device_map)
    if ctx.is_main:
        ctx.barrier()
    model.to(ctx.device)  # moves the fp32 head (and everything on MPS/CPU)
    if ctx.is_main:
        model.backbone.print_trainable_parameters()

    trainer = Trainer(cfg, model, tokenizer, ctx, dtype, train, val, calib, evals, hub=hub)
    trainer.fit(resume=args.resume)
    trainer.calibrate_and_report()
    cleanup()


if __name__ == "__main__":
    main()

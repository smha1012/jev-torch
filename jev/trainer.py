"""Trainer: fit -> select best by val KL -> per-kind temperature calibration -> evaluation report.

The Trainer only needs a JEVModel and lists of JEVExample, so the same class trains the general
model from scratch (JEVModel.build) or continues from a checkpoint (JEVModel.load(is_trainable=True)).
"""

from __future__ import annotations

import contextlib
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .collate import KIND_SLOTS, JEVCollator, JEVDataset, LengthGroupedBatchSampler
from .config import JEVConfig
from .distributed import DistContext
from .losses import JEVLoss, apply_temperatures, compute_metrics, fit_temperatures, metrics_by
from .model import JEVModel
from .push_to_hub import build_model_card
from .schema import KINDS, JEVExample

MAX_K = max(n for _, n in KIND_SLOTS.values())


def _approx_len(ex: JEVExample) -> int:
    return len(ex.state) + len(ex.question) + sum(len(o) for o in ex.options)


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "/"))
        elif isinstance(v, (int, float)):
            out[key] = v
    return out


def init_wandb(cfg: JEVConfig, out: Path, world_size: int):
    """Start (or resume) a W&B run on rank 0. Returns None when disabled.

    The run id is stored in <output_dir>/wandb_id so a resumed training continues the same run.
    Without WANDB_API_KEY (or a prior `wandb login`) it logs offline instead of blocking on a
    prompt; sync later with `wandb sync <output_dir>/wandb`.
    """
    tc = cfg.train
    if not tc.wandb_project:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb_project is set but wandb is not installed (pip install wandb); logging disabled")
        return None
    import netrc
    import secrets
    import os

    logged_in = bool(os.environ.get("WANDB_API_KEY"))
    if not logged_in:
        try:
            logged_in = netrc.netrc().authenticators("api.wandb.ai") is not None
        except (FileNotFoundError, netrc.NetrcParseError):
            pass
    mode = os.environ.get("WANDB_MODE") or ("online" if logged_in else "offline")
    if mode == "offline":
        print("W&B: no API key found, logging offline to", out / "wandb")

    id_file = out / "wandb_id"
    run_id = id_file.read_text().strip() if id_file.exists() else secrets.token_hex(4)
    id_file.write_text(run_id)
    try:
        import subprocess

        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.splitlines()[0].strip()
    except Exception:
        gpu = "none"
    return wandb.init(
        project=tc.wandb_project, entity=tc.wandb_entity, name=tc.wandb_run_name or out.name,
        tags=tc.wandb_tags or None, id=run_id, resume="allow", mode=mode, dir=str(out),
        config={**cfg.to_dict(), "world_size": world_size, "gpu": gpu},
    )


def _fmt(m: dict) -> str:
    return " ".join(f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, float))


class Trainer:
    def __init__(self, cfg: JEVConfig, model: JEVModel, tokenizer, ctx: DistContext, dtype: torch.dtype,
                 train: list[JEVExample], val: list[JEVExample], calib: list[JEVExample] | None = None,
                 evals: dict[str, list[JEVExample]] | None = None, hub=None):
        self.cfg, self.tc = cfg, cfg.train
        self.model, self.tokenizer, self.ctx = model, tokenizer, ctx
        self.train_ex, self.val_ex, self.calib_ex, self.evals = train, val, calib, evals or {}
        self.out = Path(self.tc.output_dir)
        self.collator = JEVCollator(tokenizer, max_length=cfg.data.max_length)
        self.device = ctx.device
        self.use_autocast = ctx.device.type == "cuda" and dtype == torch.bfloat16
        self.criterion = JEVLoss(cfg.train.loss)
        self.history: list[dict] = []
        self.wandb = None
        self.hub = hub  # HubUploader on rank 0 when pushing is enabled, else None

    # -- helpers ------------------------------------------------------------------------

    def autocast(self):
        if self.use_autocast:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def to_device(self, batch):
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def log(self, record: dict):
        if not self.ctx.is_main:
            return
        with open(self.out / "log.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        if self.wandb:
            self.wandb.log({k: v for k, v in record.items() if isinstance(v, (int, float))}, step=record.get("step"))

    # -- evaluation (sharded across ranks) ----------------------------------------------

    @torch.no_grad()
    def predict_logits(self, examples: list[JEVExample], desc: str = "eval") -> dict[str, torch.Tensor]:
        """Raw logits for all examples, in input order, padded to MAX_K options. Each rank scores
        its own shard (length-sorted for speed); results are gathered on every rank."""
        model = self.model
        was_training = model.training
        model.eval()
        dataset = JEVDataset(examples)
        mine = list(range(self.ctx.rank, len(examples), self.ctx.world_size))
        mine.sort(key=lambda i: _approx_len(examples[i]))
        bs = self.tc.eval_batch_size
        parts = {"idx": [], "logits": [], "target": [], "label": [], "option_mask": [], "kind": []}
        t0 = time.time()
        for s in range(0, len(mine), bs):
            ids = mine[s : s + bs]
            batch = self.to_device(self.collator([dataset[i] for i in ids]))
            with self.autocast():
                logits = model(**batch)
            pad = MAX_K - logits.size(1)
            parts["idx"].append(torch.tensor(ids))
            parts["logits"].append(torch.nn.functional.pad(logits.float(), (0, pad), value=float("-inf")).cpu())
            parts["target"].append(torch.nn.functional.pad(batch["target"], (0, pad)).cpu())
            parts["option_mask"].append(torch.nn.functional.pad(batch["option_mask"], (0, pad)).cpu())
            parts["label"].append(batch["label"].cpu())
            parts["kind"].append(batch["kind"].cpu())
        local = {k: torch.cat(v) if v else torch.empty(0) for k, v in parts.items()}
        gathered = self.ctx.all_gather_object(local)
        merged = {k: torch.cat([g[k] for g in gathered if g[k].numel()]) for k in local}
        order = merged.pop("idx").argsort()
        self.ctx.print(f"  [{desc}] {len(examples)} rows in {time.time() - t0:.0f}s")
        if was_training:
            model.train()
        return {k: v[order] for k, v in merged.items()}

    def evaluate(self, examples, temps: torch.Tensor | None = None, desc="eval", groups=("kind",)):
        res = self.predict_logits(examples, desc)
        if temps is not None:
            res["logits"] = apply_temperatures(res["logits"], res["kind"], temps)
        m = compute_metrics(**res)
        for g in groups:
            values = [KINDS[k] for k in res["kind"].tolist()] if g == "kind" else [ex.get(g) for ex in examples]
            m[f"by_{g}"] = metrics_by(res, values)
        return m, res

    # -- checkpointing ------------------------------------------------------------------

    def save_state(self, name: str, optimizer, scheduler, state: dict):
        if self.ctx.is_main:
            self.model.save(self.out / name)
            torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "history": self.history, "python_rng": random.getstate(), **state},
                       self.out / name / "trainer_state.pt")
        self.ctx.barrier()

    # -- Hub ------------------------------------------------------------------------------

    def push_epoch(self, epoch: int, step: int):
        """Upload the model as it is at the end of an epoch (uncalibrated), tagged epoch-N."""
        if self.hub is not None and self.ctx.is_main:
            snap = self.out / "epoch_snapshot"
            self.model.save(snap)
            val = self.history[-1] if self.history else {}
            metrics = f" Latest validation: top-1 {val['acc']:.3f}, KL {val['kl']:.4f}." if "acc" in val else ""
            status = (f"Intermediate checkpoint after epoch {epoch + 1} (step {step}), not calibrated yet."
                      f"{metrics} The final calibrated model is tagged `final`.")
            self.hub.push(snap, build_model_card(snap, self.hub.repo, status),
                          f"epoch {epoch + 1} (step {step})", tag=f"epoch-{epoch + 1}")
            shutil.rmtree(snap, ignore_errors=True)
        self.ctx.barrier()

    # -- training -------------------------------------------------------------------------

    def fit(self, resume: bool = False):
        tc, ctx = self.tc, self.ctx
        per_step = tc.micro_batch_size * ctx.world_size
        if tc.global_batch_size % per_step:
            raise ValueError(f"global_batch_size {tc.global_batch_size} must be divisible by "
                             f"micro_batch_size x world_size = {per_step}")
        accum = tc.global_batch_size // per_step
        if ctx.is_main:
            self.out.mkdir(parents=True, exist_ok=True)
            self.cfg.save(self.out / "config.yaml")
            self.wandb = init_wandb(self.cfg, self.out, ctx.world_size)

        sampler = LengthGroupedBatchSampler(
            [_approx_len(ex) for ex in self.train_ex], tc.micro_batch_size, seed=tc.seed,
            group_by_length=tc.group_by_length, rank=ctx.rank, world_size=ctx.world_size, multiple_of=accum,
        )
        loader = DataLoader(
            JEVDataset(self.train_ex, shuffle_prob=self.cfg.data.shuffle_prob,
                       loss_weights=self.cfg.data.loss_weights), batch_sampler=sampler,
            collate_fn=self.collator, num_workers=tc.num_workers, pin_memory=self.device.type == "cuda",
        )
        steps_per_epoch = len(sampler) // accum
        total_steps = steps_per_epoch * tc.epochs
        if tc.max_steps:
            total_steps = min(total_steps, tc.max_steps)
        if total_steps == 0:
            raise ValueError("no optimizer steps: dataset smaller than one global batch")

        head_params = list(self.model.head.parameters())
        head_ids = {id(p) for p in head_params}
        lora_params = [p for p in self.model.trainable_parameters() if id(p) not in head_ids]
        optimizer = torch.optim.AdamW(
            [{"params": lora_params, "lr": tc.lr}, {"params": head_params, "lr": tc.head_lr}],
            betas=tuple(tc.betas), weight_decay=tc.weight_decay, fused=self.device.type == "cuda",
        )
        warmup = int(total_steps * tc.warmup_ratio)

        def lr_lambda(s):  # linear warmup, cosine decay to min_lr_ratio
            if s < warmup:
                return (s + 1) / max(1, warmup)
            progress = min(1.0, (s - warmup) / max(1, total_steps - warmup))
            return tc.min_lr_ratio + (1 - tc.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        step, start_epoch, start_micro, best_kl = 0, 0, 0, float("inf")
        last = self.out / "last"
        if resume and (last / "trainer_state.pt").exists():
            state = torch.load(last / "trainer_state.pt", map_location="cpu", weights_only=False)
            self.model.load_adapter_weights(last)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            step, start_epoch, start_micro = state["step"], state["epoch"], state["micro_in_epoch"]
            best_kl, self.history = state["best_kl"], state["history"]
            random.setstate(state["python_rng"])
            ctx.print(f"resumed from {last} at step {step} (epoch {start_epoch}, micro-batch {start_micro})")
        elif resume:
            ctx.print(f"--resume given but {last} has no trainer_state.pt; starting fresh")

        n_train = sum(p.numel() for p in self.model.trainable_parameters())
        ctx.print(f"loss = {self.criterion.describe()}")
        ctx.print(f"world_size={ctx.world_size} micro_batch={tc.micro_batch_size} grad_accum={accum} "
                  f"global_batch={tc.global_batch_size} steps/epoch={steps_per_epoch} total_steps={total_steps} "
                  f"trainable={n_train / 1e6:.1f}M")

        ddp = DDP(self.model, device_ids=[ctx.local_rank]) if ctx.enabled else self.model

        def validate():
            nonlocal best_kl
            m, _ = self.evaluate(self.val_ex, desc="val", groups=())
            self.history.append({"step": step, **m})
            self.log({"step": step, **{f"val/{k}": v for k, v in m.items()}})
            ctx.print(f"[step {step}] val {_fmt(m)}")
            if m["kl"] < best_kl:
                best_kl = m["kl"]
                if ctx.is_main:
                    self.model.save(self.out / "best")
                ctx.barrier()
                ctx.print(f"  new best (val kl {best_kl:.4f}) -> {self.out / 'best'}")

        if step == 0:
            validate()  # step-0 baseline: head initialized from the verbalizer rows

        self.model.train()
        t0, tokens, losses, done = time.time(), 0, [], step >= total_steps
        term_sums: dict[str, torch.Tensor] = {}
        for epoch in range(start_epoch, tc.epochs):
            if done:
                break
            micro_start = start_micro if epoch == start_epoch else 0
            sampler.set_epoch(epoch, start=micro_start)
            micro = micro_start
            for batch in loader:
                batch = self.to_device(batch)
                micro += 1
                sync = micro % accum == 0
                no_sync = ddp.no_sync() if ctx.enabled and not sync else contextlib.nullcontext()
                with no_sync:  # backward must be inside no_sync, or DDP all-reduces every micro-step
                    with self.autocast():
                        logits = ddp(**batch)
                        loss, parts = self.criterion(logits, batch["target"], batch["option_mask"],
                                                     batch["kind"], batch["label"], batch["weight"])
                    (loss / accum).backward()
                losses.append(loss.detach())
                for name, v in parts.items():
                    term_sums[name] = term_sums.get(name, 0.0) + v
                tokens += int(batch["attention_mask"].sum())
                if not sync:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), tc.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % tc.log_every == 0:
                    dt = time.time() - t0
                    rec = {"step": step, "epoch": epoch, "loss": ctx.mean(torch.stack(losses).mean().item()),
                           "grad_norm": float(grad_norm), "lr": scheduler.get_last_lr()[0],
                           "tokens_per_s": ctx.sum(tokens) / dt,
                           **{f"loss/{n}": ctx.mean((v / len(losses)).item()) for n, v in term_sums.items()}}
                    self.log(rec)
                    eta = (total_steps - step) * dt / tc.log_every / 3600
                    ctx.print(f"step {step}/{total_steps} loss {rec['loss']:.4f} gnorm {rec['grad_norm']:.2f} "
                              f"lr {rec['lr']:.2e} {rec['tokens_per_s']:.0f} tok/s eta {eta:.1f}h")
                    t0, tokens, losses, term_sums = time.time(), 0, [], {}
                if step % tc.eval_every == 0:
                    validate()
                    t0 = time.time()
                if step % tc.save_every == 0 or step >= total_steps:
                    next_epoch = micro >= len(sampler)
                    self.save_state("last", optimizer, scheduler, {
                        "step": step, "epoch": epoch + int(next_epoch),
                        "micro_in_epoch": 0 if next_epoch else micro, "best_kl": best_kl,
                    })
                if step >= total_steps:
                    done = True
                    break
            if micro >= len(sampler):  # this epoch ran to its end
                self.push_epoch(epoch, step)

        if not self.history or self.history[-1]["step"] != step:
            validate()

    # -- calibration + final report -----------------------------------------------------------

    def calibrate_and_report(self) -> dict:
        ctx = self.ctx
        best = self.out / "best"
        self.model.load_adapter_weights(best)
        report: dict = {"val_history": self.history}

        temps = torch.ones(len(KINDS))
        if self.calib_ex:
            res = self.predict_logits(self.calib_ex, desc="calib")
            temps = fit_temperatures(res["logits"], res["target"], res["option_mask"], res["kind"])
        self.model.temperature.copy_(temps)
        report["temperature"] = dict(zip(KINDS, temps.tolist()))
        ctx.print("temperatures " + " ".join(f"{k}={v:.3f}" for k, v in report["temperature"].items()))
        if ctx.is_main:
            self.model.save(best)
        ctx.barrier()

        for name, examples in self.evals.items():
            raw, res = self.evaluate(examples, desc=name, groups=("kind",))
            cal_logits = apply_temperatures(res["logits"], res["kind"], temps)
            cal = compute_metrics(**{**res, "logits": cal_logits})
            cal["by_kind"] = metrics_by({**res, "logits": cal_logits}, [KINDS[k] for k in res["kind"].tolist()])
            families = [ex.get("family") for ex in examples]
            if any(f is not None for f in families):
                cal["by_family"] = metrics_by({**res, "logits": cal_logits}, families)
            report[name] = {"uncalibrated": raw, "calibrated": cal}
            ctx.print(f"{name} uncalibrated {_fmt(raw)}")
            ctx.print(f"{name} calibrated   {_fmt(cal)}")
            for k, m in cal["by_kind"].items():
                ctx.print(f"  {name}/{k:6s} {_fmt(m)}")

        if ctx.is_main:
            (self.out / "report.json").write_text(json.dumps(report, indent=2))
            ctx.print(f"report -> {self.out / 'report.json'}")
            if self.hub is not None:
                pushed = self.hub.push(best, build_model_card(best, self.hub.repo),
                                       "final: calibrated best checkpoint", tag="final")
                report["hub_repo"] = self.hub.repo if pushed else None
                (self.out / "report.json").write_text(json.dumps(report, indent=2))
            if self.wandb:
                if report.get("hub_repo"):
                    self.wandb.summary["hub_repo"] = f"https://huggingface.co/{report['hub_repo']}"
                self.wandb.summary.update(_flatten({k: v for k, v in report.items() if k != "val_history"}))
                self.wandb.save(str(self.out / "report.json"), base_path=str(self.out), policy="now")
                self.wandb.finish()
        return report

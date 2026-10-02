# 🏃 Training on RunPod: step-by-step

This guide goes from an empty RunPod account to a trained, calibrated model on the Hugging Face Hub.
It uses the two scripts in [`scripts/`](../scripts): `runpod_setup.sh` (once per pod) and `runpod_train.sh`
(each run).

> [!IMPORTANT]
> **Tested environment.** This guide has only been tested on the setup below (JEV-9B). Other GPUs, GPU counts or
> images should work, but behave differently in the ways listed in [Other GPUs](#other-gpus) and have not been
> verified.
>
> | | |
> |---|---|
> | RunPod image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
> | GPUs | 2× NVIDIA H100 SXM (80 GB) |
> | Software | Ubuntu 24.04 · Python 3.12 · torch 2.8.0+cu128 · Triton 3.4.0 · transformers 5.17.0 · flash-linear-attention 0.5.2 |
> | Status (2026-10-02) | JEV-9B trained end to end (4,750 steps, about 3.5 h) and evaluated; see the README results. JEV-27B not run yet |

**Contents**

1. [Before you start](#1-before-you-start)
2. [Create the pod](#2-create-the-pod)
3. [Get the code](#3-get-the-code)
4. [Add your tokens](#4-add-your-tokens)
5. [Run the setup](#5-run-the-setup)
6. [Start training](#6-start-training)
7. [Watch the run](#7-watch-the-run)
8. [Stop, resume, restart](#8-stop-resume-restart)
9. [When training finishes](#9-when-training-finishes)
    - [Same pod or a fresh pod?](#same-pod-or-a-fresh-pod)
10. [Useful commands](#10-useful-commands)
11. [Troubleshooting](#11-troubleshooting)
12. [Cheat sheet](#12-cheat-sheet)

---

## 1. Before you start

| You need | Why | Where |
|---|---|---|
| A RunPod account with credit | GPU time | [runpod.io](https://www.runpod.io) |
| A Hugging Face **write** token | upload checkpoints (optional, but recommended) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| A W&B API key | live training charts (optional) | [wandb.ai/authorize](https://wandb.ai/authorize) |

Checkpoints are uploaded to **the account that owns the HF token**, as `<account>/<run name>` (for example
`your-name/jev-9b`). The repo is created private.

## 2. Create the pod

**Template / image:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8, CUDA 12.8). The setup
script is tested against this image and keeps its torch build untouched.

**GPU:** pick by config. Times are for about one epoch (the configs' `max_steps`).

| Config | Recommended GPU | Time |
|---|---|---|
| `jev-2b.yaml` | 1× L40S, A100 or H100 | ~1.5 h on H100 (est.) |
| `jev-9b.yaml` | **1× H100 80GB** or 1× B200 | ~6 h on H100 (est.) · **~3 h on B200** (measured by autotrust) |
| `jev-27b.yaml` | 1× B200, or 4–8× H100 80GB | **~9.2 h on 1× B200** (measured) · ~3 h on 8× H100 (est.) |
| `jev-27b-qlora.yaml` | 1× L40S or A6000 (48 GB) | slower than bf16 |

A pod with 2+ GPUs also lets the setup's smoke test check multi-GPU training before your real run.

### Other GPUs

Only 2× H100 SXM has been tried (see the box at the top). On other hardware, expect these differences:

| | What changes |
|---|---|
| **Hopper (H100, H200)** | with the image's Triton 3.4, flash-linear-attention needs its TileLang backend for correct gradients ([fla #640](https://github.com/fla-org/flash-linear-attention/issues/640)). The setup installs `tilelang` and `nvcc` and checks that the backend is active |
| **Ampere / Ada (A100, L40S, A6000) and Blackwell (B200)** | the Triton kernels are used directly; TileLang is not installed. Not tested here |
| **Speed and memory** | the times and memory figures in this guide are estimates scaled from autotrust's B200 measurements, not measured on these GPUs |
| **Results** | not bit-identical across GPU types or GPU counts: kernels differ, bf16 rounding differs, and with more GPUs each rank sees a different slice of every batch. Metrics should agree closely, not exactly |
| **Other images** | a different torch / CUDA / Triton combination can change which kernels run; the setup re-checks torch and the kernels, but only this image has been tried |

**Storage:** attach a **network volume mounted at `/workspace`**. Everything that must survive a restart
(code, Hugging Face cache, checkpoints) lives there.

| Config | Network volume | Why |
|---|---|---|
| 2B / 9B | **50 GB** | 9B weights 19.3 GB + smoke-test model 1.8 GB + dataset 0.5 GB + checkpoints |
| 27B | **100 GB** | 27B weights 55.6 GB + the rest |

The container disk can stay at the default (20 GB is enough).

Deploy the pod, then open a terminal: **Connect → Web Terminal**, or SSH.

## 3. Get the code

```bash
cd /workspace
git clone https://github.com/smha1012/jev-torch.git
cd jev-torch
```

<details>
<summary>The repository is private?</summary>
<br>

Use a GitHub personal access token with read access, then remove it from the saved remote:

```bash
git clone https://<github-user>:<github-token>@github.com/smha1012/jev-torch.git
cd jev-torch && git remote set-url origin https://github.com/smha1012/jev-torch.git
```

</details>

Already cloned on this volume? Update instead: `cd /workspace/jev-torch && git pull`.

## 4. Add your tokens

```bash
echo "HF_TOKEN=hf_your_token" > .env.local
echo "WANDB_API_KEY=your_key" >> .env.local      # optional
```

| Key | |
|---|---|
| `HF_TOKEN` | write token: enables Hub uploads (leave empty to keep checkpoints local) |
| `WANDB_API_KEY` | optional: without it, W&B logs offline |
| `JEV_OVERRIDES` | optional personal settings, see below |
| `TZ` | optional: timestamps in your time zone, e.g. `TZ=KST-9` for Korea (the pod clock is UTC) |

Tokens set as **RunPod pod environment variables** are picked up too, even in terminals that do not
inherit them (the scripts read them from the pod's environment). Without any token, the train script stops
before starting, because nothing would be uploaded; add `--set train.hf_push=null` to train without uploads.

`.env.local` is in `.gitignore`, so it never ends up in a commit. Values exported in the shell (or set in the
RunPod dashboard) take precedence over the file.

<details>
<summary>Personal settings with <code>JEV_OVERRIDES</code></summary>
<br>

Anything you would pass as `--set` can live here, so the shared configs stay untouched:

```bash
# read your own snapshot of the dataset, push to a differently named repo, turn W&B off
echo 'JEV_OVERRIDES="data.hf_dataset=your-name/jev-distill-corpus-v3 train.hf_push=your-name/jev-9b-v2 train.wandb_project=null"' >> .env.local
```

They are applied before `--set` and printed at the start of the log.

</details>

## 5. Run the setup

Once per pod. Pass the config you plan to train so its model is downloaded in advance:

```bash
bash scripts/runpod_setup.sh configs/jev-9b.yaml
```

It takes about 5–10 minutes, longer for 27B because of the 55 GB download. Each step prints a timestamped
header:

| Step | What to look for |
|---|---|
| **Environment** | your GPUs are listed |
| **Pin torch / Python dependencies** | `transformers 5.17.x · peft 0.21.x · flash-linear-attention 0.5.x` |
| **Re-check torch integrity** | `available True · bf16 True` (it stops if torch was replaced), then `ok fla`. A `missing fla` line means training will be very slow |
| **Tokens** | `HF_TOKEN OK · account <you> · role write` (a `read` role means uploads will fail) |
| **Unit tests** | `N passed` |
| **GPU smoke test** | `smoke test passed`: 4 real training steps of Qwen3.5-0.8B, calibration and report on this GPU |

If the smoke test fails, the script stops and shows the end of `/tmp/jev-smoke.log`
(`less /tmp/jev-smoke.log` for all of it). Nothing expensive has
started yet, so this is the cheapest place to find a problem. To skip it on a pod you already validated:
`SKIP_SMOKE=1 bash scripts/runpod_setup.sh`.

Open a new terminal afterwards (or run `source ~/.bashrc`) so the cache settings apply to your shell.

## 6. Start training

```bash
bash scripts/runpod_train.sh configs/jev-9b.yaml
```

```text
started pid 4321 -> runs/jev-9b/train.log
  follow:  tail -f runs/jev-9b/train.log
  stop:    pkill -f 'jev\.(launch|train)'
```

The command then **shows the log live**. Training itself runs in the background in its own session, so
`Ctrl+C`, closing the terminal or losing the connection only stops the display, never the training. Watch
again any time with `tail -f runs/jev-9b/train.log`.

**Common variations:**

```bash
# use 4 GPUs (the global batch stays at 128 rows; accumulation adjusts)
bash scripts/runpod_train.sh configs/jev-9b.yaml --set train.num_gpus=4

# a separate experiment: a new output_dir (and therefore a new Hub repo name)
bash scripts/runpod_train.sh configs/jev-9b.yaml --set train.output_dir=runs/jev-9b-brier \
    train.loss="{kl: 1.0, brier: 0.25}"

# no Hub uploads for this run
bash scripts/runpod_train.sh configs/jev-9b.yaml --set train.hf_push=null
```

> [!IMPORTANT]
> The train script **always resumes** from `runs/<name>/last` when it exists. For a new experiment, use a new
> `train.output_dir`; reusing one continues the old run.

## 7. Watch the run

```bash
tail -f runs/jev-9b/train.log         # Ctrl+C stops watching, not training
watch -n 5 nvidia-smi                 # GPU memory and utilization
pgrep -af 'jev\.(launch|train)'       # is training still running? (no output = stopped)
```

The first lines show what the run is using. Check them once:

```text
model=Qwen/Qwen3.5-9B device=cuda:0 dtype=torch.bfloat16 world_size=1
Hub: pushing to https://huggingface.co/your-name/jev-9b as your-name (private)
data=jev_distill (https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3 @ fc99c63…)
  rows:   train=655806 val=4000 calib=13766 test_set_30k=29955 ood=13058
loss = 1·kl + 0.5·rps[score]
```

Then, during training:

```text
[step 0] val acc=0.5123 kl=0.4012 ...           <- baseline before training (zero-shot head)
step 10/4750 loss 0.3121 gnorm 1.05 lr 3.5e-05 8120 tok/s eta 2.9h
...
[step 500] val acc=0.8410 kl=0.0612 ...
  new best (val kl 0.0612) -> runs/jev-9b/best
```

| Field | Meaning | What is normal |
|---|---|---|
| `loss` | training loss | falls fast in the first few hundred steps, then slowly |
| `gnorm` | gradient norm (clipped at 1.0) | single digits or below; growing values mean trouble |
| `tok/s` | throughput | decides your cost; compare with the table in section 2 |
| `eta` | estimated hours left | stabilizes after a few dozen steps |
| `val kl` | distance to the teacher on the validation subset, every 500 steps | lower is better; drives `best/` |

**After 5–10 minutes**, decide whether the GPU choice is right: if `eta` is much longer than expected, stop
and pick a larger GPU or more GPUs. If `nvidia-smi` shows lots of free memory, a larger
`train.micro_batch_size` can raise `tok/s`.

With `WANDB_API_KEY` set, the same numbers appear live at [wandb.ai](https://wandb.ai) in project `jev-torch`,
including each loss term (`loss/kl`, `loss/rps`).

## 8. Stop, resume, restart

| Situation | What to do |
|---|---|
| Stop training | `pkill -f 'jev\.(launch\|train)'` |
| Continue after stopping or a crash (same pod) | run **the same** `runpod_train.sh` command again |
| Pod was stopped and started again, or preempted | rerun the setup first, then the same train command (see below) |

> [!WARNING]
> Stopping a pod wipes its **container disk**, which includes the installed Python packages. `/workspace`
> (code, checkpoints, Hugging Face cache) survives. After a restart, reinstall before resuming. With the
> smoke test skipped and the model already cached, this takes a few minutes:
>
> ```bash
> cd /workspace/jev-torch
> SKIP_SMOKE=1 bash scripts/runpod_setup.sh
> bash scripts/runpod_train.sh configs/jev-9b.yaml      # same command and flags as before
> ```

Progress is saved to `runs/<name>/last` every 250 steps (`train.save_every`), so at most 250 steps are
repeated. The log confirms the resume point:

```text
resumed from runs/jev-9b/last at step 2750 (epoch 0, micro-batch 22000)
```

The same W&B run continues, and the Hub upload continues to the same repo.

## 9. When training finishes

The last lines of the log:

```text
temperatures noul=1.004 choice=0.987 score=1.010
test_set_30k calibrated   acc=0.8xxx kl=0.0xxx ...
  test_set_30k/choice ...
Hub: pushed 'final: calibrated best checkpoint' -> https://huggingface.co/your-name/jev-9b (tag final)
report -> runs/jev-9b/report.json
```

**How close is it to TypeSafe Jev?** The end of the log prints a comparison on the 25,376 Jev-labelled
rows of `test_set_30k`, the same rows autotrust reports on:

```text
vs teacher on test_set_30k (25,376 teacher-labelled rows):
  metric                                      this model    JEV-9B   JEV-27B
  mean KL to teacher (lower is better)            0.0xxx    0.0190    0.0170
  choice top-1 agreement, teacher rows             xx.x%     90.2%         -
  choice top-1 agreement, all choice rows          xx.x%     89.8%     90.3%
  ECE                                             0.0xxx    0.0007    0.0009
```

The same block is stored under `vs_teacher` in `report.json`. To recompute it later, for any checkpoint or
Hub model:

```bash
python3 -m jev.evaluate --model runs/jev-9b/best
```

**Results:** `runs/jev-9b/report.json` has the test and OOD metrics, before and after calibration, by kind
and by domain family. A readable summary:

```bash
python3 - <<'PY'
import json
r = json.load(open("runs/jev-9b/report.json"))
print("temperatures", r["temperature"])
for split in ("test_set_30k", "ood"):
    if split in r:
        cal = r[split]["calibrated"]
        print(f"{split:13s} all     acc={cal['acc']:.3f} kl={cal['kl']:.4f} ece={cal['ece']:.4f} n={cal['n']}")
        for kind, m in cal["by_kind"].items():
            print(f"{'':13s} {kind:7s} acc={m['acc']:.3f} kl={m['kl']:.4f} ece={m['ece']:.4f} n={m['n']}")
PY
```
 For comparison, autotrust's JEV-9B reports about 90% top-1 agreement with the teacher
on choice questions and a mean KL of about 0.019.

**Try the model:**

```bash
python3 examples/inference.py --model runs/jev-9b/best          # local checkpoint
python3 examples/inference.py --model your-name/jev-9b          # from the Hub (needs HF_TOKEN while private)
```

**W&B without a key?** The run was logged offline. Upload it once a key is set:

```bash
wandb login                                   # or put WANDB_API_KEY in .env.local
wandb sync runs/jev-9b/wandb/offline-run-*
```

**Hub upload missing?** If there was no token during training, or the final push failed (the log says
`Hub: ⚠️ push failed`), upload the calibrated checkpoint by hand (it reads `HF_TOKEN` from `.env.local`):

```bash
python3 -m jev.push_to_hub --ckpt runs/jev-9b/best --repo your-name/jev-9b --tag final
```

**Freeze the versions** that just worked, so the next pod installs exactly the same set:

```bash
uv pip freeze --system > scripts/requirements.lock
git add scripts/requirements.lock && git commit -m "Lock versions from a successful run" && git push
```

**Stop the pod** in the RunPod console to stop paying for the GPU. The network volume keeps the checkpoints
(it is billed separately, at a much lower rate). Delete it when you no longer need them.

**Publishing:** the Hub repo starts private. Make it public in the repo's *Settings* on huggingface.co when
you are ready.

### Same pod or a fresh pod?

A run's results live in `runs/<name>/` on the volume the pod had when it trained. Whether you can use
them depends on that volume, not on the code:

| | You have `runs/<name>/` | You don't (a new pod without that volume) |
|---|---|---|
| When | the training pod, or a new pod with the **same network volume** on `/workspace` | a new pod with a new or no network volume |
| Check | `ls runs/jev-9b/best` lists `adapter/ head.pt jev_config.json` | `ls runs/` fails or is empty |
| Model argument | `runs/jev-9b/best` | the Hub repo, e.g. `your-name/jev-9b` |
| Base model download | cached in `/workspace/hf_cache` | downloaded again (19 GB for 9B) |
| First step | `cd /workspace/jev-torch` | clone, then `SKIP_SMOKE=1 bash scripts/runpod_setup.sh` (installs packages) |

To keep results across pods, create the pod with the **same network volume** mounted on `/workspace`; the
Hub copy (tag `final`) is the fallback when it is gone.

**With `runs/`** (same pod or same volume):

```bash
cd /workspace/jev-torch
python3 -m jev.evaluate --model runs/jev-9b/best
python3 -m jev.bench --model runs/jev-9b/best --out runs/jev-9b/bench.json
python3 examples/inference.py --model runs/jev-9b/best
```

**Without `runs/`** (fresh pod): use the Hub repo directly. A private repo needs the token in this shell
first, which `source scripts/_env.sh` loads from the pod's variables or `.env.local`:

```bash
cd /workspace/jev-torch
source scripts/_env.sh
python3 -m jev.evaluate --model your-name/jev-9b
python3 -m jev.bench --model your-name/jev-9b --out bench.json
python3 examples/inference.py --model your-name/jev-9b
```

Refreshing the **model card** is the one task that needs a local run directory (it reads
`runs/<name>/report.json` and `config.yaml`). On a fresh pod, rebuild one from the Hub copy first:

```bash
source scripts/_env.sh
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('your-name/jev-9b', revision='final', local_dir='runs/jev-9b/best', allow_patterns=['jev_config.json','head.pt','adapter/*'])"
python3 -c "from jev.config import load_config; load_config('configs/jev-9b.yaml').save('runs/jev-9b/config.yaml')"
python3 -m jev.evaluate --model runs/jev-9b/best --split test_set_30k ood --update_report runs/jev-9b/report.json
python3 -m jev.push_to_hub --ckpt runs/jev-9b/best --repo your-name/jev-9b --card_only
```

(Add the same `--set` overrides to `load_config(...)` that the original run used, e.g. `['train.num_gpus=2']`,
so the card describes it correctly.)

## 10. Useful commands

| Task | Command |
|---|---|
| Update the code | `cd /workspace/jev-torch && git pull` |
| Rerun setup without the smoke test | `SKIP_SMOKE=1 bash scripts/runpod_setup.sh` |
| Is training running? | `pgrep -af 'jev\.(launch\|train)'` |
| Follow the log | `tail -f runs/jev-9b/train.log` |
| Last validation results | `grep "val acc" runs/jev-9b/train.log \| tail -5` |
| GPU usage | `watch -n 5 nvidia-smi` |
| Disk space | `df -h /workspace` and `du -sh /workspace/hf_cache runs/*` |
| Remove an old run | `rm -rf runs/<old-run>` (local only; its Hub repo is untouched) |
| Compare a model with TypeSafe Jev (teacher agreement) | `python3 -m jev.evaluate --model runs/jev-9b/best` |
| Accuracy on ground truth next to TypeSafe Jev | `python3 -m jev.bench --model runs/jev-9b/best --out runs/jev-9b/bench.json` |
| Chart of those results | `python3 -m jev.plot_bench runs/jev-9b/bench.json` |
| Score your own questions | `python3 -m jev.predict --ckpt runs/jev-9b/best --input my.jsonl` |
| Evaluate on labeled data | `python3 -m jev.predict --ckpt runs/jev-9b/best --input labeled.jsonl --metrics` |
| Upload a checkpoint by hand | `python3 -m jev.push_to_hub --ckpt runs/jev-9b/best --repo your-name/jev-9b` |
| Sync an offline W&B run | `wandb sync runs/jev-9b/wandb/offline-run-*` |
| Smoke-test log | `less /tmp/jev-smoke.log` |

The input format for `jev.predict` is described in the [README](../README.md#-data-format).

## 11. Troubleshooting

| Message or symptom | Cause | Fix |
|---|---|---|
| `!! torch changed 2.8.0 -> ...` | an install replaced the image's torch | use the image from section 2; do not install torch yourself (pip or uv) |
| `missing fla` or `falling back to its reference PyTorch implementation` for `chunk_gated_delta_rule` | fast Qwen3.5 kernels not installed | rerun `bash scripts/runpod_setup.sh` (it installs them with uv) |
| `the HF token belongs to 'X', which cannot write to "Y"` | `hf_push` names someone else's namespace | use `hf_push: auto`, or a token of that account |
| `cannot create or access ... It needs WRITE access` | read-only token | create a write token, or `--set train.hf_push=null` |
| `CUDA out of memory` | micro-batch too large for this GPU | halve `train.micro_batch_size` (16 → 8 → 4). It must divide `128 / num_gpus` |
| `global_batch_size 128 must be divisible by ...` | micro-batch × GPUs does not divide 128 | choose `micro_batch_size` so that `micro × num_gpus` divides 128 |
| `config asks for train.num_gpus=N but only M CUDA device(s) are visible` | more GPUs requested than the pod has | lower `train.num_gpus` |
| `warning: runs/... is not on /workspace` | repo cloned outside the network volume | re-clone under `/workspace`, or checkpoints vanish when the pod stops |
| `No space left on device` | volume too small | resize the volume (section 2) or delete old `runs/*` and `$HF_HOME` models |
| `Triton >= 3.4.0 and < 3.7.1 on Hopper GPUs produces incorrect results ... install tilelang` | H100/H200 with the image's Triton 3.4: fla needs its TileLang backend, which needs nvcc | `git pull` and rerun `bash scripts/runpod_setup.sh` (it installs tilelang and nvcc and checks the backend) |
| very low `tok/s` | slow kernels, or a GPU smaller than planned | check the smoke-test warnings; compare with section 2 |
| the run restarts from step 0 unexpectedly | `output_dir` changed, so `last/` was not found | use the same `train.output_dir` (and `--set` flags) as the original run |
| `'runs/jev-9b/best' is neither a checkpoint directory nor a Hub repo id` | a new pod without the volume that holds `runs/` | use the Hub repo (`--model your-name/jev-9b`) or attach the same network volume ([Same pod or a fresh pod?](#same-pod-or-a-fresh-pod)) |
| `ModuleNotFoundError: No module named 'jev'` (or `fla`) after a restart | the container disk was wiped | `SKIP_SMOKE=1 bash scripts/runpod_setup.sh` (section 8) |

Full logs: training in `runs/<name>/train.log`, smoke test in `/tmp/jev-smoke.log`.

## 12. Cheat sheet

Every block below can be copied and run as it is. The examples use 2 GPUs; change `train.num_gpus=2` to
match your pod.

**① Get the code** (once per network volume)

```bash
cd /workspace && git clone https://github.com/smha1012/jev-torch.git && cd jev-torch
```

**② Tokens.** Skip this if you set `HF_TOKEN` / `WANDB_API_KEY` as environment variables when creating the pod.

```bash
echo "HF_TOKEN=hf_your_token" > .env.local
echo "WANDB_API_KEY=your_key" >> .env.local
```

**③ Setup** (once per pod start; ends with `smoke test passed`)

```bash
bash scripts/runpod_setup.sh configs/jev-9b.yaml
```

**④ Train.** Starts in the background. Running the same line again later resumes the run.

```bash
bash scripts/runpod_train.sh configs/jev-9b.yaml --set train.num_gpus=2
```

**⑤ Watch the log again** (④ already shows it). `Ctrl+C` only stops watching; training keeps running.

```bash
tail -f runs/jev-9b/train.log
```

**⑥ Watch the GPUs.** `Ctrl+C` to leave.

```bash
watch -n 5 nvidia-smi
```

**⑦ Is training running?** No output means it has stopped.

```bash
pgrep -af 'jev\.(launch|train)'
```

**⑧ Try the trained model** (after training finishes)

```bash
python3 examples/inference.py --model runs/jev-9b/best
```

**⑨ Freeze the package versions** (after the first successful run)

```bash
uv pip freeze --system > scripts/requirements.lock
```

---

**After a pod restart** (installed packages are wiped, `/workspace` is kept): run setup again, then the same
train line as before.

```bash
cd /workspace/jev-torch
SKIP_SMOKE=1 bash scripts/runpod_setup.sh
bash scripts/runpod_train.sh configs/jev-9b.yaml --set train.num_gpus=2
```

> [!WARNING]
> **Stop training** only when you mean to. This kills the run (it can be resumed later with ④):
>
> ```bash
> pkill -f 'jev\.(launch|train)'
> ```

**Only if needed:**

| Situation | Command |
|---|---|
| The automatic Hub upload failed | `python3 -m jev.push_to_hub --ckpt runs/jev-9b/best --repo your-name/jev-9b --tag final` |
| W&B ran without a key (offline) | `wandb sync runs/jev-9b/wandb/offline-run-*` |
| Check free disk space | `df -h /workspace` |

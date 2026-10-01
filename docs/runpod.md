# 🏃 Training on RunPod: step-by-step

This guide goes from an empty RunPod account to a trained, calibrated model on the Hugging Face Hub.
It uses the two scripts in [`scripts/`](../scripts): `runpod_setup.sh` (once per pod) and `runpod_train.sh`
(each run).

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
10. [Troubleshooting](#10-troubleshooting)
11. [Cheat sheet](#11-cheat-sheet)

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
cp .env.sample .env.local
nano .env.local            # or vi
```

```bash
HF_TOKEN=hf_...            # write token: enables Hub uploads (leave empty to keep checkpoints local)
WANDB_API_KEY=...          # optional: without it, W&B logs offline
JEV_OVERRIDES=             # optional personal settings, see below
```

`.env.local` is in `.gitignore`, so it never ends up in a commit. Values exported in the shell (or set in the
RunPod dashboard) take precedence over the file.

<details>
<summary>Personal settings with <code>JEV_OVERRIDES</code></summary>
<br>

Anything you would pass as `--set` can live here, so the shared configs stay untouched:

```bash
# read your own snapshot of the dataset, push to a differently named repo, turn W&B off
JEV_OVERRIDES="data.hf_dataset=your-name/jev-distill-corpus-v3 train.hf_push=your-name/jev-9b-v2 train.wandb_project=null"
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

If the smoke test fails, the script stops and shows the end of `/tmp/jev-smoke.log`. Nothing expensive has
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

Training runs in the background with `nohup`, so closing the terminal or losing SSH does not stop it.

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
tail -f runs/jev-9b/train.log      # Ctrl+C stops watching, not training
watch -n 5 nvidia-smi              # GPU memory and utilization
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
| Stop training | `pkill -f 'jev\.(launch|train)'` |
| Continue after stopping, a crash, or a pod restart | run **the same** `runpod_train.sh` command again |
| Pod was stopped and started again | `cd /workspace/jev-torch`, then the same command. The volume kept everything |

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

**Results:** `runs/jev-9b/report.json` has the test and OOD metrics, before and after calibration, by kind
and by domain family. For comparison, autotrust's JEV-9B reports about 90% top-1 agreement with the teacher
on choice questions and a mean KL of about 0.019.

**Try the model:**

```bash
python3 examples/inference.py --model runs/jev-9b/best          # local checkpoint
python3 examples/inference.py --model your-name/jev-9b          # from the Hub (needs HF_TOKEN while private)
```

**Freeze the versions** that just worked, so the next pod installs exactly the same set:

```bash
pip freeze > scripts/requirements.lock
git add scripts/requirements.lock && git commit -m "Lock versions from a successful run" && git push
```

**Stop the pod** in the RunPod console to stop paying for the GPU. The network volume keeps the checkpoints
(it is billed separately, at a much lower rate). Delete it when you no longer need them.

**Publishing:** the Hub repo starts private. Make it public in the repo's *Settings* on huggingface.co when
you are ready.

## 10. Troubleshooting

| Message or symptom | Cause | Fix |
|---|---|---|
| `!! torch changed 2.8.0 -> ...` | pip replaced the image's torch | use the image from section 2; do not `pip install torch` yourself |
| `missing fla` or `falling back to its reference PyTorch implementation` for `chunk_gated_delta_rule` | fast Qwen3.5 kernels not installed | `pip install "flash-linear-attention>=0.5,<0.6"`, then rerun the setup |
| `the HF token belongs to 'X', which cannot write to "Y"` | `hf_push` names someone else's namespace | use `hf_push: auto`, or a token of that account |
| `cannot create or access ... It needs WRITE access` | read-only token | create a write token, or `--set train.hf_push=null` |
| `CUDA out of memory` | micro-batch too large for this GPU | halve `train.micro_batch_size` (16 → 8 → 4). It must divide `128 / num_gpus` |
| `global_batch_size 128 must be divisible by ...` | micro-batch × GPUs does not divide 128 | choose `micro_batch_size` so that `micro × num_gpus` divides 128 |
| `config asks for train.num_gpus=N but only M CUDA device(s) are visible` | more GPUs requested than the pod has | lower `train.num_gpus` |
| `warning: runs/... is not on /workspace` | repo cloned outside the network volume | re-clone under `/workspace`, or checkpoints vanish when the pod stops |
| `No space left on device` | volume too small | resize the volume (section 2) or delete old `runs/*` and `$HF_HOME` models |
| very low `tok/s` | slow kernels, or a GPU smaller than planned | check the smoke-test warnings; compare with section 2 |
| the run restarts from step 0 unexpectedly | `output_dir` changed, so `last/` was not found | use the same `train.output_dir` (and `--set` flags) as the original run |

Full logs: training in `runs/<name>/train.log`, smoke test in `/tmp/jev-smoke.log`.

## 11. Cheat sheet

```bash
# once per pod
cd /workspace && git clone https://github.com/smha1012/jev-torch.git && cd jev-torch
cp .env.sample .env.local && nano .env.local
bash scripts/runpod_setup.sh configs/jev-9b.yaml

# train (rerun the same line to resume)
bash scripts/runpod_train.sh configs/jev-9b.yaml [--set train.num_gpus=4 ...]

# watch / stop
tail -f runs/jev-9b/train.log
watch -n 5 nvidia-smi
pkill -f 'jev\.(launch|train)'

# afterwards
cat runs/jev-9b/report.json
python3 examples/inference.py --model runs/jev-9b/best
pip freeze > scripts/requirements.lock
```

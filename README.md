# jev-torch

Train **JEV-style decision models** in plain PyTorch: an LLM reads a *state*, a *question* and a list of
*options*, runs **one forward pass**, and returns a **calibrated probability for every option**, with no text generation.

```
[kind] choice
[state] The nightly feed is byte-identical to yesterday; additionally arrived 3 hours late. ...
[question] Root cause for this scenario.
[options]
A. producer_change
B. schema_drift
C. infrastructure
D. expected_variation
[decision]:  ──► Qwen3.5 (+LoRA) ─► last hidden ─► 24-slot fp32 head ─► softmax(logits / T_kind)
                                                                        A .48  B .01  C .48  D .03
```

The recipe follows the published [autotrust/JEV-9B](https://huggingface.co/autotrust/JEV-9B) training card
and trains on the open [JEV distillation corpus](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3)
(741k rows of TypeSafe Jev 1.13 output distributions, 65 domains).

| | |
|---|---|
| Backbone | any HF causal LM; configs for Qwen3.5 0.8B / 2B / 9B / 27B (text tower only) |
| Trainable | LoRA r16 on every projection (linear-attention, attention, MLP) + 24-slot fp32 head |
| Head init | slot *i* = LM-head row of its verbalizer token (`false true 0-5 A-P`), so step 0 = zero-shot |
| Loss | KL(teacher ‖ model) over active slots + 0.5 · RPS on ordinal `score` rows |
| Calibration | one temperature per kind (noul / choice / score), fit on the calibration split |
| Scale | single GPU, multi-GPU data parallel (torchrun), 4-bit QLoRA; resumable checkpoints |
| Data | same corpus and ~1 epoch for every size, as autotrust did for 9B (4.75k steps) and 27B (5k steps) |

## Data format

JSONL, one example per line (the corpus schema):

```json
{"kind": "choice",
 "state": "A distribution center shows a count variance of 2.3%; ...",
 "question": "Supplier response for this scenario.",
 "options": ["issue_warning", "renegotiate", "dual_source", "maintain"],
 "target": [0.6, 0.07, 0.32, 0.01],
 "domain": "inventory_supply"}
```

| field | | |
|---|---|---|
| `kind` | | `noul` (exactly 2 options: false-like, true-like) · `score` (up to 6 ordered levels, 0–5) · `choice` (2–16 options). Default `choice` |
| `state` | ✓ | the situation / record (may be `""`) |
| `question` | ✓ | what is being decided |
| `options` | ✓ | the menu; probabilities are a distribution over exactly these |
| `target` | one of | teacher probability per option (soft label, auto-normalized): **the main training signal** |
| `label` | these | hard label index; derived as `argmax(target)` when absent |
| `domain`, `meta` | | reporting / filtering; unknown keys are folded into `meta` |

See `examples/example.jsonl`.

## Quick start

```bash
pip install -e ".[dev]"            # add ",cuda" on NVIDIA machines
pytest -q                          # fast tests, no downloads

python -m jev.launch --config configs/debug.yaml          # laptop smoke test (Qwen3.5-0.8B, CPU/MPS)
python -m jev.predict --ckpt runs/debug/best --input examples/example.jsonl
```

## Configs and GPUs

Everything, including the GPU count, is set in the YAML config. Override any field from the command line:

```bash
python -m jev.launch --config configs/jev-9b.yaml --set train.num_gpus=4 train.max_steps=2000
```

`train.num_gpus > 1` makes the launcher start one process per GPU through `torchrun` (data parallel; each
GPU holds a full model copy). `global_batch_size` (128 rows/step) stays fixed; gradient accumulation is
derived as `global_batch_size / (micro_batch_size × num_gpus)`.

| config | backbone (trainable) | GPU memory | measured / estimated time |
|---|---|---|---|
| `debug.yaml` | Qwen3.5-0.8B | laptop | minutes (256 rows) |
| `jev-2b.yaml` | 1.9B (15.6M) | ~10 GB (est.) | est. ~1.5 h on H100, ~4 h on L40S / A100 |
| `jev-9b.yaml` | 7.9B (40.1M) | ~25 GB (est.) | **measured ~3 h on 1× B200**; est. ~6 h on 1× H100 |
| `jev-27b.yaml` | 25.6B (108.8M) | ~60 GB at micro-batch 8 (est.); autotrust peak 79 GB | **measured ~9.2 h on 1× B200**; est. ~20 h on 1× H100, ~3 h on 8× H100 |
| `jev-27b-qlora.yaml` | 25.6B NF4 (108.8M) | ~30 GB (est.) | not measured; slower than bf16 |

"Measured" numbers are from the autotrust/JEV-9B and JEV-27B model cards (same corpus, same recipe,
~1 epoch). Everything else is an estimate: memory from parameter and activation counts (gradient
checkpointing, no LM-head logits), time from the measured B200 throughput scaled by peak bf16 FLOPs at the
same utilization (~17%). Treat estimates as ±2x. The trainer prints `tok/s` and an ETA every `log_every`
steps, so run a few minutes and read them before committing to a long run.

Placeholder labels: 148k `yuri_v1` rows carry placeholder targets and autotrust down-weights them.
The configs set `data.loss_weights: {source: {yuri_v1: 0.1}}`. The exact weight is not published.

## RunPod

Base image: **`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`** (Runpod PyTorch 2.8.0). Put the repo on
the network volume (`/workspace`) so checkpoints and the HF cache survive pod restarts.

```bash
cd /workspace && git clone <repo-url> jev-torch && cd jev-torch
cp .env.sample .env.local && vi .env.local          # HF_TOKEN, WANDB_API_KEY (both optional)
bash scripts/runpod_setup.sh configs/jev-9b.yaml     # install (keeps the image's torch), checks, pre-download
bash scripts/runpod_train.sh configs/jev-9b.yaml     # background run, log in runs/jev-9b/train.log
```

**Weights & Biases:** on by default (`train.wandb_project: jev-torch`; also `wandb_entity`, `wandb_run_name`,
`wandb_tags`). Logged: train loss, grad norm, lr, tokens/s, val metrics every `eval_every`, and the final
test/ood metrics (by kind and family) plus per-kind temperatures in the run summary. The run id is kept in
`runs/<name>/wandb_id`, so a resumed training continues the same W&B run. Without `WANDB_API_KEY` it logs
offline (`wandb sync runs/<name>/wandb` later). Set `train.wandb_project: null` to disable.

`runpod_train.sh` always passes `--resume`: if the pod is preempted, run the same command again and training
continues from `runs/<name>/last` (saved every `save_every` steps). Use a new `train.output_dir` for a new
experiment.

## Outputs

```
runs/jev-9b/
  config.yaml     resolved config
  log.jsonl       loss, grad norm, lr, tokens/s, val metrics
  last/           resumable state (LoRA + head + optimizer + scheduler + sampler position)
  best/           best val-KL checkpoint, with per-kind temperatures after calibration
  report.json     val history + test_set_30k / ood metrics, uncalibrated vs calibrated, by kind and family
```

Metrics: `acc` = top-1 agrees with the teacher's argmax · `kl` = KL to the teacher distribution ·
`tv` = total-variation distance · `ece` = calibration error · `brier`.

Use a checkpoint from Python:

```python
from jev import JEVPredictor
jev = JEVPredictor("runs/jev-9b/best")
jev.predict(kind="noul", state="Canary p99 latency is up 40% after the deploy.",
            question="Should the rollout be paused?", options=["false", "true"])
# {'false': 0.08, 'true': 0.92}
```

## Code map

| file | |
|---|---|
| `jev/schema.py` | `JEVExample`, validation, JSONL I/O |
| `jev/sources.py` | data-source registry: `jev_distill`, `jsonl` (your own files), `commonsense_qa` |
| `jev/collate.py` | prompt, 24-slot mapping, 60/40 head-tail state truncation, length-grouped DDP sampler |
| `jev/model.py` | `JEVModel`: backbone + LoRA + head; `build` (fresh) / `load` / `save` |
| `jev/losses.py` | KL, RPS, per-kind temperature scaling, metrics |
| `jev/trainer.py` | `Trainer`: DDP training, sharded eval, checkpoint/resume, calibration, report |
| `jev/launch.py` | reads `train.num_gpus`, re-executes under torchrun |
| `jev/predict.py` | `JEVPredictor` + CLI |

## Extending

- **Your own data in this format:** `data.source: jsonl` and `data.path: <dir>` with `train/validation/calibration/<eval>.jsonl`.
- **A new dataset:** register an adapter; the trainer does not change.
  ```python
  from jev import register_source, JEVExample
  @register_source("my_data")
  def my_data(cfg, split):
      return [JEVExample(kind="choice", state=r.text, question=r.q, options=r.opts, target=r.probs) for r in ...]
  ```
- **Filtering:** `data.include: {source: [openjev_v2]}`, `data.exclude: {family: [theology]}`.
- **Continuing from a trained model:** `JEVModel.load(ckpt, is_trainable=True)` returns a model the same
  `Trainer` accepts. This is the hook for domain fine-tuning on top of a general JEV.

## License

Code: Apache-2.0. The corpus is Apache-2.0 (its Open-Jev stream CC0); its `yuri_v3` labels are outputs of
the closed TypeSafe Jev 1.13 model. Check that model's terms before commercial use of derived weights.

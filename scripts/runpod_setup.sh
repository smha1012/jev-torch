#!/usr/bin/env bash
# One-shot RunPod setup.
#
#   Image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404  (torch 2.8.0 · CUDA 12.8.1 · Ubuntu 24.04)
#
#   cd /workspace && git clone https://github.com/smha1012/jev-torch.git && cd jev-torch
#   read -rsp "HF_TOKEN: " T && echo "HF_TOKEN=$T" > .env.local    # tokens, typed hidden (see docs/runpod.md)
#   bash scripts/runpod_setup.sh                       # install + checks + GPU smoke test
#   bash scripts/runpod_setup.sh configs/jev-9b.yaml   # ... and pre-download that run's model
#
#   SKIP_SMOKE=1 bash scripts/runpod_setup.sh          # skip the ~3 min GPU smoke test
#
# ⚠️ Never replace the image's torch. Installing transformers & co. without a constraint can pull a newer
#    torch built for a different CUDA. torch is pinned to the image version, and re-checked after install.
# ⚠️ Every dependency has an upper bound (pyproject.toml): an unbounded package installs whatever is newest
#    on the day the pod is created and can break a run that worked yesterday. After the first successful
#    run, freeze the exact set:  pip freeze > scripts/requirements.lock  (this script then installs from it).
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-}"
PY="${PYTHON:-python3}"
export PIP_BREAK_SYSTEM_PACKAGES=1   # Ubuntu 24.04 marks the system Python "externally managed" (PEP 668)

say() { printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }

say "Environment"
$PY -c "import sys, torch; print(f'  python {sys.version.split()[0]} · torch {torch.__version__} · cuda {torch.version.cuda} · {torch.cuda.device_count()} GPU(s)')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /'
NGPU="$($PY -c 'import torch; print(torch.cuda.device_count())')"

say "System packages"
if command -v apt-get >/dev/null; then
  (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux >/dev/null) \
    && echo "  tmux installed" || echo "  (apt failed; tmux is optional)"
fi

say "Caches on the network volume"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p "$HF_HOME"
if ! grep -q "jev-torch env" ~/.bashrc 2>/dev/null; then
  cat >> ~/.bashrc <<EOF
# jev-torch env
export HF_HOME=$HF_HOME
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF
fi
echo "  HF_HOME=$HF_HOME"

say "Pin torch (keep the image's build)"
TORCH_VER="$($PY -c 'import torch; print(torch.__version__.split("+")[0])')"
echo "torch==${TORCH_VER}" > /tmp/jev-constraints.txt
echo "  torch==${TORCH_VER}"

say "Python dependencies"
$PY -m pip install -q --upgrade pip
if [ -f scripts/requirements.lock ]; then
  echo "  installing the frozen set from scripts/requirements.lock"
  grep -v -E '^(torch|nvidia-|triton|-e |jev-torch)' scripts/requirements.lock > /tmp/jev-lock.txt
  $PY -m pip install -q -c /tmp/jev-constraints.txt -r /tmp/jev-lock.txt
fi
$PY -m pip install -q -c /tmp/jev-constraints.txt -e ".[cuda,dev]"
$PY -c "import transformers, peft, fla; print(f'  transformers {transformers.__version__} · peft {peft.__version__} · flash-linear-attention {fla.__version__}')"

say "Optional: causal-conv1d"
# Small speedup for Qwen3.5's short convolution; transformers falls back to torch without it.
# Building needs nvcc; without it, skip instead of failing slowly.
if command -v nvcc >/dev/null; then
  timeout 1200 $PY -m pip install -q -c /tmp/jev-constraints.txt --no-build-isolation causal-conv1d \
    && echo "  installed" || echo "  (not installed; using the torch fallback)"
else
  echo "  skipped: no nvcc in this image (torch fallback is used)"
fi

say "Re-check torch integrity"
# Even with the constraint, a resolver can swap torch for a PyPI build of another CUDA. Catch it here.
$PY - <<PY
import sys, torch
now, want = torch.__version__.split("+")[0], "${TORCH_VER}"
print(f"  torch {torch.__version__} · cuda {torch.version.cuda} · available {torch.cuda.is_available()} "
      f"· bf16 {torch.cuda.is_available() and torch.cuda.is_bf16_supported()}")
if now != want:
    sys.exit(f"!! torch changed {want} -> {now}; the image's CUDA build was replaced")
if not torch.cuda.is_available():
    sys.exit("!! torch.cuda.is_available() is False")
PY
# Installing is not importing: check the kernels actually import.
$PY - <<'PY'
import importlib
for mod, why in [("fla", "REQUIRED for speed: Qwen3.5 gated-delta-rule kernels"),
                 ("causal_conv1d", "optional"), ("bitsandbytes", "only for load_in_4bit configs"),
                 ("wandb", "W&B logging")]:
    try:
        m = importlib.import_module(mod)
        print(f"  ok       {mod} {getattr(m, '__version__', '')}")
    except Exception as e:
        print(f"  missing  {mod}: {why} ({type(e).__name__}: {e})")
PY

say "Tokens"
set +u
if [ -f .env.local ]; then set -a; . ./.env.local; set +a; echo "  .env.local loaded"; fi
set -u
if [ -n "${HF_TOKEN:-}" ]; then
  # Training pushes checkpoints with this token (train.hf_push: auto), so it must be able to write.
  $PY - <<'PY'
import os, sys
from huggingface_hub import HfApi
try:
    who = HfApi().whoami(token=os.environ["HF_TOKEN"])
except Exception as e:
    sys.exit(f"  !! HF_TOKEN rejected by the Hub: {e}")
role = who.get("auth", {}).get("accessToken", {}).get("role", "unknown")
print(f"  HF_TOKEN      OK · account {who['name']} · role {role}")
if role == "read":
    print("  !! read-only token: Hub pushes will fail. Use a write token, or train with --set train.hf_push=null")
PY
else
  echo "  HF_TOKEN      not set: checkpoints stay local (no Hub uploads), anonymous downloads"
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
  $PY -m wandb login --relogin "$WANDB_API_KEY" >/dev/null && echo "  WANDB_API_KEY OK"
else
  echo "  WANDB_API_KEY not set: W&B logs offline (wandb sync later)"
fi

say "Unit tests"
$PY -m pytest -q tests

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  SMOKE_GPUS=$(( NGPU >= 2 ? 2 : 1 ))
  say "GPU smoke test: Qwen3.5-0.8B, 4 steps, ${SMOKE_GPUS} GPU(s)"
  rm -rf runs/smoke
  # JEV_OVERRIDES is blanked so personal settings (e.g. a Hub target) never apply to the smoke test.
  if JEV_OVERRIDES= $PY -m jev.launch --config configs/smoke.yaml --set train.num_gpus=$SMOKE_GPUS > /tmp/jev-smoke.log 2>&1 \
     && [ -f runs/smoke/report.json ]; then
    grep -E "tok/s|temperatures|test_set_30k calibrated" /tmp/jev-smoke.log | tail -4 | sed 's/^/  /'
    if grep -q "falling back to its reference PyTorch implementation" /tmp/jev-smoke.log; then
      echo "  !! a kernel fell back to the slow PyTorch path:"
      grep "falling back" /tmp/jev-smoke.log | sort -u | sed 's/^/     /'
    fi
    echo "  smoke test passed (log: /tmp/jev-smoke.log)"
  else
    tail -30 /tmp/jev-smoke.log
    echo "  !! smoke test FAILED. Full log: /tmp/jev-smoke.log"
    exit 1
  fi
fi

if [ -n "$CONFIG" ]; then
  say "Pre-downloading the model for $CONFIG"
  $PY - "$CONFIG" <<'PY'
import sys
from huggingface_hub import snapshot_download
from jev.config import load_config
cfg = load_config(sys.argv[1])
snapshot_download(cfg.model.name)
print(f"  {cfg.model.name} cached")
PY
fi

cat <<EOF

────────────────────────────────────────────────────────────
Setup done. Start training (runs in the background, survives SSH disconnects):

  bash scripts/runpod_train.sh ${CONFIG:-configs/jev-9b.yaml}
  tail -f runs/<name>/train.log

  # more GPUs:  --set train.num_gpus=4
  # no Hub:     --set train.hf_push=null

Watch the first few log lines: tok/s and ETA tell you if the GPU choice is right,
and nvidia-smi shows the memory headroom for micro_batch_size.
After the first successful run, freeze the versions:
  pip freeze > scripts/requirements.lock && git add scripts/requirements.lock
────────────────────────────────────────────────────────────
EOF

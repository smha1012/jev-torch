#!/usr/bin/env bash
# One-time setup on a RunPod pod.
# Tested target image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 (torch 2.8.0, CUDA 12.8.1).
#
#   cd /workspace && git clone <this repo> jev-torch && cd jev-torch
#   bash scripts/runpod_setup.sh                          # install + checks
#   bash scripts/runpod_setup.sh configs/jev-9b.yaml      # ... and pre-download that model + dataset
#
# Tokens: put HF_TOKEN / WANDB_API_KEY in .env.local (cp .env.sample .env.local). Both are optional:
# without HF_TOKEN downloads are rate-limited; without WANDB_API_KEY W&B logs offline.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-}"
# Tokens live in .env.local (gitignored; template: .env.sample): HF_TOKEN, WANDB_API_KEY, ...
if [ -f .env.local ]; then set -a; . ./.env.local; set +a; fi

# Keep caches on the network volume so they survive pod restarts.
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

echo "== GPUs"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "== Python packages"
# Pin torch to the version shipped in the image so pip never replaces the CUDA build.
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo "torch==${TORCH_VER}" > /tmp/jev-constraints.txt
python -m pip install -q --upgrade pip
python -m pip install -q -c /tmp/jev-constraints.txt -e ".[cuda,dev]"
# causal-conv1d is an optional fast path for Qwen3.5's short conv; transformers falls back to torch without it.
python -m pip install -q -c /tmp/jev-constraints.txt --no-build-isolation causal-conv1d \
  || echo "   (causal-conv1d not installed: optional, using the torch fallback)"

[ -n "${HF_TOKEN:-}" ] && echo "HF_TOKEN: set" || echo "HF_TOKEN: not set (anonymous downloads)"
if [ -n "${WANDB_API_KEY:-}" ]; then
  python -m wandb login --relogin "$WANDB_API_KEY" >/dev/null && echo "W&B: logged in"
else
  echo "W&B: WANDB_API_KEY not set (runs will log offline; sync later with 'wandb sync runs/<name>/wandb')"
fi

echo "== Environment check"
python - <<'EOF'
import importlib, torch
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | gpus {torch.cuda.device_count()} "
      f"| bf16 {torch.cuda.is_bf16_supported()}")
for mod, why in [("fla", "flash-linear-attention: fast Qwen3.5 gated delta rule (required for speed)"),
                 ("causal_conv1d", "optional conv fast path"),
                 ("bitsandbytes", "only for load_in_4bit / QLoRA configs"),
                 ("transformers", ""), ("peft", "")]:
    try:
        m = importlib.import_module(mod)
        print(f"  ok   {mod} {getattr(m, '__version__', '')}")
    except Exception as e:
        print(f"  MISSING {mod}: {why} ({type(e).__name__})")
EOF

echo "== Unit tests"
python -m pytest -q tests

if [ -n "$CONFIG" ]; then
  echo "== Pre-downloading model and dataset for $CONFIG"
  python - "$CONFIG" <<'EOF'
import sys
from huggingface_hub import snapshot_download
from jev.config import load_config
cfg = load_config(sys.argv[1])
print("model:", cfg.model.name)
snapshot_download(cfg.model.name)
if cfg.data.source == "jev_distill":
    from datasets import load_dataset
    load_dataset("SargeDev/jev-distill-corpus-v3")
print("done")
EOF
fi

echo
echo "Setup complete. Start training with:"
echo "  bash scripts/runpod_train.sh ${CONFIG:-configs/jev-9b.yaml}"

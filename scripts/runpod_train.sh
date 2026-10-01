#!/usr/bin/env bash
# Start (or resume) training in the background on a RunPod pod. The GPU count comes from the config
# (train.num_gpus); overrides are passed straight through.
#
#   bash scripts/runpod_train.sh configs/jev-9b.yaml
#   bash scripts/runpod_train.sh configs/jev-27b.yaml --set train.num_gpus=8 train.micro_batch_size=16
#
# Checkpoints go to the Hub automatically when HF_TOKEN is set (train.hf_push: auto).
# Always runs with --resume: if the pod was restarted, training continues from <output_dir>/last.
# Use a new train.output_dir for a new experiment.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:?usage: runpod_train.sh CONFIG [--set section.field=value ...]}"
shift
PY="${PYTHON:-python3}"
# Tokens live in .env.local (gitignored; template: .env.sample): HF_TOKEN, WANDB_API_KEY, ...
set +u
if [ -f .env.local ]; then set -a; . ./.env.local; set +a; fi
set -u

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=$($PY - --config "$CONFIG" "$@" <<'EOF'
import sys
from jev.config import load_config
from jev.train import parse_args
a = parse_args(sys.argv[1:])
print(load_config(a.config, a.set).train.output_dir)
EOF
)
mkdir -p "$OUT"
case "$(realpath "$OUT")" in
  /workspace/*) ;;
  *) echo "warning: $OUT is not on /workspace; checkpoints will be lost if the pod is stopped" ;;
esac

LOG="$OUT/train.log"
nohup $PY -m jev.launch --config "$CONFIG" --resume "$@" >> "$LOG" 2>&1 &
echo "started pid $! -> $LOG"
echo "  follow:  tail -f $LOG"
echo "  stop:    pkill -f 'jev\.(launch|train)'"

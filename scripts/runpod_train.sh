#!/usr/bin/env bash
# Start (or resume) training in the background on a RunPod pod. The GPU count comes from the config
# (train.num_gpus); overrides are passed straight through.
#
#   bash scripts/runpod_train.sh configs/jev-9b.yaml
#   bash scripts/runpod_train.sh configs/jev-27b.yaml --set train.num_gpus=8 train.micro_batch_size=16
#
# Shows the log live (Ctrl+C stops watching only; JEV_NO_FOLLOW=1 to just start). Checkpoints go to the
# Hub (train.hf_push: auto), so a token is required unless you pass --set train.hf_push=null.
# Always runs with --resume: if the pod was restarted, training continues from <output_dir>/last.
# Use a new train.output_dir for a new experiment.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:?usage: runpod_train.sh CONFIG [--set section.field=value ...]}"
shift
PY="${PYTHON:-python3}"
# Tokens: this shell's variables, RunPod pod variables, then .env.local (see scripts/_env.sh).
. scripts/_env.sh

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
# TileLang (used by fla on Hopper GPUs) compiles kernels at runtime and needs nvcc.
if [ -z "${CUDA_HOME:-}" ]; then
  for d in /usr/local/cuda /usr/local/cuda-*; do [ -x "$d/bin/nvcc" ] && export CUDA_HOME="$d" && break; done
fi
[ -n "${CUDA_HOME:-}" ] && export PATH="$CUDA_HOME/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

read -r OUT HF_PUSH < <($PY - --config "$CONFIG" "$@" <<'EOF'
import sys
from jev.config import load_config
from jev.env import env_overrides
from jev.train import parse_args
a = parse_args(sys.argv[1:])
t = load_config(a.config, env_overrides() + a.set).train
print(t.output_dir, t.hf_push or "null")
EOF
)

# Without a token the run would finish with nothing uploaded. Refuse instead of skipping silently.
if [ "$HF_PUSH" != "null" ] && [ -z "${HF_TOKEN:-}" ] \
   && [ "$($PY -c 'from huggingface_hub import get_token; print(bool(get_token()))')" != "True" ]; then
  echo "!! No HF token found (checked this shell, RunPod pod variables and .env.local)."
  echo "   The trained model would not be uploaded. Either set HF_TOKEN, e.g."
  echo "     echo \"HF_TOKEN=hf_...\" > .env.local"
  echo "   or run without uploads by adding:  --set train.hf_push=null"
  exit 1
fi
mkdir -p "$OUT"
case "$(realpath "$OUT")" in
  /workspace/*) ;;
  *) echo "warning: $OUT is not on /workspace; checkpoints will be lost if the pod is stopped" ;;
esac

[ -n "${JEV_OVERRIDES:-}" ] && echo "personal overrides (JEV_OVERRIDES): $JEV_OVERRIDES"
LOG="$OUT/train.log"
# Own session (setsid): Ctrl+C or closing the terminal can never reach the training processes.
DETACH=""; command -v setsid >/dev/null && DETACH="setsid -w"
$DETACH nohup $PY -m jev.launch --config "$CONFIG" --resume "$@" >> "$LOG" 2>&1 < /dev/null &
PID=$!
echo "started pid $PID -> $LOG"
echo "  stop training:  pkill -f 'jev\.(launch|train)'"
[ "${JEV_NO_FOLLOW:-0}" = "1" ] && exit 0

# Show the log live. Training runs in the background: Ctrl+C (or closing the terminal) only stops watching.
echo "  showing the log below; Ctrl+C stops watching, training keeps running"
echo
trap 'echo; echo "Stopped watching. Training continues (pid $PID). Watch again: tail -f $LOG"; exit 0' INT
tail -n 0 --pid="$PID" -f "$LOG"
echo
if grep -q "^report -> " "$LOG"; then
  echo "Training finished: $OUT/report.json"
else
  echo "!! The training process exited. Look for the error above, or: grep -n -A25 Traceback $LOG"
fi

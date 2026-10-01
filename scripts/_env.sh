# Sourced by the RunPod scripts: make tokens and personal settings visible to this shell.
#
# Priority: variables already exported in this shell > RunPod pod variables > .env.local.
# RunPod injects pod variables into the container's first process, but SSH / Jupyter terminals do not
# always inherit them, so they are read from /proc/1/environ as well. Empty values never override.

_JEV_KEYS="HF_TOKEN WANDB_API_KEY WANDB_ENTITY TZ JEV_OVERRIDES HF_HOME"

_jev_pod_var() {
  [ -r /proc/1/environ ] || return 0
  tr '\0' '\n' < /proc/1/environ | grep -m1 "^$1=" | cut -d= -f2- || true
}

for _k in $_JEV_KEYS; do
  if [ -z "${!_k:-}" ]; then
    _v="$(_jev_pod_var "$_k")"
    [ -n "$_v" ] && export "$_k=$_v"
  fi
done

if [ -f .env.local ]; then
  while IFS= read -r _line || [ -n "$_line" ]; do
    case "$_line" in ''|\#*) continue ;; esac
    _k="${_line%%=*}"; _k="${_k// /}"
    [[ "$_k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    _v="${_line#*=}"
    _v="${_v%\"}"; _v="${_v#\"}"; _v="${_v%\'}"; _v="${_v#\'}"
    [ -n "$_v" ] && [ -z "${!_k:-}" ] && export "$_k=$_v"
  done < .env.local
fi
unset _k _v _line

"""Load tokens from .env.local into os.environ (no python-dotenv dependency).

Variables already set in the environment win, so values injected by RunPod's dashboard or a
shell `export` are never overwritten.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path


def load_env(path: str | Path = ".env.local") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:  # an empty `HF_TOKEN=` line in the template means "not set"
            os.environ.setdefault(key.strip(), value)


def env_overrides() -> list[str]:
    """Config overrides from JEV_OVERRIDES, e.g. in .env.local:

        JEV_OVERRIDES="train.hf_push=me/jev-9b data.hf_dataset=me/jev-distill-corpus-v3"

    Personal settings stay out of the (public) configs and out of git. Applied before --set, so the
    command line still wins.
    """
    return shlex.split(os.environ.get("JEV_OVERRIDES", ""))

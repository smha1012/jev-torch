"""Snapshot a Hugging Face dataset into your own namespace, pinned to one source commit.

    HF_TOKEN=<write token> python scripts/mirror_dataset.py \
        --src SargeDev/jev-distill-corpus-v3 --dst your-name/jev-distill-corpus-v3 [--private]

Why: training then reads a copy that cannot change or disappear under you. Data files are byte-identical.
The source's dataset card (including its split -> file mapping, which `load_dataset` relies on) is kept
verbatim, with a one-line "Mirrored from <source> @ <commit>" footer for attribution.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def build_card(original: str, src: str, sha: str) -> str:
    """The source card verbatim, plus a one-line attribution footer."""
    footer = (f"\n\n---\n<sub>Mirrored from [{src}](https://huggingface.co/datasets/{src}) "
              f"@ [`{sha[:7]}`](https://huggingface.co/datasets/{src}/tree/{sha})</sub>\n")
    return original.rstrip() + footer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--revision", help="source commit / tag to snapshot (default: current main)")
    p.add_argument("--private", action="store_true")
    p.add_argument("--card_only", action="store_true", help="only refresh the README of an existing mirror")
    args = p.parse_args()

    api = HfApi()
    info = api.dataset_info(args.src, revision=args.revision)
    sha = info.sha
    print(f"source {args.src} @ {sha}")

    tag = f"src-{sha[:12]}"
    with tempfile.TemporaryDirectory() as tmp:
        patterns = ["README.md"] if args.card_only else None
        local = Path(snapshot_download(args.src, repo_type="dataset", revision=sha, local_dir=tmp,
                                       allow_patterns=patterns))
        (local / "README.md").write_text(build_card((local / "README.md").read_text(), args.src, sha))

        api.create_repo(args.dst, repo_type="dataset", private=args.private, exist_ok=True)
        commit = api.upload_folder(repo_id=args.dst, repo_type="dataset", folder_path=str(local),
                                   allow_patterns=patterns, ignore_patterns=[".cache/*"],
                                   commit_message=f"Snapshot of {args.src}@{sha[:12]}"
                                   if not args.card_only else "Update dataset card")
        try:  # (re)point the tag at this commit; data files are unchanged either way
            api.delete_tag(args.dst, repo_type="dataset", tag=tag)
        except Exception:
            pass
        api.create_tag(args.dst, repo_type="dataset", tag=tag, revision=commit.oid)
    print(f"mirrored -> https://huggingface.co/datasets/{args.dst} (tag {tag})")


if __name__ == "__main__":
    main()

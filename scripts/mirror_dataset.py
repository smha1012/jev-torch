"""Snapshot a Hugging Face dataset into your own namespace, pinned to one source commit.

    HF_TOKEN=<write token> python scripts/mirror_dataset.py \
        --src SargeDev/jev-distill-corpus-v3 --dst seungminh/jev-distill-corpus-v3 [--private]

Why: training then reads a copy that cannot change or disappear under you. The source's dataset card
(including its split -> file mapping, which `load_dataset` relies on) is kept verbatim, with a
provenance note prepended: source repo, exact commit, date, and license. Files are byte-identical.
"""

from __future__ import annotations

import argparse
import datetime as dt
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def provenance_note(src: str, sha: str, today: str) -> str:
    return f"""
> [!NOTE]
> **Snapshot mirror.** This is an unmodified copy of
> [`{src}`](https://huggingface.co/datasets/{src}) at commit
> [`{sha[:12]}`](https://huggingface.co/datasets/{src}/tree/{sha}), taken on {today}.
> All data files are byte-identical to the source; only this note was added to the card.
> It is pinned so that training with [jev-torch](https://github.com/smha1012/jev-torch) stays reproducible.
> All credit belongs to the original authors. The license (below) is unchanged.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--revision", help="source commit / tag to snapshot (default: current main)")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    api = HfApi()
    info = api.dataset_info(args.src, revision=args.revision)
    sha = info.sha
    print(f"source {args.src} @ {sha}")

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(snapshot_download(args.src, repo_type="dataset", revision=sha, local_dir=tmp))
        card = (local / "README.md").read_text()
        today = dt.date.today().isoformat()
        if card.startswith("---"):
            end = card.index("---", 3) + 3  # keep the YAML front matter (split mapping) on top
            card = card[:end] + "\n" + provenance_note(args.src, sha, today) + card[end:]
        else:
            card = provenance_note(args.src, sha, today) + "\n" + card
        (local / "README.md").write_text(card)

        api.create_repo(args.dst, repo_type="dataset", private=args.private, exist_ok=True)
        commit = api.upload_folder(repo_id=args.dst, repo_type="dataset", folder_path=str(local),
                                   ignore_patterns=[".cache/*"],
                                   commit_message=f"Snapshot of {args.src}@{sha[:12]}")
        api.create_tag(args.dst, repo_type="dataset", tag=f"src-{sha[:12]}", revision=commit.oid, exist_ok=True)
    print(f"mirrored -> https://huggingface.co/datasets/{args.dst} (tag src-{sha[:12]})")


if __name__ == "__main__":
    main()

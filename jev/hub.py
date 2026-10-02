"""Hugging Face Hub uploads during training.

`train.hf_push` decides where checkpoints go:
    "auto"         (default) push to <token account>/<output_dir name> if an HF token is available,
                   otherwise skip quietly
    "owner/name"   push there; a missing token or missing write access stops the run at startup
    null           never push

Checks run before any GPU time is spent: token present, the account may write to the namespace
(user or one of its orgs), the repo exists (created private by default), and a warning if it already
holds files that this run will overwrite. Uploads during training never crash the run; a failed push
is reported and training continues.

What gets pushed: the current model at the end of every epoch (tag `epoch-N`, not yet calibrated),
and the calibrated best checkpoint with its evaluation report at the end (tag `final`).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from .model import CHECKPOINT_FILES


def resolve_repo(hf_push: str | None, account: str, orgs: list[str], run_name: str) -> str | None:
    """Repo id for this run, or raise ValueError if the account cannot write to it."""
    if hf_push is None:
        return None
    repo = f"{account}/{run_name}" if hf_push == "auto" else hf_push
    if repo.count("/") != 1:
        raise ValueError(f'hf_push must be "auto", null or "owner/name", got {hf_push!r}')
    namespace = repo.split("/")[0]
    if namespace != account and namespace not in orgs:
        raise ValueError(f'the HF token belongs to {account!r}, which cannot write to "{namespace}" '
                         f"(orgs: {orgs or 'none'}); use a token with write access to that namespace")
    return repo


def stage_checkpoint(ckpt: Path, stage: Path, card: str | None = None) -> Path:
    """Copy the files a checkpoint needs on the Hub into `stage`, plus a README model card."""
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy(ckpt / "jev_config.json", stage)
    shutil.copy(ckpt / "head.pt", stage)
    shutil.copytree(ckpt / "adapter", stage / "adapter", dirs_exist_ok=True)
    (stage / "adapter" / "README.md").unlink(missing_ok=True)  # PEFT's empty template card
    if card is not None:
        (stage / "README.md").write_text(card)
    return stage


class HubUploader:
    def __init__(self, repo: str, private: bool):
        from huggingface_hub import HfApi

        self.repo, self.private = repo, private
        self.api = HfApi()

    @classmethod
    def setup(cls, hf_push: str | None, private: bool, run_name: str) -> "HubUploader | None":
        """Validate everything up front. Returns None when pushing is disabled or skipped;
        raises SystemExit when an explicit target cannot be used."""
        if hf_push is None:
            return None
        from huggingface_hub import HfApi, get_token

        if not (os.environ.get("HF_TOKEN") or get_token()):
            if hf_push == "auto":
                print("Hub: no HF token (HF_TOKEN / .env.local / hf auth login); checkpoints stay local")
                return None
            raise SystemExit(f"[!] train.hf_push={hf_push!r} needs an HF token (HF_TOKEN or .env.local)")
        api = HfApi()
        try:
            who = api.whoami()
        except Exception as e:
            raise SystemExit(f"[!] HF token rejected by the Hub: {e}")
        try:
            repo = resolve_repo(hf_push, who["name"], [o["name"] for o in who.get("orgs", [])], run_name)
        except ValueError as e:
            raise SystemExit(f"[!] {e}")
        try:
            api.create_repo(repo, private=private, exist_ok=True)
        except Exception as e:
            raise SystemExit(f"[!] cannot create or access {repo} with this HF token ({type(e).__name__}: {e}). "
                             f"It needs WRITE access; or set train.hf_push=null to keep checkpoints local.")
        try:
            existing = [f for f in api.list_repo_files(repo) if f not in ("README.md", ".gitattributes")]
        except Exception:
            existing = []
        if existing:
            print(f"Hub: ⚠️ {repo} already has {len(existing)} files; this run will overwrite them "
                  f"(earlier versions stay in the repo history). Use a new train.hf_push to compare runs.")
        print(f"Hub: pushing to https://huggingface.co/{repo} as {who['name']} "
              f"({'private' if private else 'public'})")
        return cls(repo, private)

    def push_card(self, card: str, message: str = "Update model card") -> bool:
        """Replace only README.md; weights and tags stay as they are."""
        try:
            self.api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                                 repo_id=self.repo, commit_message=message)
            print(f"Hub: updated the model card -> https://huggingface.co/{self.repo}", flush=True)
            return True
        except Exception as e:
            print(f"Hub: ⚠️ card update failed ({type(e).__name__}: {e})", flush=True)
            return False

    def push(self, ckpt: Path, card: str, message: str, tag: str | None = None) -> bool:
        """Upload one checkpoint. Never raises: a failed push must not kill a long training run."""
        try:
            with tempfile.TemporaryDirectory() as tmp:
                stage = stage_checkpoint(Path(ckpt), Path(tmp), card)
                info = self.api.upload_folder(repo_id=self.repo, folder_path=str(stage),
                                              allow_patterns=[*CHECKPOINT_FILES, "README.md"],
                                              commit_message=message)
            if tag:
                try:  # re-point the tag if a previous or resumed run already created it
                    self.api.delete_tag(self.repo, tag=tag)
                except Exception:
                    pass
                self.api.create_tag(self.repo, tag=tag, revision=info.oid)
            print(f"Hub: pushed {message!r} -> https://huggingface.co/{self.repo}"
                  + (f" (tag {tag})" if tag else ""), flush=True)
            return True
        except Exception as e:
            print(f"Hub: ⚠️ push failed ({type(e).__name__}: {e}); training continues", flush=True)
            return False

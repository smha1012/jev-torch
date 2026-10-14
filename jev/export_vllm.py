"""Export a checkpoint as a vLLM LoRA module, so decisions can be served by a stock vLLM server.

    python -m jev.export_vllm --ckpt runs/jev-9b/best --out runs/jev-9b/adapter_vllm
    python -m jev.export_vllm --ckpt your-name/jev-9b --backbone autotrust/JEV-9B --out adapter_vllm

The model scores slot i as  head(h)[i] = W_head[i] · h + b[i],  where h is the final hidden state at the
last prompt token: exactly the input of the backbone's lm_head. A LoRA on lm_head whose delta is
W_head - W_lm on the 24 verbalizer rows (rank <= 24, zero on every other row) therefore makes vLLM's
next-token logits of the verbalizer tokens equal the head's slot logits without the bias. The backbone
LoRA is zero-padded to the same rank (its delta is unchanged), and the bias and per-kind temperatures go
to decision_head.json for the client (jev/vllm_client.py), which reads the verbalizer logprobs restricted
by `allowed_token_ids` and adds them back. This is the packaging autotrust/JEV-9B ships as adapter_vllm/.

Serve:
    vllm serve <backbone> --enable-lora --max-lora-rank 32 --lora-modules jev-decision=<out> \\
        --logprobs-mode processed_logprobs
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from .collate import KIND_SLOTS, SLOT_VERBALIZERS
from .config import ModelConfig
from .schema import KINDS

_LAYER_KEY = re.compile(r"^base_model\.model\.(?:model\.)?(?:language_model\.)?(layers\..+)$")


def pad_lora(name: str, w: torch.Tensor, rank: int) -> torch.Tensor:
    """Zero-pad a LoRA A [r, in] or B [out, r] matrix to `rank`; B @ A is unchanged."""
    if ".lora_A." in name:
        return torch.cat([w, w.new_zeros(rank - w.shape[0], w.shape[1])], 0)
    if ".lora_B." in name:
        return torch.cat([w, w.new_zeros(w.shape[0], rank - w.shape[1])], 1)
    raise ValueError(f"not a LoRA weight: {name}")


def rename_layer_key(name: str, prefix: str) -> str:
    """PEFT key saved from the bare backbone -> the key vLLM expects for the full causal LM."""
    m = _LAYER_KEY.match(name)
    if not m:
        raise ValueError(f"unexpected adapter key {name}")
    return f"base_model.model.{prefix}{m.group(1)}"


def head_to_lm_head_lora(w_lm: torch.Tensor, w_head: torch.Tensor, ids: list[int], rank: int, scale: float):
    """LoRA (A [rank, H], B [V, rank]) with  w_lm[ids] + scale * B[ids] @ A == w_head  and B zero elsewhere."""
    if len(set(ids)) != len(ids) or len(ids) > rank:
        raise ValueError(f"need {len(ids)} distinct verbalizer ids and rank >= {len(ids)}")
    delta = w_head.float() - w_lm[ids].float()
    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    k = s.numel()  # min(len(ids), H)
    a = torch.zeros(rank, w_lm.shape[1])
    a[:k] = vh
    b = torch.zeros(w_lm.shape[0], rank)
    b[ids, :k] = u * s / scale
    return a, b


def _backbone_file(backbone: str, filename: str, optional: bool = False) -> Path | None:
    if Path(backbone).is_dir():
        path = Path(backbone) / filename
        return None if optional and not path.exists() else path
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return Path(hf_hub_download(backbone, filename))
    except EntryNotFoundError:
        if optional:
            return None
        raise


def load_backbone(backbone: str) -> tuple[torch.Tensor, str]:
    """(lm_head weight, layer prefix) of a local directory or Hub repo, reading only the shard it needs."""
    from safetensors import safe_open

    config = json.loads(_backbone_file(backbone, "config.json").read_text())
    multimodal = any("ConditionalGeneration" in a for a in config.get("architectures", []))
    if config.get("tie_word_embeddings") or config.get("text_config", {}).get("tie_word_embeddings"):
        raise ValueError(f"{backbone} ties lm_head to the embeddings; a lm_head LoRA would not be separable")
    index = _backbone_file(backbone, "model.safetensors.index.json", optional=True)
    shard = json.loads(index.read_text())["weight_map"]["lm_head.weight"] if index else "model.safetensors"
    with safe_open(str(_backbone_file(backbone, shard)), framework="pt") as f:
        w_lm = f.get_tensor("lm_head.weight")
    return w_lm, "model.language_model." if multimodal else "model."


def export(ckpt: str, out: str | Path, backbone: str | None = None, rank: int = 32) -> dict:
    from safetensors.torch import load_file, save_file

    from .model import load_tokenizer, resolve_checkpoint, verbalizer_ids

    ckpt = resolve_checkpoint(ckpt)
    meta = json.loads((ckpt / "jev_config.json").read_text())
    if meta.get("slot_verbalizers", SLOT_VERBALIZERS) != SLOT_VERBALIZERS:
        raise ValueError(f"{ckpt} uses a different slot layout")
    cfg = ModelConfig(**meta["model"])
    adapter_cfg = json.loads((ckpt / "adapter" / "adapter_config.json").read_text())
    if adapter_cfg.get("use_rslora") or adapter_cfg.get("use_dora"):
        raise ValueError("rsLoRA / DoRA adapters are not supported")
    scale = adapter_cfg["lora_alpha"] / adapter_cfg["r"]
    rank = max(rank, adapter_cfg["r"], len(SLOT_VERBALIZERS))

    backbone = backbone or cfg.name
    w_lm, prefix = load_backbone(backbone)
    ids = verbalizer_ids(load_tokenizer(cfg.name))
    state = torch.load(ckpt / "head.pt", map_location="cpu", weights_only=True)
    w_head, bias = state["head"]["weight"], state["head"]["bias"]
    if w_head.shape != (len(ids), w_lm.shape[1]):
        raise ValueError(f"head {tuple(w_head.shape)} does not match {backbone} lm_head {tuple(w_lm.shape)}")

    tensors, targets = {}, {"lm_head"}
    for name, w in load_file(str(ckpt / "adapter" / "adapter_model.safetensors")).items():
        name = rename_layer_key(name, prefix)
        targets.add(name.split(".")[-3])
        tensors[name] = pad_lora(name, w, rank).to(torch.bfloat16).contiguous()
    a, b = head_to_lm_head_lora(w_lm, w_head, ids, rank, scale)
    a, b = a.to(torch.bfloat16), b.to(torch.bfloat16)
    tensors["base_model.model.lm_head.lora_A.weight"] = a.contiguous()
    tensors["base_model.model.lm_head.lora_B.weight"] = b.contiguous()

    # what vLLM will compute for the verbalizer rows, after bf16 rounding of the LoRA factors
    eff = w_lm[ids].float() + scale * (b[ids].float() @ a.float())
    err = ((eff - w_head.float()).norm() / w_head.float().norm()).item()

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "adapter_model.safetensors"))
    (out / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": backbone,
        "r": rank, "lora_alpha": scale * rank, "lora_dropout": 0.0, "bias": "none",
        "use_rslora": False, "use_dora": False, "fan_in_fan_out": False, "inference_mode": True,
        "target_modules": sorted(targets),
    }, indent=2))
    head = {
        "bias": bias.float().tolist(),
        "verbalizer_ids": ids,
        "verbalizers": SLOT_VERBALIZERS,
        "kind_slots": {k: list(v) for k, v in KIND_SLOTS.items()},
        "temperature": dict(zip(KINDS, state["temperature"].float().tolist())),
        "tokenizer": cfg.name,
        "backbone": backbone,
        "note": "slot logit = verbalizer logprob (allowed_token_ids, processed_logprobs) + bias; "
                "probabilities = softmax(slot logits / temperature[kind])",
    }
    (out / "decision_head.json").write_text(json.dumps(head, indent=1))
    return {"out": str(out), "backbone": backbone, "rank": rank, "lm_head_rel_err": err,
            "targets": sorted(targets), "verbalizer_ids": ids}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint directory or Hugging Face Hub repo id")
    p.add_argument("--out", required=True, help="output directory for the vLLM LoRA module")
    p.add_argument("--backbone", help="model vLLM will serve (default: the checkpoint's base model); "
                                      "a text-only one such as autotrust/JEV-9B avoids the multimodal class")
    p.add_argument("--rank", type=int, default=32, help="LoRA rank of the export (>= 24); pass it as --max-lora-rank")
    args = p.parse_args()

    from .env import load_env

    load_env()  # HF_TOKEN for private or gated checkpoints
    info = export(args.ckpt, args.out, args.backbone, args.rank)
    print(json.dumps(info, indent=1))
    print(f"\nvllm serve {info['backbone']} --enable-lora --max-lora-rank {info['rank']} "
          f"--lora-modules jev-decision={info['out']} --logprobs-mode processed_logprobs")


if __name__ == "__main__":
    main()

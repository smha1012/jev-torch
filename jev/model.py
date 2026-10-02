"""JEV decision model: causal LM backbone (+LoRA) -> final-token hidden -> 24-slot fp32 head.

Works with any HF causal LM whose text tower loads via AutoModelForCausalLM (Qwen2.5, Qwen3,
Qwen3.5 incl. its hybrid linear-attention layers, Llama, ...). LoRA targets missing from a given
architecture are ignored, so one target list covers all of them.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .collate import NUM_SLOTS, SLOT_VERBALIZERS
from .config import ModelConfig
from .schema import KINDS

CHECKPOINT_VERSION = 1


class JEVModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, config: ModelConfig):
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.head = nn.Linear(hidden_size, NUM_SLOTS)  # kept in fp32 even when the backbone is bf16
        # One temperature per kind (noul, choice, score), fit after training on the calibration split.
        self.register_buffer("temperature", torch.ones(len(KINDS)))

    # -- forward ----------------------------------------------------------------

    def forward(self, input_ids, attention_mask, slot_index, option_mask, **_):
        """Raw (uncalibrated) option logits [B, K]; padded options are -inf."""
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(dim=1) - 1
        pooled = h[torch.arange(h.size(0), device=h.device), last]
        with torch.autocast(device_type=h.device.type, enabled=False):
            slots = self.head(pooled.float())  # [B, 24]
        logits = slots.gather(1, slot_index)  # [B, K]
        return logits.masked_fill(~option_mask, float("-inf"))

    def calibrated(self, logits, kind):
        return logits / self.temperature[kind].unsqueeze(-1)

    @torch.no_grad()
    def predict_proba(self, batch):
        return torch.softmax(self.calibrated(self.forward(**batch), batch["kind"]), dim=-1)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # -- construction -------------------------------------------------------------

    @classmethod
    def build(cls, cfg: ModelConfig, dtype: torch.dtype = torch.float32, device_map=None):
        """Fresh model: pretrained backbone + new LoRA + head initialized from verbalizer rows."""
        tokenizer = load_tokenizer(cfg.name)
        lm = _load_causal_lm(cfg, dtype, device_map)
        rows = _verbalizer_rows(lm, tokenizer)
        hidden = lm.get_output_embeddings().weight.shape[1]
        backbone = lm.base_model  # drop lm_head: the decision head replaces it
        del lm

        if cfg.load_in_4bit:
            from peft import prepare_model_for_kbit_training

            backbone = prepare_model_for_kbit_training(
                backbone, use_gradient_checkpointing=cfg.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        elif cfg.gradient_checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            backbone.enable_input_require_grads()

        present = {n.split(".")[-1] for n, m in backbone.named_modules() if isinstance(m, nn.Linear)}
        targets = [t for t in cfg.lora_targets if t in present]
        if not targets:
            raise ValueError(f"none of lora_targets {cfg.lora_targets} exist in {cfg.name}")
        lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                          target_modules=targets, task_type="FEATURE_EXTRACTION")
        backbone = get_peft_model(backbone, lora)

        model = cls(backbone, hidden, cfg)
        # JEV init: slot i starts as the LM-head row of its verbalizer token, so at step 0 the model
        # already equals "zero-shot next-token logits restricted to the answer tokens".
        with torch.no_grad():
            model.head.weight.copy_(rows)
            model.head.bias.zero_()
        return model, tokenizer

    @classmethod
    def load(cls, ckpt_dir: str | Path, dtype: torch.dtype = torch.float32, device_map=None,
             is_trainable: bool = False, revision: str | None = None):
        """Load a checkpoint from a local directory or a Hugging Face Hub repo id
        (e.g. "your-name/jev-9b"). `is_trainable=True` keeps LoRA trainable, e.g. to continue
        training or to fine-tune the general model on new data with the same Trainer."""
        ckpt = resolve_checkpoint(ckpt_dir, revision)
        meta = json.loads((ckpt / "jev_config.json").read_text())
        cfg = ModelConfig(**meta["model"])
        tokenizer = load_tokenizer(cfg.name)
        lm = _load_causal_lm(cfg, dtype, device_map)
        hidden = lm.get_output_embeddings().weight.shape[1]
        base = lm.base_model
        del lm
        if is_trainable and cfg.gradient_checkpointing:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.enable_input_require_grads()
        backbone = PeftModel.from_pretrained(base, ckpt / "adapter", is_trainable=is_trainable)
        model = cls(backbone, hidden, cfg)
        model.load_head(ckpt)
        return model, tokenizer

    # -- persistence ----------------------------------------------------------------

    def save(self, out_dir: str | Path):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(out / "adapter")
        torch.save({"head": self.head.state_dict(), "temperature": self.temperature.cpu()}, out / "head.pt")
        meta = {
            "version": CHECKPOINT_VERSION,
            "model": self.config.__dict__,
            "slot_verbalizers": SLOT_VERBALIZERS,
            "temperature": dict(zip(KINDS, self.temperature.tolist())),
        }
        (out / "jev_config.json").write_text(json.dumps(meta, indent=2))

    def load_head(self, ckpt_dir: str | Path):
        state = torch.load(Path(ckpt_dir) / "head.pt", map_location="cpu")
        self.head.load_state_dict(state["head"])
        self.temperature.copy_(state["temperature"])

    def load_adapter_weights(self, ckpt_dir: str | Path):
        """Copy saved LoRA + head weights into this (already built) model — used for resuming."""
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        path = Path(ckpt_dir) / "adapter" / "adapter_model.safetensors"
        set_peft_model_state_dict(self.backbone, load_file(str(path)))
        self.load_head(ckpt_dir)


CHECKPOINT_FILES = ["jev_config.json", "head.pt", "adapter/*"]


def resolve_checkpoint(path_or_repo: str | Path, revision: str | None = None) -> Path:
    """Local checkpoint directory as-is; otherwise download the checkpoint files from the Hub."""
    path = Path(path_or_repo)
    if path.is_dir():
        if not (path / "jev_config.json").exists():
            raise FileNotFoundError(f"{path} is not a jev-torch checkpoint (no jev_config.json)")
        return path
    if path.exists() or str(path_or_repo).count("/") != 1:
        raise FileNotFoundError(
            f"{path_or_repo!r} is neither a checkpoint directory nor a Hub repo id. On a fresh pod the run "
            "directory does not exist; pass the Hub repo instead (e.g. --model your-name/jev-9b) or "
            "attach the network volume that holds runs/ (docs/runpod.md, 'Same pod or a fresh pod?').")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(path_or_repo), revision=revision, allow_patterns=CHECKPOINT_FILES))


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def _load_causal_lm(cfg: ModelConfig, dtype, device_map):
    kwargs = {"dtype": dtype}
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation
    if device_map is not None:
        kwargs["device_map"] = device_map
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["lm_head"],
        )
    # For multimodal checkpoints (e.g. Qwen3.5) this loads only the text tower.
    return AutoModelForCausalLM.from_pretrained(cfg.name, **kwargs)


def _verbalizer_rows(lm, tokenizer) -> torch.Tensor:
    ids = []
    for word in SLOT_VERBALIZERS:
        # Prefer the " word" token (what follows "[decision]:"); Qwen splits the space off digits,
        # so fall back to the bare token.
        for cand in (" " + word, word):
            tok = tokenizer(cand, add_special_tokens=False)["input_ids"]
            if len(tok) == 1:
                ids.append(tok[0])
                break
        else:
            raise ValueError(f"verbalizer {word!r} is not a single token for this tokenizer")
    return lm.get_output_embeddings().weight[ids].detach().float().cpu()

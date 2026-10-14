"""Score options through a vLLM server that serves an export from jev.export_vllm.

    from jev.vllm_client import JEVVLLMClient
    jev = JEVVLLMClient("http://localhost:8000", "runs/jev-9b/adapter_vllm/decision_head.json")
    jev.predict(kind="noul", state="...", question="Pause the rollout?", options=["false", "true"])
    # -> {"false": ..., "true": ...}   same probabilities as JEVPredictor, up to bf16 rounding

The prompt is built and tokenized by the training collator (prefix / state / suffix tokenized
separately, state truncated 60/40 to max_length) and sent as token ids, so vLLM sees exactly the
sequence the model was trained on and reads its next-token logits at the position the head pools at.
"""

from __future__ import annotations

import json
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from .collate import KIND_SLOTS, JEVCollator, check_example, prompt_parts
from .schema import KINDS, JEVExample


def find_decision_head(model: str) -> str:
    """decision_head.json of an export directory, of the adapter_vllm/ inside a checkpoint, or of a Hub repo."""
    for p in (Path(model) / "decision_head.json", Path(model) / "adapter_vllm" / "decision_head.json"):
        if p.exists():
            return str(p)
    if not Path(model).exists() and model.count("/") == 1:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(model, "adapter_vllm/decision_head.json")
    raise FileNotFoundError(f"no decision_head.json in {model} or {model}/adapter_vllm; pass --decision_head")


def decision_probs(logprobs: dict[int, float], ids: list[int], bias: list[float], temperature: float) -> list[float]:
    """Calibrated probabilities from the verbalizer logprobs of one request.

    vLLM returns logprobs normalised over `allowed_token_ids`, i.e. the head logits minus one constant;
    the constant cancels in the softmax, so adding the bias and dividing by the temperature is exact.
    """
    missing = [i for i in ids if i not in logprobs]
    if missing:
        raise RuntimeError(f"verbalizer ids {missing} missing from the response (raise --max-logprobs?)")
    z = [(logprobs[i] + b) / temperature for i, b in zip(ids, bias)]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    return [x / sum(e) for x in e]


NEUTRAL_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0,
                    "presence_penalty": 0.0, "frequency_penalty": 0.0}


class JEVVLLMClient:
    """Drop-in for JEVPredictor (`logits`, `predict_batch`, `predict`, `temperature`) backed by a vLLM server,
    so jev.evaluate / jev.bench can score the served model with the same code."""

    def __init__(self, url: str, decision_head: str, model: str = "jev-decision", max_length: int = 1024,
                 timeout: float = 120.0, concurrency: int = 64):
        from .model import load_tokenizer

        self.url, self.model, self.timeout, self.concurrency = url.rstrip("/"), model, timeout, concurrency
        with open(decision_head) as f:
            self.head = json.load(f)
        self.collator = JEVCollator(load_tokenizer(self.head["tokenizer"]), max_length=max_length)

    @property
    def temperature(self) -> torch.Tensor:
        return torch.tensor([self.head["temperature"][k] for k in KINDS])

    def request(self, ex: JEVExample) -> dict:
        """Body of the /v1/completions request for one question (useful for batching clients)."""
        check_example(ex)
        first, _ = KIND_SLOTS[ex.kind]
        ids = self.head["verbalizer_ids"][first: first + len(ex.options)]
        return {
            "model": self.model,
            "prompt": self.collator._encode(*prompt_parts(ex.kind, ex.state, ex.question, ex.options)),
            "max_tokens": 1, "logprobs": len(ids), "allowed_token_ids": ids,
            "add_special_tokens": False, "return_tokens_as_token_ids": True,
            # processed logprobs reflect sampling settings, and vLLM fills unset ones from the backbone's
            # generation_config.json (Qwen3.8-27B: top_k 20, top_p 0.95, which drops tail options): pin them
            **NEUTRAL_SAMPLING,
        }

    def _logits(self, ex: JEVExample) -> torch.Tensor:
        body = self.request(ex)
        req = urllib.request.Request(f"{self.url}/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            top = json.load(r)["choices"][0]["logprobs"]["top_logprobs"][0]  # {"token_id:123": logprob}
        logprobs = {int(k.split(":", 1)[1]): v for k, v in top.items()}
        ids = body["allowed_token_ids"]
        missing = [i for i in ids if i not in logprobs]
        if missing:
            raise RuntimeError(f"verbalizer ids {missing} missing from the response (raise --max-logprobs?)")
        first, _ = KIND_SLOTS[ex.kind]
        bias = self.head["bias"][first: first + len(ids)]
        return torch.tensor([logprobs[i] + b for i, b in zip(ids, bias)])

    def logits(self, examples: list[JEVExample]) -> list[torch.Tensor]:
        """Uncalibrated logits per example: the head's logits up to a per-example constant (softmax-invariant)."""
        with ThreadPoolExecutor(self.concurrency) as pool:
            return list(pool.map(self._logits, examples))

    def predict_batch(self, examples: list[JEVExample]) -> list[dict[str, float]]:
        temps = self.temperature
        return [dict(zip(ex.options, torch.softmax(lg / temps[KINDS.index(ex.kind)], -1).tolist()))
                for ex, lg in zip(examples, self.logits(examples))]

    def predict(self, state: str, question: str, options: list[str], kind: str = "choice") -> dict[str, float]:
        ex = JEVExample(state=state, question=question, options=options, kind=kind, label=0)
        return self.predict_batch([ex])[0]

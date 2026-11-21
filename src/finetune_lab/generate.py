"""Batched greedy generation for evaluation (left padding, deterministic)."""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from finetune_lab.prompts import render

log = logging.getLogger(__name__)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(name: str, adapter: str | None = None) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    model: Any = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    model = model.to(device())
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    system: str,
    batch_size: int = 32,
    max_new_tokens: int = 16,
) -> tuple[list[str], float]:
    """Return generations and mean milliseconds per example."""
    outputs: list[str] = []
    started = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        prompts = [render(tokenizer, t, system) for t in texts[start : start + batch_size]]
        batch = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        generated = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = generated[:, batch["input_ids"].shape[1] :]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        if (start // batch_size) % 20 == 0:
            log.info("generated %d / %d", len(outputs), len(texts))
    elapsed_ms = (time.perf_counter() - started) * 1000
    return outputs, elapsed_ms / max(1, len(texts))

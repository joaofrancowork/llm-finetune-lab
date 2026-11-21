"""LoRA supervised fine-tuning with a plain PyTorch loop.

No Trainer abstraction on purpose: the whole recipe is visible in one screen.
The loss is computed on the answer tokens only (prompt tokens are masked with
-100), so the model learns to produce the label rather than to repeat prompts.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from finetune_lab.config import LoraConfig
from finetune_lab.generate import device
from finetune_lab.prompts import FINETUNED_SYSTEM, render

log = logging.getLogger(__name__)
IGNORE = -100


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode(tokenizer: Any, text: str, label: str, max_length: int) -> dict[str, list[int]]:
    prompt_ids = tokenizer(render(tokenizer, text, FINETUNED_SYSTEM), add_special_tokens=False)[
        "input_ids"
    ]
    answer_ids = tokenizer(label + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    prompt_ids = prompt_ids[: max_length - len(answer_ids)]
    return {
        "input_ids": prompt_ids + answer_ids,
        "labels": [IGNORE] * len(prompt_ids) + answer_ids,
    }


def collate(pad_id: int) -> Any:
    def fn(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(x["input_ids"]) for x in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), IGNORE, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, x in enumerate(batch):
            n = len(x["input_ids"])
            ids[i, :n] = torch.tensor(x["input_ids"])
            labels[i, :n] = torch.tensor(x["labels"])
            mask[i, :n] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}

    return fn


def train(
    train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: LoraConfig, out: Path
) -> dict[str, Any]:
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    seed_everything(cfg.seed)
    dev = device()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    model: Any = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.float32)
    model = model.to(dev)
    model = get_peft_model(
        model,
        PeftLoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            target_modules=cfg.target_modules,
            task_type="CAUSAL_LM",
        ),
    )
    trainable, total = model.get_nb_trainable_parameters()
    log.info(
        "trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )

    def dataset(df: pd.DataFrame) -> list[dict[str, list[int]]]:
        return [
            encode(tokenizer, t, y, cfg.max_length)
            for t, y in zip(df["text"], df["intent"], strict=True)
        ]

    loader: DataLoader[Any] = DataLoader(
        dataset(train_df),  # type: ignore[arg-type]
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate(tokenizer.pad_token_id),
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    val_loader: DataLoader[Any] = DataLoader(
        dataset(val_df),  # type: ignore[arg-type]
        batch_size=cfg.eval_batch_size,
        collate_fn=collate(tokenizer.pad_token_id),
    )
    steps = math.ceil(len(loader) * cfg.epochs / cfg.gradient_accumulation)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate, weight_decay=0.0
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(steps * cfg.warmup_ratio), steps)

    history: list[dict[str, float]] = []
    step, started = 0, time.perf_counter()
    model.train()
    for epoch in range(math.ceil(cfg.epochs)):
        for i, batch in enumerate(loader):
            if step >= steps:
                break
            batch = {k: v.to(dev) for k, v in batch.items()}
            loss = model(**batch).loss / cfg.gradient_accumulation
            loss.backward()
            if (i + 1) % cfg.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0 or step == steps:
                    history.append(
                        {"step": step, "train_loss": loss.item() * cfg.gradient_accumulation}
                    )
                    log.info(
                        "epoch %d step %d/%d loss %.4f",
                        epoch,
                        step,
                        steps,
                        history[-1]["train_loss"],
                    )

    model.eval()
    with torch.no_grad():
        losses = [model(**{k: v.to(dev) for k, v in b.items()}).loss.item() for b in val_loader]
    summary = {
        "steps": step,
        "minutes": round((time.perf_counter() - started) / 60, 1),
        "val_loss": float(np.mean(losses)),
        "trainable_params": trainable,
        "total_params": total,
        "device": str(dev),
        "config": cfg.model_dump(),
        "history": history,
    }
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(out)
    (out / "training.json").write_text(json.dumps(summary, indent=2))
    return summary

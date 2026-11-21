from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class LoraConfig(BaseModel):
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    rank: int = Field(default=16, ge=1)
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]  # fmt: skip
    learning_rate: float = 2e-4
    epochs: float = 1.0
    batch_size: int = 16
    gradient_accumulation: int = 1
    warmup_ratio: float = 0.05
    max_length: int = 96
    seed: int = 42
    eval_batch_size: int = 32
    max_new_tokens: int = 16

    @classmethod
    def from_yaml(cls, path: Path) -> LoraConfig:
        return cls.model_validate(yaml.safe_load(path.read_text()) or {})

"""banking77: 13,083 customer-support messages labelled with one of 77 intents.

Casanueva et al., "Efficient Intent Detection with Dual Sentence Encoders" (2020).
CC BY 4.0, https://huggingface.co/datasets/PolyAI/banking77. Downloaded at run time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)
REPO = "PolyAI/banking77"
DATA_DIR = Path("data")


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    labels: list[str]


def download(data_dir: Path = DATA_DIR) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    paths = {}
    for split in ("train", "test"):
        paths[split] = Path(
            hf_hub_download(
                REPO,
                f"data/{split}-00000-of-00001.parquet",
                repo_type="dataset",
                local_dir=data_dir,
            )
        )
    return paths


def label_names(parquet_path: Path) -> list[str]:
    """Class names are stored in the Hugging Face features metadata of the parquet file."""
    metadata = pq.read_schema(parquet_path).metadata or {}
    features = json.loads(metadata[b"huggingface"])["info"]["features"]
    names: list[str] = features["label"]["names"]
    return names


def load(paths: dict[str, Path], val_frac: float = 0.1, seed: int = 42) -> Splits:
    labels = label_names(paths["train"])
    frames = {}
    for split, path in paths.items():
        df = pd.read_parquet(path)
        df["intent"] = df["label"].map(lambda i: labels[int(i)])
        frames[split] = df[["text", "intent"]].reset_index(drop=True)
    train, val = train_test_split(
        frames["train"], test_size=val_frac, stratify=frames["train"]["intent"], random_state=seed
    )
    return Splits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=frames["test"],
        labels=labels,
    )

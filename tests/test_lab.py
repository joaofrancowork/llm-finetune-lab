from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from finetune_lab import baseline, cli
from finetune_lab.config import LoraConfig
from finetune_lab.data import label_names, load
from finetune_lab.evaluate import INVALID, normalise, report, score, to_label
from finetune_lab.prompts import FINETUNED_SYSTEM, messages, render, zero_shot_system

LABELS = ["card_arrival", "lost_or_stolen_card", "top_up_failed"]


def write_parquet(path: Path, texts: list[str], labels: list[int]) -> Path:
    features = {"label": {"names": LABELS, "_type": "ClassLabel"}, "text": {"_type": "Value"}}
    table = pa.table({"text": texts, "label": labels})
    meta = {b"huggingface": json.dumps({"info": {"features": features}}).encode()}
    pq.write_table(table.replace_schema_metadata(meta), path)
    return path


@pytest.fixture
def parquet_files(tmp_path: Path) -> dict[str, Path]:
    phrases = {
        0: ["when will my card arrive", "my new card has not arrived yet", "card delivery status"],
        1: ["i lost my card", "someone stole my card", "my card was stolen yesterday"],
        2: ["top up did not work", "my top-up failed", "could not add money top up error"],
    }
    texts, labels = [], []
    for label, examples in phrases.items():
        for i in range(10):
            texts.append(f"{examples[i % 3]} please help {i}")
            labels.append(label)
    return {
        "train": write_parquet(tmp_path / "train.parquet", texts, labels),
        "test": write_parquet(tmp_path / "test.parquet", texts[::3], labels[::3]),
    }


def test_label_names_come_from_parquet_metadata(parquet_files: dict[str, Path]) -> None:
    assert label_names(parquet_files["train"]) == LABELS


def test_load_maps_labels_and_stratifies(parquet_files: dict[str, Path]) -> None:
    splits = load(parquet_files, val_frac=0.2)
    assert set(splits.train["intent"]) == set(LABELS)
    assert splits.val["intent"].value_counts().min() == 2
    assert len(splits.test) == 10 and splits.labels == LABELS


@pytest.mark.parametrize(
    ("generation", "expected"),
    [
        ("card_arrival", "card_arrival"),
        ("Card Arrival", "card_arrival"),
        ("`lost_or_stolen_card`.", "lost_or_stolen_card"),
        ("Intent: top_up_failed", "top_up_failed"),
        ("top-up failed", "top_up_failed"),
        ("lost_or_stolen_cards", "lost_or_stolen_card"),
        ("card_arrival\nBecause the user asks...", "card_arrival"),
        ("I think the customer is asking about money", INVALID),
        ("", INVALID),
    ],
)
def test_to_label(generation: str, expected: str) -> None:
    assert to_label(generation, LABELS) == expected


def test_normalise_strips_prefixes_and_quotes() -> None:
    assert normalise(" 'Label = Card arrival' ") == "card_arrival"


def test_score_counts_invalid_and_confusions() -> None:
    gold = ["card_arrival", "card_arrival", "top_up_failed", "lost_or_stolen_card"]
    predicted = ["card_arrival", INVALID, "top_up_failed", "card_arrival"]
    result = score("m", gold, predicted, latency_ms=12.5)
    assert result.accuracy == 0.5 and result.invalid_rate == 0.25
    assert ("lost_or_stolen_card", "card_arrival", 1) in result.top_confusions
    text = report([result])
    assert "| m | 50.0% |" in text and "most frequent errors" in text
    with pytest.raises(ValueError):
        score("m", ["a"], [])


def test_prompts() -> None:
    system = zero_shot_system(LABELS)
    assert all(label in system for label in LABELS)
    assert messages(" hi ", FINETUNED_SYSTEM)[1] == {"role": "user", "content": "hi"}

    class FakeTokenizer:
        def apply_chat_template(
            self, conversation: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool
        ) -> str:
            assert not tokenize and add_generation_prompt
            return "|".join(m["content"] for m in conversation) + "|assistant:"

    assert render(FakeTokenizer(), "where is my card", "sys") == "sys|where is my card|assistant:"


def test_config_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("rank: 8\nlearning_rate: 0.001\n")
    cfg = LoraConfig.from_yaml(path)
    assert cfg.rank == 8 and cfg.learning_rate == 0.001 and cfg.alpha == 32
    path.write_text("")
    assert LoraConfig.from_yaml(path).rank == 16


def test_baseline_learns_the_toy_intents(parquet_files: dict[str, Path]) -> None:
    splits = load(parquet_files, val_frac=0.2)
    result = baseline.run(splits.train, splits.test)
    assert result.accuracy >= 0.9 and result.invalid_rate == 0.0


def test_cli_baseline_and_report(
    parquet_files: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "download", lambda: parquet_files)
    assert cli.main(["--config", "missing.yaml", "--limit", "6", "baseline"]) == 0
    saved = json.loads((tmp_path / "runs" / "baseline.json").read_text())
    assert saved["n"] == 6
    assert pd.read_csv(tmp_path / "runs" / "baseline-predictions.csv").shape[0] == 6
    capsys.readouterr()
    assert cli.main(["report"]) == 0
    assert "TF-IDF" in capsys.readouterr().out
    assert (tmp_path / "docs" / "results.md").exists()

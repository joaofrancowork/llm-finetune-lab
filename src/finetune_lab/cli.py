"""ft-lab baseline | zero-shot | train | finetuned | report"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from finetune_lab import baseline
from finetune_lab.config import LoraConfig
from finetune_lab.data import download, load
from finetune_lab.evaluate import Scores, report, score, to_label

RUNS = Path("runs")


def _save(scores: Scores, predictions: pd.DataFrame, slug: str) -> None:
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{slug}.json").write_text(json.dumps(asdict(scores), indent=2))
    predictions.to_csv(RUNS / f"{slug}-predictions.csv", index=False)
    print(scores.row())


def _evaluate_llm(
    name: str,
    slug: str,
    test: pd.DataFrame,
    labels: list[str],
    cfg: LoraConfig,
    adapter: str | None,
    system: str,
) -> None:  # pragma: no cover - needs model weights
    from finetune_lab.generate import generate, load_model

    model, tokenizer = load_model(cfg.model, adapter)
    raw, latency = generate(
        model, tokenizer, list(test["text"]), system, cfg.eval_batch_size, cfg.max_new_tokens
    )
    predicted = [to_label(g, labels) for g in raw]
    scores = score(name, list(test["intent"]), predicted, latency)
    _save(scores, test.assign(generation=raw, predicted=predicted), slug)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ft-lab")
    parser.add_argument("--config", type=Path, default=Path("configs/lora.yaml"))
    parser.add_argument("--limit", type=int, help="Evaluate on the first N test examples")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("baseline", "zero-shot", "train", "finetuned", "report"):
        sub.add_parser(name)
    compare_parser = sub.add_parser("compare", help="Error analysis between two prediction files")
    compare_parser.add_argument("--a", type=Path, default=RUNS / "lora-predictions.csv")
    compare_parser.add_argument("--b", type=Path, default=RUNS / "baseline-predictions.csv")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "compare":
        from finetune_lab.analysis import compare, to_markdown

        names = [p.stem.removesuffix("-predictions") for p in (args.a, args.b)]
        result = compare(pd.read_csv(args.a), pd.read_csv(args.b), *names)
        text = to_markdown(result)
        print(text)
        Path("docs").mkdir(exist_ok=True)
        Path("docs/error-analysis.md").write_text(text + "\n")
        return 0

    if args.command == "report":
        results = [Scores(**json.loads(p.read_text())) for p in sorted(RUNS.glob("*.json"))]
        for r in results:
            r.top_confusions = [tuple(c) for c in r.top_confusions]  # type: ignore[misc]
        text = report(sorted(results, key=lambda r: r.accuracy))
        print(text)
        Path("docs").mkdir(exist_ok=True)
        Path("docs/results.md").write_text(text + "\n")
        return 0

    cfg = LoraConfig.from_yaml(args.config) if args.config.exists() else LoraConfig()
    splits = load(download())
    test = splits.test.head(args.limit) if args.limit else splits.test

    if args.command == "baseline":
        scores, predicted = baseline.run(splits.train, test)
        _save(scores, test.assign(predicted=predicted), "baseline")
        return 0
    if args.command == "train":  # pragma: no cover - needs model weights
        from finetune_lab.train import train

        summary = train(splits.train, splits.val, cfg, Path("checkpoints/lora"))
        print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
        return 0

    from finetune_lab.prompts import FINETUNED_SYSTEM, zero_shot_system  # pragma: no cover

    if args.command == "zero-shot":  # pragma: no cover
        _evaluate_llm(
            f"{cfg.model.split('/')[-1]} zero-shot",
            "zero_shot",
            test,
            splits.labels,
            cfg,
            None,
            zero_shot_system(splits.labels),
        )
    else:  # pragma: no cover
        _evaluate_llm(
            f"{cfg.model.split('/')[-1]} + LoRA",
            "lora",
            test,
            splits.labels,
            cfg,
            "checkpoints/lora",
            FINETUNED_SYSTEM,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

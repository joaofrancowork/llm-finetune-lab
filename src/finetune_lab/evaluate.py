"""Turn free-text generations into labels and score them.

A generative classifier can answer with something that is not a label at all.
Those answers are counted as *invalid* (and wrong), never silently mapped to the
closest class at a low similarity, so the invalid rate is visible in the report.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field

from sklearn.metrics import accuracy_score, f1_score

INVALID = "<invalid>"
FUZZY_CUTOFF = 0.9


def normalise(text: str) -> str:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    cleaned = first_line.strip().strip("`'\".").lower()
    cleaned = re.sub(r"^(intent|label)\s*[:=]\s*", "", cleaned)
    return re.sub(r"[\s\-]+", "_", cleaned)


def to_label(generation: str, labels: list[str]) -> str:
    candidate = normalise(generation)
    if candidate in labels:
        return candidate
    close = difflib.get_close_matches(candidate, labels, n=1, cutoff=FUZZY_CUTOFF)
    return close[0] if close else INVALID


@dataclass
class Scores:
    name: str
    n: int
    accuracy: float
    macro_f1: float
    invalid_rate: float
    latency_ms_per_example: float = 0.0
    top_confusions: list[tuple[str, str, int]] = field(default_factory=list)

    def row(self) -> str:
        return (
            f"| {self.name} | {self.accuracy * 100:.1f}% | {self.macro_f1 * 100:.1f}% | "
            f"{self.invalid_rate * 100:.1f}% | {self.latency_ms_per_example:.1f} |"
        )


def score(name: str, gold: list[str], predicted: list[str], latency_ms: float = 0.0) -> Scores:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")
    confusions = Counter((g, p) for g, p in zip(gold, predicted, strict=True) if g != p)
    return Scores(
        name=name,
        n=len(gold),
        accuracy=float(accuracy_score(gold, predicted)),
        macro_f1=float(f1_score(gold, predicted, average="macro", labels=sorted(set(gold)))),
        invalid_rate=predicted.count(INVALID) / len(predicted) if predicted else 0.0,
        latency_ms_per_example=latency_ms,
        top_confusions=[(g, p, n) for (g, p), n in confusions.most_common(5)],
    )


def report(results: list[Scores]) -> str:
    lines = [
        "| Model | Accuracy | Macro-F1 | Invalid outputs | ms / example |",
        "|---|---|---|---|---|",
        *[r.row() for r in results],
    ]
    for r in results:
        if r.top_confusions:
            pairs = "; ".join(f"{g} → {p} ({n})" for g, p, n in r.top_confusions)
            lines.append(f"\n**{r.name}**, most frequent errors: {pairs}")
    return "\n".join(lines)

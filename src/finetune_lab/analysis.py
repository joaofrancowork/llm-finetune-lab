"""Compare two classifiers on the same test set: where they agree, and whether combining helps.

If two ~91% models make *different* mistakes, a simple combination can beat both;
if they make the same mistakes, the remaining errors are about the labels
(ambiguous or overlapping intents), not the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import f1_score


@dataclass
class Comparison:
    a: str
    b: str
    n: int
    accuracy_a: float
    accuracy_b: float
    agreement: float
    accuracy_when_agree: float
    both_wrong: float
    oracle: float
    per_class: pd.DataFrame


def compare(a: pd.DataFrame, b: pd.DataFrame, name_a: str = "a", name_b: str = "b") -> Comparison:
    if len(a) != len(b) or list(a["text"]) != list(b["text"]):
        raise ValueError("prediction files must cover the same test examples in the same order")
    gold = a["intent"]
    pa, pb = a["predicted"], b["predicted"]
    right_a, right_b = pa == gold, pb == gold
    agree = pa == pb
    labels = sorted(gold.unique())
    per_class = pd.DataFrame(
        {
            "intent": labels,
            f"f1_{name_a}": f1_score(gold, pa, labels=labels, average=None),
            f"f1_{name_b}": f1_score(gold, pb, labels=labels, average=None),
        }
    )
    per_class["gap"] = per_class[f"f1_{name_a}"] - per_class[f"f1_{name_b}"]
    return Comparison(
        a=name_a,
        b=name_b,
        n=len(gold),
        accuracy_a=float(right_a.mean()),
        accuracy_b=float(right_b.mean()),
        agreement=float(agree.mean()),
        accuracy_when_agree=float(right_a[agree].mean()) if agree.any() else 0.0,
        both_wrong=float((~right_a & ~right_b).mean()),
        oracle=float((right_a | right_b).mean()),
        per_class=per_class.sort_values("gap"),
    )


def to_markdown(c: Comparison, rows: int = 5) -> str:
    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    def table(df: pd.DataFrame) -> list[str]:
        out = [f"| Intent | F1 {c.a} | F1 {c.b} |", "|---|---|---|"]
        for r in df.itertuples(index=False):
            out.append(f"| {r[0]} | {r[1]:.2f} | {r[2]:.2f} |")
        return out

    lines = [
        f"# Error analysis: {c.a} vs {c.b} ({c.n} test messages)",
        "",
        "| | |",
        "|---|---|",
        f"| Accuracy {c.a} | {pct(c.accuracy_a)} |",
        f"| Accuracy {c.b} | {pct(c.accuracy_b)} |",
        f"| Same prediction | {pct(c.agreement)} |",
        f"| Accuracy when they agree | {pct(c.accuracy_when_agree)} |",
        f"| Both wrong | {pct(c.both_wrong)} |",
        f"| Oracle (either right) | {pct(c.oracle)} |",
        "",
        f"## Intents where {c.b} is better",
        *table(c.per_class.head(rows)),
        "",
        f"## Intents where {c.a} is better",
        *table(c.per_class.tail(rows).iloc[::-1]),
    ]
    return "\n".join(lines)

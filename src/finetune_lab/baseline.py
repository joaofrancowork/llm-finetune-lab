"""The classical baseline an LLM has to justify itself against: TF-IDF + logistic regression."""

from __future__ import annotations

import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from finetune_lab.evaluate import Scores, score


def build() -> Pipeline:
    features = FeatureUnion(
        [
            ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            (
                "chars",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True
                ),
            ),
        ]
    )
    return Pipeline([("features", features), ("clf", LogisticRegression(C=10, max_iter=2000))])


def run(train: pd.DataFrame, test: pd.DataFrame) -> tuple[Scores, list[str]]:
    model = build().fit(train["text"], train["intent"])
    started = time.perf_counter()
    predicted = list(model.predict(test["text"]))
    latency = (time.perf_counter() - started) * 1000 / max(1, len(test))
    scores = score("TF-IDF + logistic regression", list(test["intent"]), predicted, latency)
    return scores, predicted

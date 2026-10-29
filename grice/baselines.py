"""
Baselines: LLM classifier (direct or maxim-structured prompt), action
multiplicity (also reported inverted) and entropy over TF-IDF similarity to
environment objects. All baselines are scored on the same grouped test folds
as GRICE.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             matthews_corrcoef, precision_recall_fscore_support)
from sklearn.model_selection import GroupShuffleSplit

from .symbolic import parse_context


def _score(y_true, y_pred) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary",
                                                  zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "precision": p, "recall": r, "f1": f1}


# =========================================================================
# fold-aware scoring
# =========================================================================
def _folds(df: pd.DataFrame, n_seeds: int, test_size: float = 0.2):
    """Grouped train/test splits, one GroupShuffleSplit per seed, as in pipeline.evaluate."""
    y = df["is_ambiguous"].astype(int).to_numpy()
    groups = df["group_id"].to_numpy()
    idx = np.arange(len(df))
    for seed in range(n_seeds):
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size,
                                        random_state=seed).split(idx, y, groups))
        yield seed, tr, te


def _aggregate(per_seed: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = per_seed[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in per_seed]
        out[k] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


# =========================================================================
# action multiplicity (unsupervised: no threshold to fit)
# =========================================================================
def action_multiplicity_scores(df: pd.DataFrame) -> np.ndarray:
    preds = []
    for _, row in df.iterrows():
        _, entities = parse_context(row["context"])
        text = str(row["instruction"]).lower()
        cands = [o for o in entities if str(o).lower().strip() and str(o).lower().strip() in text]
        preds.append(int(len(cands) > 1))
    return np.asarray(preds)


def action_multiplicity(df: pd.DataFrame, n_seeds: int = 5) -> Dict[str, float]:
    preds = action_multiplicity_scores(df)
    y = df["is_ambiguous"].astype(int).to_numpy()
    per_seed = []
    for _, _tr, te in _folds(df, n_seeds):
        m = _score(y[te], preds[te])
        m["positive_rate"] = float(preds[te].mean())
        per_seed.append(m)
    out = _aggregate(per_seed)
    out["fitted_on_train"] = False
    out["degenerate"] = bool(out["positive_rate"] in (0.0, 1.0))
    return out


# =========================================================================
# entropy over TF-IDF similarity to environment objects
# =========================================================================
def entropy_scores(df: pd.DataFrame) -> np.ndarray:
    scores = []
    for _, row in df.iterrows():
        _, entities = parse_context(row["context"])
        entities = [e for e in entities if str(e).strip()]
        if len(entities) < 2:
            scores.append(0.0)
            continue
        vec = TfidfVectorizer().fit([str(row["instruction"])] + [str(e) for e in entities])
        q = vec.transform([str(row["instruction"])]).toarray()[0]
        M = vec.transform([str(e) for e in entities]).toarray()
        denom = (np.linalg.norm(q) * np.linalg.norm(M, axis=1) + 1e-9)
        sims = np.clip(M @ q / denom, 0, None)
        if sims.sum() <= 0:
            scores.append(0.0)
            continue
        p = sims / sims.sum()
        scores.append(float(shannon_entropy(p) / np.log(len(entities))))
    return np.asarray(scores)


def _best_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Threshold maximising F1 on the training fold; thresholds that yield a
    single predicted class are skipped."""
    candidates = np.unique(scores)
    if len(candidates) > 200:
        candidates = np.quantile(scores, np.linspace(0, 1, 200))
    best, best_f1 = float(candidates[0]), -1.0
    for thr in candidates:
        preds = (scores > thr).astype(int)
        if preds.min() == preds.max():
            continue                      # degenerate, skip
        _, _, f1, _ = precision_recall_fscore_support(y, preds, average="binary",
                                                      zero_division=0)
        if f1 > best_f1:
            best, best_f1 = float(thr), float(f1)
    return best


def entropy_baseline(df: pd.DataFrame, n_seeds: int = 5,
                     threshold: Optional[float] = None) -> Dict[str, float]:
    """Entropy baseline. Unless `threshold` is given, the threshold is fitted on
    each training fold and applied to the test fold."""
    scores = entropy_scores(df)
    y = df["is_ambiguous"].astype(int).to_numpy()
    per_seed = []
    for _, tr, te in _folds(df, n_seeds):
        thr = _best_threshold(scores[tr], y[tr]) if threshold is None else threshold
        preds = (scores[te] > thr).astype(int)
        m = _score(y[te], preds)
        m["threshold"] = thr
        m["positive_rate"] = float(preds.mean())
        per_seed.append(m)
    out = _aggregate(per_seed)
    out["fitted_on_train"] = threshold is None
    out["degenerate"] = bool(out["positive_rate"] in (0.0, 1.0))
    out["score_zero_fraction"] = float((scores == 0).mean())
    return out


# =========================================================================
# LLM classifier (zero-shot: nothing is fitted)
# =========================================================================
def llm_classifier_preds(df: pd.DataFrame, backend, structured: bool = False):
    from .prompts import DIRECT_SYSTEM, direct_prompt, structured_prompt
    preds, n_fail = [], 0
    for _, row in df.iterrows():
        ctx, _ = parse_context(row["context"])
        prompt = (structured_prompt(row["instruction"], ctx) if structured
                  else direct_prompt(row["instruction"], ctx))
        val = backend.yes_no(prompt, DIRECT_SYSTEM)
        if np.isnan(val):
            n_fail += 1
            preds.append(0)          # parse failure counted as NO
        else:
            preds.append(int(val))
    return np.asarray(preds), n_fail


def llm_classifier(df: pd.DataFrame, backend, structured: bool = False,
                   n_seeds: int = 5) -> Dict[str, float]:
    """Zero-shot LLM classifier, scored on the test folds."""
    preds, n_fail = llm_classifier_preds(df, backend, structured)
    y = df["is_ambiguous"].astype(int).to_numpy()
    per_seed = []
    for _, _tr, te in _folds(df, n_seeds):
        m = _score(y[te], preds[te])
        m["positive_rate"] = float(preds[te].mean())
        per_seed.append(m)
    out = _aggregate(per_seed)
    out["parse_failure_rate"] = n_fail / max(len(df), 1)
    out["fitted_on_train"] = False
    # degenerate: the same answer for every test item
    out["degenerate"] = bool(out["positive_rate"] in (0.0, 1.0))
    out["full_dataset_positive_rate"] = float(preds.mean())
    return out


# =========================================================================
# inverted scoring
# =========================================================================
def inverted_score(df: pd.DataFrame, preds: np.ndarray, n_seeds: int = 5) -> Dict[str, float]:
    """Score a baseline with its predictions flipped (used when its accuracy is
    below 0.5)."""
    y = df["is_ambiguous"].astype(int).to_numpy()
    per_seed = [_score(y[te], 1 - preds[te]) for _, _tr, te in _folds(df, n_seeds)]
    return _aggregate(per_seed)

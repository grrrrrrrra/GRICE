"""
Fusion, classification and evaluation.

Fused score per maxim:
    v_hat_m = lambda_m * v_sym_m + (1 - lambda_m) * v_llm_m

The GRICE classifier uses exactly the four fused scores
(v_hat_quantity, v_hat_quality, v_hat_relation, v_hat_manner).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MAXIMS = ("quantity", "quality", "relation", "manner")
FUSED = [f"v_hat_{m}" for m in MAXIMS]
SYMBOLIC = [f"v_sym_{m}" for m in MAXIMS]
LLM = [f"v_llm_{m}" for m in MAXIMS]


# =========================================================================
# fusion
# =========================================================================
def fuse(df: pd.DataFrame, lambdas: Dict[str, float] | float = 0.5,
         llm_nan_policy: str = "symbolic") -> pd.DataFrame:
    """Add the four v_hat columns.

    llm_nan_policy:
        'symbolic'  where the LLM score is NaN, use the symbolic score
        'drop'      keep NaN
    The number of NaN LLM cells is stored in out.attrs["n_llm_fallback_cells"].
    """
    if isinstance(lambdas, (int, float)):
        lambdas = {m: float(lambdas) for m in MAXIMS}
    out = df.copy()
    n_fallback = 0
    for m in MAXIMS:
        lam = float(lambdas[m])
        sym = out[f"v_sym_{m}"].astype(float)
        llm = out[f"v_llm_{m}"].astype(float)
        fused = lam * sym + (1.0 - lam) * llm
        bad = llm.isna()
        n_fallback += int(bad.sum())
        if llm_nan_policy == "symbolic":
            fused = fused.where(~bad, sym)
        out[f"v_hat_{m}"] = fused
    out.attrs["n_llm_fallback_cells"] = n_fallback
    return out


def make_model(kind: str = "logreg", random_state: int = 42):
    if kind == "logreg":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000,
                                                    random_state=random_state))])
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=random_state)
    raise ValueError(f"unknown model: {kind}")


def coefficients(model, feature_cols: Sequence[str]) -> Dict[str, float]:
    """Logistic-regression coefficients or random-forest feature importances."""
    if isinstance(model, Pipeline):
        clf = model.named_steps["clf"]
        return {c: float(w) for c, w in zip(feature_cols, clf.coef_[0])}
    if hasattr(model, "feature_importances_"):
        return {c: float(w) for c, w in zip(feature_cols, model.feature_importances_)}
    return {}


# =========================================================================
# evaluation
# =========================================================================
def _metrics(y_true, y_pred, y_score=None) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary",
                                                  zero_division=0)
    out = {"accuracy": accuracy_score(y_true, y_pred),
           "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
           "mcc": matthews_corrcoef(y_true, y_pred),
           "precision": p, "recall": r, "f1": f1,
           "positive_rate": float(np.mean(y_pred))}
    if y_score is not None and len(set(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    return out


def tune_threshold(y_true: np.ndarray, proba: np.ndarray,
                   objective: str = "mcc") -> float:
    """Threshold that maximises `objective` (MCC or balanced accuracy) on the
    training-fold probabilities."""
    grid = np.unique(np.quantile(proba, np.linspace(0.02, 0.98, 97)))
    best, best_score = 0.5, -2.0
    for thr in grid:
        pred = (proba >= thr).astype(int)
        if pred.min() == pred.max():
            continue
        score = (matthews_corrcoef(y_true, pred) if objective == "mcc"
                 else balanced_accuracy_score(y_true, pred))
        if score > best_score:
            best, best_score = float(thr), float(score)
    return best


def evaluate(df: pd.DataFrame, feature_cols: Sequence[str] = FUSED,
             target: str = "is_ambiguous", group_col: str = "group_id",
             model_kind: str = "logreg", n_seeds: int = 5,
             test_size: float = 0.2, tune_thr: bool = True,
             thr_objective: str = "mcc") -> Dict:
    """Repeated grouped hold-out (GroupShuffleSplit); returns mean and std over seeds."""
    feature_cols = list(feature_cols)
    if set(feature_cols) == set(FUSED) and len(feature_cols) != 4:
        raise AssertionError("the GRICE feature vector must be exactly 4 columns")

    sub = df.dropna(subset=feature_cols + [target]).reset_index(drop=True)
    X = sub[feature_cols].to_numpy(dtype=float)
    y = sub[target].astype(int).to_numpy()
    groups = sub[group_col].to_numpy()

    per_seed, coefs, cms, thrs = [], [], [], []
    for seed in range(n_seeds):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(splitter.split(X, y, groups))
        model = make_model(model_kind, random_state=seed)
        model.fit(X[tr], y[tr])
        has_proba = hasattr(model, "predict_proba")
        score = model.predict_proba(X[te])[:, 1] if has_proba else None
        if tune_thr and has_proba:
            thr = tune_threshold(y[tr], model.predict_proba(X[tr])[:, 1], thr_objective)
            pred = (score >= thr).astype(int)
        else:
            thr = 0.5
            pred = model.predict(X[te])
        thrs.append(thr)
        per_seed.append(_metrics(y[te], pred, score))
        coefs.append(coefficients(model, feature_cols))
        cms.append(confusion_matrix(y[te], pred, labels=[0, 1]))

    keys = per_seed[0].keys()
    summary = {k: {"mean": float(np.mean([m[k] for m in per_seed if k in m])),
                   "std": float(np.std([m[k] for m in per_seed if k in m]))}
               for k in keys}
    mean_coefs = ({k: float(np.mean([c[k] for c in coefs])) for k in coefs[0]}
                  if coefs and coefs[0] else {})
    return {
        "n": len(sub),
        "n_groups": int(pd.Series(groups).nunique()),
        "threshold_tuned": bool(tune_thr),
        "threshold_mean": float(np.mean(thrs)),
        "threshold_std": float(np.std(thrs)),
        "features": feature_cols,
        "model": model_kind,
        "n_seeds": n_seeds,
        "summary": summary,
        "per_seed": per_seed,
        "coefficients": mean_coefs,
        "confusion_mean": np.mean(cms, axis=0).round(1).tolist(),
    }


# =========================================================================
# ablations
# =========================================================================
def ablations(df: pd.DataFrame, model_kind: str = "logreg", n_seeds: int = 5) -> pd.DataFrame:
    rows = []

    def add(name: str, cols: Sequence[str]):
        res = evaluate(df, cols, model_kind=model_kind, n_seeds=n_seeds)
        rows.append({"setting": name,
                     "n_features": len(cols),
                     "acc": res["summary"]["accuracy"]["mean"],
                     "acc_std": res["summary"]["accuracy"]["std"],
                     "mcc": res["summary"]["mcc"]["mean"],
                     "mcc_std": res["summary"]["mcc"]["std"],
                     "f1": res["summary"]["f1"]["mean"],
                     "f1_std": res["summary"]["f1"]["std"]})

    add("full (4 fused maxims)", FUSED)
    for m in MAXIMS:
        add(f"leave-one-out: -{m}", [c for c in FUSED if not c.endswith(m)])
    for m in MAXIMS:
        add(f"single maxim: {m}", [f"v_hat_{m}"])
    add("symbolic only", SYMBOLIC)
    add("LLM only", LLM)
    add("all 8 raw (learned lambda)", SYMBOLIC + LLM)
    return pd.DataFrame(rows)


def lambda_sweep(df: pd.DataFrame, values: Iterable[float] = (0.0, 0.3, 0.5, 0.7, 1.0),
                 model_kind: str = "logreg", n_seeds: int = 5) -> pd.DataFrame:
    rows = []
    for lam in values:
        fused = fuse(df, lam)
        res = evaluate(fused, FUSED, model_kind=model_kind, n_seeds=n_seeds)
        sm = res["summary"]
        rows.append({"lambda": lam,
                     "acc": sm["accuracy"]["mean"], "acc_std": sm["accuracy"]["std"],
                     "mcc": sm["mcc"]["mean"], "mcc_std": sm["mcc"]["std"],
                     "auc": sm.get("roc_auc", {}).get("mean", float("nan")),
                     "f1": sm["f1"]["mean"], "f1_std": sm["f1"]["std"]})
    return pd.DataFrame(rows)


def per_type_recall(df: pd.DataFrame, feature_cols: Sequence[str] = FUSED,
                    model_kind: str = "logreg", n_seeds: int = 5,
                    test_size: float = 0.2) -> pd.DataFrame:
    """Accuracy per ambiguity_type, pooled over the test folds."""
    sub = df.dropna(subset=list(feature_cols) + ["is_ambiguous"]).reset_index(drop=True)
    X = sub[list(feature_cols)].to_numpy(dtype=float)
    y = sub["is_ambiguous"].astype(int).to_numpy()
    groups = sub["group_id"].to_numpy()
    types = sub.get("ambiguity_type", pd.Series(["unknown"] * len(sub))).astype(str).to_numpy()

    hits: Dict[str, List[int]] = {}
    for seed in range(n_seeds):
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size,
                                        random_state=seed).split(X, y, groups))
        model = make_model(model_kind, random_state=seed)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        for i, t in zip(te, pred == y[te]):
            hits.setdefault(types[i], []).append(int(t))
    rows = [{"ambiguity_type": t, "n_test_predictions": len(v),
             "accuracy": float(np.mean(v))} for t, v in sorted(hits.items())]
    return pd.DataFrame(rows)

"""
Dataset loading.

Both loaders return a table with the columns
    uid, group_id, dataset, instruction, context, is_ambiguous, ambiguity_type, user_intent

In AmbiK, the ambiguous task and its unambiguous counterpart come from the same
source row and share an environment, so they get the same `group_id`. All
train/test splits are grouped by `group_id`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd


UNIFIED_COLS = ["uid", "group_id", "dataset", "instruction", "context",
                "is_ambiguous", "ambiguity_type", "user_intent"]


# accepted names of the AmbiK context column, in order of preference
AMBIK_CONTEXT_CANDIDATES = ("environment_short", "environment_full", "environment")

# default IntroPlan ambiguity_type values treated as unambiguous
INTROPLAN_UNAMBIGUOUS_TYPES = {"unambiguous", "unambiguous_direct", "unambiguous_task",
                               "none", "no_ambiguity", "clear"}


def _pick_context_col(raw: pd.DataFrame, requested: Optional[str]) -> str:
    if requested:
        if requested not in raw.columns:
            raise ValueError(f"context column '{requested}' not in file; "
                             f"available: {list(raw.columns)}")
        return requested
    for c in AMBIK_CONTEXT_CANDIDATES:
        if c in raw.columns:
            return c
    raise ValueError(f"no context column found; available: {list(raw.columns)}")


# AmbiK types where the robot is expected to ask for help; common_sense_knowledge
# tasks are treated as resolvable without asking
AMBIK_HELP_EXPECTED = ("preferences", "safety")


def load_ambik(path: str, n_per_class: Optional[int] = None,
               random_state: int = 42, context_col: Optional[str] = None,
               keep_intent: bool = True,
               positive_types: Optional[Sequence[str]] = None,
               other_positive_policy: str = "drop") -> pd.DataFrame:
    """Load AmbiK as ambiguous_task (1) vs unambiguous_direct (0).

    Both instructions of a source row share a group_id. n_per_class source rows
    are sampled at random. `user_intent` is kept in the output and is not a
    model input.

    positive_types: restrict the ambiguous class to these `ambiguity_type`
    values (e.g. AMBIK_HELP_EXPECTED). Ambiguous rows of other types are
    handled by `other_positive_policy`:

        'drop'      exclude them
        'negative'  label them 0
        'keep'      label them 1 (positive_types is ignored)

    With positive_types=None, every ambiguous_task is 1 and every
    unambiguous_direct is 0.
    """
    raw = pd.read_csv(path)
    ctx = _pick_context_col(raw, context_col)
    for col in ("ambiguous_task", "unambiguous_direct"):
        if col not in raw.columns:
            raise ValueError(f"AmbiK file is missing column '{col}'; "
                             f"available: {list(raw.columns)}")
    raw = raw.dropna(subset=["ambiguous_task", "unambiguous_direct"]).reset_index(drop=True)
    raw["group_id"] = [f"ambik-g{i:05d}" for i in range(len(raw))]

    if n_per_class is not None and n_per_class < len(raw):
        raw = raw.sample(n=n_per_class, random_state=random_state).reset_index(drop=True)

    intent = (raw["user_intent"] if keep_intent and "user_intent" in raw.columns
              else pd.Series([""] * len(raw)))

    amb_label = pd.Series(1, index=raw.index)
    amb_drop = pd.Series(False, index=raw.index)
    if positive_types is not None and other_positive_policy != "keep":
        types = raw.get("ambiguity_type", pd.Series([""] * len(raw))).astype(str)
        is_pos = types.isin(set(positive_types))
        if other_positive_policy == "drop":
            amb_drop = ~is_pos
        elif other_positive_policy == "negative":
            amb_label = is_pos.astype(int)
        else:
            raise ValueError(f"unknown other_positive_policy: {other_positive_policy}")
        print(f"[ambik] positive types: {sorted(set(positive_types))}")
        print(f"[ambik] ambiguous rows kept as positive: {int(is_pos.sum())} / {len(raw)}"
              f"  (others -> {other_positive_policy})")

    amb = pd.DataFrame({
        "group_id": raw["group_id"], "instruction": raw["ambiguous_task"].astype(str),
        "context": raw[ctx], "is_ambiguous": amb_label, "user_intent": intent,
        "ambiguity_type": raw.get("ambiguity_type", pd.Series(["unknown"] * len(raw))),
    })
    unamb = pd.DataFrame({
        "group_id": raw["group_id"], "instruction": raw["unambiguous_direct"].astype(str),
        "context": raw[ctx], "is_ambiguous": 0, "user_intent": intent,
        "ambiguity_type": "unambiguous_direct",
    })
    amb = amb[~amb_drop.to_numpy()]
    df = pd.concat([amb, unamb], ignore_index=True)
    df["dataset"] = "ambik"
    df["uid"] = [f"ambik-{i:05d}" for i in range(len(df))]
    return df[UNIFIED_COLS]


def load_introplan(path: str, instruction_col: str = "task",
                   context_col: str = "environment_full",
                   label_col: Optional[str] = None,
                   type_col: str = "ambiguity_type",
                   unambiguous_types: Optional[set] = None,
                   group_col: Optional[str] = None) -> pd.DataFrame:
    """Load IntroPlan.

    If `label_col` is not given, the label is derived from `ambiguity_type`: a
    row is unambiguous if its type is in `unambiguous_types`. The observed types
    and the resulting class balance are printed.
    """
    raw = pd.read_csv(path)
    missing = [c for c in (instruction_col, context_col) if c not in raw.columns]
    if missing:
        raise ValueError(f"IntroPlan file is missing columns: {missing}; "
                         f"available: {list(raw.columns)}")

    if label_col and label_col in raw.columns:
        labels = raw[label_col].astype(int)
    else:
        if type_col not in raw.columns:
            raise ValueError("need either an explicit label column or "
                             f"'{type_col}' to derive one")
        unamb_set = {t.lower() for t in (unambiguous_types or INTROPLAN_UNAMBIGUOUS_TYPES)}
        types = raw[type_col].astype(str).str.strip().str.lower()
        unknown = sorted(set(types))
        labels = (~types.isin(unamb_set)).astype(int)
        print(f"[introplan] types found: {unknown}")
        print(f"[introplan] treated as UNAMBIGUOUS: "
              f"{sorted(set(types) & unamb_set)}")
        print(f"[introplan] -> {int(labels.sum())} ambiguous / "
              f"{int((labels == 0).sum())} unambiguous")
        if labels.nunique() < 2:
            raise ValueError("label derivation produced a single class — "
                             "pass --unambiguous-types explicitly")

    df = pd.DataFrame({
        "instruction": raw[instruction_col].astype(str),
        "context": raw[context_col],
        "is_ambiguous": labels,
        "ambiguity_type": raw.get(type_col, pd.Series(["unknown"] * len(raw))),
        "user_intent": raw.get("user_intent", pd.Series([""] * len(raw))),
    })
    df["group_id"] = (raw[group_col].astype(str) if group_col and group_col in raw.columns
                      else [f"introplan-g{i:05d}" for i in range(len(raw))])
    df["dataset"] = "introplan"
    df["uid"] = [f"introplan-{i:05d}" for i in range(len(df))]
    return df[UNIFIED_COLS]


def describe(df: pd.DataFrame) -> str:
    lines = [
        f"rows            : {len(df)}",
        f"groups          : {df['group_id'].nunique()}",
        f"ambiguous       : {int(df['is_ambiguous'].sum())}",
        f"unambiguous     : {int((df['is_ambiguous'] == 0).sum())}",
        f"mean len (amb)  : {df.loc[df.is_ambiguous == 1, 'instruction'].str.split().str.len().mean():.1f} tokens",
        f"mean len (unamb): {df.loc[df.is_ambiguous == 0, 'instruction'].str.split().str.len().mean():.1f} tokens",
    ]
    if "ambiguity_type" in df:
        counts = df["ambiguity_type"].value_counts().to_dict()
        lines.append(f"types           : {counts}")
    return "\n".join(lines)

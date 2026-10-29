#!/usr/bin/env python
"""Stage 1 — build the unified dataset and compute symbolic maxim scores.

    python scripts/01_symbolic.py --dataset ambik --input data/ambik_test_900.csv \
        --context-col environment_short --feature-variant v2 \
        --output out/ambik_symbolic_v2.csv
"""
import argparse
import json
import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grice import data as gdata                      # noqa: E402
from grice.symbolic import RelevanceScorer, symbolic_scores, parse_context  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ambik", "introplan"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-per-class", type=int, default=None)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--feature-variant", choices=["paper", "v2"], default="paper",
                    help="v2 = features of Table I (used for all reported results); "
                         "paper = earlier set with stopword share, type-token ratio "
                         "and absolute parse depth")
    ap.add_argument("--st-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--st-device", default="cpu",
                    help="device for the sentence embedder; cpu is the default and "
                         "is fast enough for this stage")
    ap.add_argument("--context-col", default=None,
                    help="AmbiK: environment_short | environment_full (auto-detected)")
    ap.add_argument("--instruction-col", default="task", help="IntroPlan only")
    ap.add_argument("--label-col", default=None,
                    help="IntroPlan: explicit 0/1 column; omit to derive from ambiguity_type")
    ap.add_argument("--unambiguous-types", nargs="*", default=None,
                    help="IntroPlan: ambiguity_type values that mean 'clear'")
    ap.add_argument("--positive-types", nargs="*", default=None,
                    help="AmbiK: restrict the ambiguous class to these types "
                         "(e.g. preferences safety). Omit for construction-based labels.")
    ap.add_argument("--other-positive-policy", choices=["drop", "negative", "keep"],
                    default="drop", help="AmbiK: what to do with ambiguous rows "
                                         "whose type is not in --positive-types")
    args = ap.parse_args()

    if args.dataset == "ambik":
        df = gdata.load_ambik(args.input, n_per_class=args.n_per_class,
                              random_state=args.random_state,
                              context_col=args.context_col,
                              positive_types=args.positive_types,
                              other_positive_policy=args.other_positive_policy)
    else:
        df = gdata.load_introplan(
            args.input, instruction_col=args.instruction_col,
            context_col=args.context_col or "environment_full",
            label_col=args.label_col,
            unambiguous_types=set(args.unambiguous_types) if args.unambiguous_types else None)

    print(gdata.describe(df))

    scorer = RelevanceScorer(args.st_model, device=args.st_device)
    if scorer.backend == "tfidf-fallback":
        print("\n!! sentence-transformers unavailable — RELATION uses TF-IDF cosine.\n"
              "   This is a different measurement. Do not mix runs.\n")
        scorer.fit_fallback(df["instruction"].astype(str).tolist()
                            + [parse_context(c)[0] for c in df["context"]])

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="symbolic"):
        rows.append(symbolic_scores(row["instruction"], row["context"], scorer,
                                    variant=args.feature_variant))
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)

    manifest = {
        "stage": "symbolic",
        "dataset": args.dataset,
        "input": os.path.abspath(args.input),
        "output": os.path.abspath(args.output),
        "n_rows": len(out),
        "n_groups": int(out["group_id"].nunique()),
        "n_per_class": args.n_per_class,
        "positive_types": args.positive_types,
        "other_positive_policy": args.other_positive_policy,
        "unambiguous_types": args.unambiguous_types,
        "random_state": args.random_state,
        "feature_variant": args.feature_variant,
        "relevance_backend": scorer.backend,
        "relevance_device": args.st_device,
    }
    with open(args.output.replace(".csv", ".manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

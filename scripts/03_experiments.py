#!/usr/bin/env python
"""Stage 3 — evaluation of GRICE and the baselines.

    python scripts/03_experiments.py --input out/ambik_features_qwen3.csv \
        --outdir out/exp_ambik_qwen3 --lambda 0.7 --exclude-groups out/dev_groups.txt

Writes to --outdir:
    main.json           GRICE: mean and std over seeds, coefficients, confusion matrix
    baselines.json      action multiplicity (and its inverted form) and entropy
    ablations.csv       leave-one-maxim-out, single maxim, symbolic/LLM only, all 8 raw scores
    lambda_sweep.csv    results for lambda in (0, 0.3, 0.5, 0.7, 1)
    per_type.csv        accuracy by ambiguity type
    coefficients.json   learned weight per maxim
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grice import baselines, pipeline                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.7)
    ap.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--no-threshold-tuning", action="store_true",
                    help="keep the 0.5 default instead of fitting the threshold "
                         "on each training fold")
    ap.add_argument("--exclude-groups", default=None,
                    help="text file with one group_id per line; these development "
                         "groups are removed before evaluation")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input)
    if args.exclude_groups:
        with open(args.exclude_groups) as fh:
            dev = {ln.strip() for ln in fh if ln.strip()}
        before = len(df)
        df = df[~df["group_id"].astype(str).isin(dev)].reset_index(drop=True)
        print(f"excluded {before - len(df)} rows from {len(dev)} development groups; "
              f"{len(df)} rows remain")
    fused = pipeline.fuse(df, args.lam)
    n_fb = fused.attrs.get("n_llm_fallback_cells", 0)
    if n_fb:
        print(f"note: {n_fb} LLM cells were NaN and fell back to the symbolic score")

    # --- 1. main result ---------------------------------------------------
    main_res = pipeline.evaluate(fused, pipeline.FUSED, model_kind=args.model,
                                 n_seeds=args.n_seeds,
                                 tune_thr=not args.no_threshold_tuning)
    main_res["lambda"] = args.lam
    main_res["n_llm_fallback_cells"] = n_fb
    with open(f"{args.outdir}/main.json", "w") as fh:
        json.dump(main_res, fh, indent=2)
    print("\n=== GRICE ===")
    print(json.dumps({k: main_res[k] for k in ("n", "n_groups", "summary",
                                               "coefficients", "confusion_mean")},
                     indent=2))

    # --- 2. symbolic baselines -------------------------------------------
    base = {"action_multiplicity": baselines.action_multiplicity(df, args.n_seeds),
            "entropy": baselines.entropy_baseline(df, args.n_seeds)}
    if base["action_multiplicity"]["accuracy"] < 0.5:
        base["action_multiplicity_INVERTED"] = baselines.inverted_score(
            df, baselines.action_multiplicity_scores(df), args.n_seeds)
        print("\nnote: action multiplicity accuracy is below 0.5; "
              "the inverted score is also reported")
    with open(f"{args.outdir}/baselines.json", "w") as fh:
        json.dump(base, fh, indent=2)
    print("\n=== symbolic baselines ===")
    print(json.dumps(base, indent=2))

    # --- 3. ablations, lambda sweep, per-type accuracy, coefficients ------
    pipeline.ablations(fused, args.model, args.n_seeds).to_csv(
        f"{args.outdir}/ablations.csv", index=False)
    pipeline.lambda_sweep(df, model_kind=args.model, n_seeds=args.n_seeds).to_csv(
        f"{args.outdir}/lambda_sweep.csv", index=False)
    ptr = pipeline.per_type_recall(fused, model_kind=args.model, n_seeds=args.n_seeds)
    ptr.to_csv(f"{args.outdir}/per_type.csv", index=False)
    print("\n=== accuracy by ambiguity type ===")
    print(ptr.round(3).to_string(index=False))
    with open(f"{args.outdir}/coefficients.json", "w") as fh:
        json.dump(main_res["coefficients"], fh, indent=2)

    print(f"\nwritten to {args.outdir}/")


if __name__ == "__main__":
    main()

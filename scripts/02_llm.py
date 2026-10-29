#!/usr/bin/env python
"""Stage 2 — per-maxim LLM assessment (one YES/NO call per maxim).

    python scripts/02_llm.py --input out/ambik_symbolic_v2.csv \
        --output out/ambik_features_qwen3.csv \
        --backend hf --model Qwen/Qwen3-8B --score-mode both --prompt-version mix

Exits with an error if the parse-failure rate exceeds --max-parse-failure
(default 5%).
"""
import argparse
import json
import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grice.llm import assess_maxims, build_backend    # noqa: E402
from grice.prompts import resolve_mix                 # noqa: E402
from grice.symbolic import parse_context              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--backend", choices=["hf", "openai", "mock"], default="hf")
    ap.add_argument("--model", required=True)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--score-mode", choices=["binary", "prob", "both"], default="both",
                    help="binary = parsed YES/NO answer; prob = P(YES) from the "
                         "answer-token distribution; both = compute each")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Qwen3 only; off by default (thinking blocks break short answers)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--prompt-version", default="v1",
                    help="v1 = maxim definition and a violation example; v2 = adds a "
                         "non-violation example; v3 = operational question that does "
                         "not name the maxim; 'mix' = v1 for quantity, quality and "
                         "relation and v3 for manner (used in the paper); or an "
                         "explicit spec such as 'quantity=v1,manner=v3'")
    ap.add_argument("--cache", default=None, help="sqlite cache path (recommended)")
    ap.add_argument("--max-parse-failure", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit).copy()

    cache = args.cache or args.output.replace(".csv", ".cache.sqlite")
    backend = build_backend({"backend": args.backend, "model_name": args.model,
                             "cache_path": cache, "load_in_4bit": args.load_in_4bit,
                             "dtype": args.dtype,
                             "enable_thinking": args.enable_thinking})

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"llm:{args.model}"):
        ctx, _ = parse_context(row["context"])
        rows.append(assess_maxims(backend, str(row["instruction"]), ctx,
                                  score_mode=args.score_mode,
                                  prompt_version=args.prompt_version))

    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)

    stats = backend.stats.report()
    per_maxim_nan = {c: float(out[c].isna().mean())
                     for c in out.columns if c.startswith("v_llm_")}
    yes_rate = {c: float(out[c].mean(skipna=True))
                for c in out.columns if c.startswith("v_llm_")}
    manifest = {"stage": "llm", "model": args.model, "backend": args.backend,
                "score_mode": args.score_mode,
                "prompt_version": args.prompt_version,
                "prompt_mix": resolve_mix(args.prompt_version),
                "input": os.path.abspath(args.input),
                "output": os.path.abspath(args.output),
                "n_rows": len(out), **stats,
                "nan_rate_per_maxim": per_maxim_nan,
                "yes_rate_per_maxim": yes_rate,
                "sample_unparsed": backend.stats.failures[:10]}
    with open(args.output.replace(".csv", ".manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(manifest, indent=2))

    if stats["parse_failure_rate"] > args.max_parse_failure:
        raise SystemExit(
            f"\nABORT: parse failure rate {stats['parse_failure_rate']:.1%} exceeds "
            f"{args.max_parse_failure:.1%}. The LLM features are not usable — fix the "
            f"prompt or the decoding before running experiments on this file.")
    for col, rate in yes_rate.items():
        if rate in (0.0, 1.0):
            print(f"\nWARNING: {col} is constant ({rate}). This maxim carries no signal "
                  f"and every backbone using it will produce identical results.")


if __name__ == "__main__":
    main()

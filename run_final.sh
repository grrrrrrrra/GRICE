#!/usr/bin/env bash
# All experiments of the paper on AmbiK and IntroPlan with Qwen3-8B, Mistral-7B
# and gpt-5-mini: GRICE, baselines, ablations, lambda sweep, per-type accuracy
# and coefficients. gpt-5-mini runs only when OPENAI_API_KEY is set.
#
#   bash run_final.sh 2>&1 | tee out/run_final.log
#
# Stages 1-2 are skipped when their output files exist, and LLM answers are
# cached in sqlite files. Approximate runtime on 2x V100: AmbiK ~2 h and
# IntroPlan ~35 min per local backbone.
set -euo pipefail

AMBIK=${AMBIK:-data/ambik_test_900.csv}
INTROPLAN=${INTROPLAN:-data/introplan_mobile.csv}
PROMPTS=${PROMPTS:-mix}
LAMBDA=${LAMBDA:-0.7}
SEEDS=${SEEDS:-5}
DEV=out/dev_groups.txt

mkdir -p out

# -------------------------------------------------------------------------
# 1. symbolic features (CPU, ~15 min for AmbiK, ~3 min for IntroPlan)
# -------------------------------------------------------------------------
if [ ! -f out/ambik_symbolic_v2.csv ]; then
  python scripts/01_symbolic.py --dataset ambik \
    --input "$AMBIK" --context-col environment_short \
    --feature-variant v2 --output out/ambik_symbolic_v2.csv
fi

if [ ! -f out/introplan_symbolic_v2.csv ]; then
  python scripts/01_symbolic.py --dataset introplan \
    --input "$INTROPLAN" --context-col environment_short \
    --instruction-col task \
    --unambiguous-types singlelabel_task unambiguous_task creative_singlelabel_task \
    --feature-variant v2 --output out/introplan_symbolic_v2.csv
fi

# -------------------------------------------------------------------------
# 2. per-maxim LLM assessment
# -------------------------------------------------------------------------
run_llm () {           # $1 dataset  $2 tag  $3 backend  $4 model
  local out="out/$1_features_$2.csv"
  if [ -f "$out" ]; then
    echo "skip $out (exists)"
    return
  fi
  echo "=== LLM: $1 / $2 ==="
  python scripts/02_llm.py \
    --input "out/$1_symbolic_v2.csv" --output "$out" \
    --backend "$3" --model "$4" --score-mode both \
    --prompt-version "$PROMPTS"
}

run_llm ambik     qwen3    hf     Qwen/Qwen3-8B
run_llm introplan qwen3    hf     Qwen/Qwen3-8B
run_llm ambik     mistral  hf     mistralai/Mistral-7B-Instruct-v0.3
run_llm introplan mistral  hf     mistralai/Mistral-7B-Instruct-v0.3
if [ -n "${OPENAI_API_KEY:-}" ]; then
  run_llm ambik     gpt5mini openai gpt-5-mini
  run_llm introplan gpt5mini openai gpt-5-mini
fi

# -------------------------------------------------------------------------
# 3. GRICE, symbolic baselines, ablations, lambda sweep, per-type accuracy
#    --exclude-groups keeps prompt selection and reported results apart
# -------------------------------------------------------------------------
EXCLUDE=""
[ -f "$DEV" ] && EXCLUDE="--exclude-groups $DEV"

for ds in ambik introplan; do
  for tag in qwen3 mistral gpt5mini; do
    f="out/${ds}_features_${tag}.csv"
    [ -f "$f" ] || continue
    echo "=== experiments: $ds / $tag ==="
    # dev groups are AmbiK group ids and do not affect IntroPlan
    python scripts/03_experiments.py --input "$f" \
      --outdir "out/exp_${ds}_${tag}" \
      --lambda "$LAMBDA" --n-seeds "$SEEDS" $EXCLUDE
  done
done

# -------------------------------------------------------------------------
# 4. LLM-classifier baseline (direct and maxim-structured prompt)
# -------------------------------------------------------------------------
python - <<'EOF'
import json, os, sys
import pandas as pd
sys.path.insert(0, ".")
from grice.baselines import llm_classifier
from grice.llm import build_backend

MODELS = {"qwen3": ("hf", "Qwen/Qwen3-8B"),
          "mistral": ("hf", "mistralai/Mistral-7B-Instruct-v0.3")}
if os.environ.get("OPENAI_API_KEY"):
    MODELS["gpt5mini"] = ("openai", "gpt-5-mini")
res = {}
for tag, (backend, model) in MODELS.items():
    files = [f"out/{ds}_features_{tag}.csv" for ds in ("ambik", "introplan")]
    if not any(os.path.exists(f) for f in files):
        continue
    bk = build_backend({"backend": backend, "model_name": model,
                        "cache_path": f"out/direct_{tag}.sqlite"})
    for f in files:
        if not os.path.exists(f):
            continue
        ds = os.path.basename(f).split("_")[0]
        d = pd.read_csv(f)
        for structured in (False, True):
            r = llm_classifier(d, bk, structured=structured, n_seeds=5)
            key = f"{ds}_{tag}_{'structured' if structured else 'direct'}"
            res[key] = r
            print(f"{key:32} acc={r['accuracy']:.3f} mcc={r['mcc']:.3f} "
                  f"f1={r['f1']:.3f} yes={r['positive_rate']:.3f}"
                  + (" | DEGENERATE" if r["degenerate"] else ""))
    del bk
json.dump(res, open("out/llm_classifier_baseline.json", "w"), indent=2)
EOF

echo
echo "done. results in out/exp_*/ and out/llm_classifier_baseline.json"

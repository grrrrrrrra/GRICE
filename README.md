# GRICE: Grice's Robotic Instruction Clarity Evaluation

Code for **GRICE**, a method for detecting ambiguous instructions given to robots through violations of Grice's four conversational maxims: Quantity, Quality, Relation and Manner. Evaluated on AmbiK and IntroPlan.

For each instruction, GRICE computes a violation score per maxim in two ways: symbolic features of the instruction and the environment, and a separate YES/NO question to an LLM for each maxim. The two scores are fused with a weight λ:

```
v_hat_m = λ · v_sym_m + (1 − λ) · v_llm_m
```

A logistic regression on the four fused scores decides whether the instruction is ambiguous.

## Repository structure

```
.
├── data/                   # evaluation data (see "Data")
├── grice/
│   ├── data.py             # dataset loading, labels and group ids
│   ├── symbolic.py         # symbolic maxim scores
│   ├── prompts.py          # per-maxim prompts and baseline prompts
│   ├── llm.py              # LLM backends: Hugging Face, OpenAI, mock
│   ├── pipeline.py         # fusion, classifier, evaluation, ablations
│   └── baselines.py        # baselines
├── scripts/
│   ├── 01_symbolic.py      # stage 1: symbolic scores
│   ├── 02_llm.py           # stage 2: LLM scores
│   └── 03_experiments.py   # stage 3: evaluation
├── out/
│   └── dev_groups.txt      # development groups excluded from evaluation
├── run_final.sh            # all experiments of the paper
└── requirements.txt
```

## Installation

The experiments were run with Python 3.11.

```bash
conda create -n grice python=3.11
conda activate grice
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

If you need a PyTorch build for a specific CUDA version, install it before `requirements.txt`. The NLTK corpora (`wordnet`, `omw-1.4`, `stopwords`) and the sentence-transformers model `all-MiniLM-L6-v2` are downloaded automatically on first use.

## LLM backends

The backend is chosen in `scripts/02_llm.py` with `--backend`:

- `hf` — local Hugging Face models (`Qwen/Qwen3-8B`, `mistralai/Mistral-7B-Instruct-v0.3`), intended to run on a GPU. `--load-in-4bit` additionally requires `bitsandbytes`, which is not in `requirements.txt`.
- `openai` — OpenAI API models (`gpt-5-mini`). Set the key first: `export OPENAI_API_KEY=<your key>`.
- `mock` — deterministic stand-in for testing the pipeline.

LLM answers are cached in an sqlite file next to the output CSV; delete it to query the model again.

## Reproducing the results

All commands are run from the repository root.

```bash
bash run_final.sh 2>&1 | tee out/run_final.log
```

`run_final.sh` runs the full pipeline on both datasets with Qwen3-8B and Mistral-7B, and with gpt-5-mini if `OPENAI_API_KEY` is set. It uses the settings of the paper: feature variant `v2`, prompt version `mix`, λ = 0.7, 5 grouped splits. Settings can be changed through the environment variables `AMBIK`, `INTROPLAN` (data files), `PROMPTS`, `LAMBDA` and `SEEDS`.

Where each result of the paper is written (`<dataset>` is `ambik` or `introplan`, `<backbone>` is `qwen3`, `mistral` or `gpt5mini`):

| Paper | File | Content |
|---|---|---|
| Tables II, III — GRICE | `out/exp_<dataset>_<backbone>/main.json` | `summary.accuracy`, `summary.mcc` (mean and std) |
| Tables II, III — LLM Classifier, Maxim-structured LLM Classifier | `out/llm_classifier_baseline.json` | keys `<dataset>_<backbone>_direct` and `<dataset>_<backbone>_structured` |
| Tables II, III — Action Multiplicity (and inverted), Entropy | `out/exp_<dataset>_<backbone>/baselines.json` | identical for every backbone |
| Table IV — component ablation | `out/exp_ambik_qwen3/ablations.csv` | one row per configuration |
| λ sweep (Section VI, RQ3) | `out/exp_ambik_qwen3/lambda_sweep.csv` | λ in 0, 0.3, 0.5, 0.7, 1 |
| Tables V, VI — accuracy by task type | `out/exp_<dataset>_<backbone>/per_type.csv` | one row per ambiguity type |
| Table VII — classifier coefficients | `out/exp_ambik_<backbone>/coefficients.json` | one weight per maxim |

### Individual stages

The flags below are the settings used in the paper; the defaults of `01_symbolic.py` (`--feature-variant paper`) and `02_llm.py` (`--prompt-version v1`) differ from them.

**Stage 1: symbolic scores.**

```bash
python scripts/01_symbolic.py --dataset ambik --input data/ambik_test_900.csv \
    --context-col environment_short --feature-variant v2 \
    --output out/ambik_symbolic_v2.csv

python scripts/01_symbolic.py --dataset introplan --input data/introplan_mobile.csv \
    --context-col environment_short --instruction-col task \
    --unambiguous-types singlelabel_task unambiguous_task creative_singlelabel_task \
    --feature-variant v2 --output out/introplan_symbolic_v2.csv
```

**Stage 2: LLM scores.**

```bash
python scripts/02_llm.py --input out/ambik_symbolic_v2.csv \
    --output out/ambik_features_qwen3.csv \
    --backend hf --model Qwen/Qwen3-8B --score-mode both --prompt-version mix
```

`--prompt-version mix` uses the v1 prompts for Quantity, Quality and Relation and the v3 prompt for Manner (see `grice/prompts.py`). The script exits with an error if more than 5% of the answers cannot be parsed (`--max-parse-failure`).

**Stage 3: evaluation.**

```bash
python scripts/03_experiments.py --input out/ambik_features_qwen3.csv \
    --outdir out/exp_ambik_qwen3 --lambda 0.7 --exclude-groups out/dev_groups.txt
```

Writes `main.json`, `baselines.json`, `ablations.csv`, `lambda_sweep.csv`, `per_type.csv` and `coefficients.json` to `--outdir`. Results are the mean and standard deviation over 5 grouped train/test splits (`--n-seeds`). The decision threshold is fitted on each training fold for `main.json`, `ablations.csv` and `lambda_sweep.csv`; `per_type.csv` uses the threshold 0.5.

### Development groups

`out/dev_groups.txt` lists 150 AmbiK groups on which the per-maxim prompt versions were selected. `03_experiments.py --exclude-groups out/dev_groups.txt` removes them before evaluation; `run_final.sh` passes this flag automatically.

## Data

| File | Rows | Source |
|---|---|---|
| `data/ambik_test_900.csv` | 900 | AmbiK |
| `data/introplan_mobile.csv` | 600 | IntroPlan |

- **AmbiK** — Ivanova et al., *AmbiK: Dataset of Ambiguous Tasks in Kitchen Environment*, ACL 2025. Original data: https://github.com/cog-model/AmbiK-dataset. Each row gives two instructions: `ambiguous_task` (label 1) and `unambiguous_direct` (label 0), which share a group id, so the file yields 1800 instructions. See the original repository for the dataset's license terms.
- **IntroPlan** — Liang et al., *Introspective Planning: Aligning Robots' Uncertainty with Inherent Task Ambiguity*, NeurIPS 2024. Original data: https://github.com/kevinliang888/IntroPlan. The label is derived from `ambiguity_type`: `singlelabel_task`, `unambiguous_task` and `creative_singlelabel_task` are unambiguous, all other types are ambiguous. Distributed under the MIT License; see `data/LICENSE-IntroPlan`.

## License

No license has been chosen for the code yet. Until a license is added, the code is not licensed for reuse, modification or redistribution. The IntroPlan-derived data file is covered by `data/LICENSE-IntroPlan`.

## Citation

Citation information will be added after publication.

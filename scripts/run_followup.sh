#!/usr/bin/env bash
# Follow-up experiments for the paper, run after phases 6-9 and the fusion sweep
# (the baseline and the sweep_l* / fusion_l* checkpoints must exist).
#
#   bash scripts/run_followup.sh 2>&1 | tee run_logs/followup.log
#
# 1. the output-form consistency loss under the controlled protocol
#    (like-for-like comparison with the error-form loss of the default sweep)
# 2. the single-view sweep with the fusion fine-tuning budget (80 epochs, lr 1e-4)
# 3. re-evaluation of the existing models with the disagreement split into
#    pixels where the reference pair agrees / differs
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
CK=denoiser/checkpoints
SEEDS="${SEEDS:-0,1,2}"

sel() { "$PY" -c "import json;print(format(json.load(open('$CK/$1_selected.json'))['lambda'],'g'))"; }

echo "=== 1/3 output-form loss, controlled protocol ==="
"$PY" -m denoiser.sweep_lambda --prefix outloss --cons-loss output --lambdas 0.1,0.5,1,2 --seeds "$SEEDS" \
    --finetune-epochs 40 -- --batch-size 16 --amp

echo "=== 2/3 single-view sweep with the fusion budget ==="
"$PY" -m denoiser.sweep_lambda --prefix long --cons-loss error --lambdas 0,1,2 --seeds "$SEEDS" \
    --finetune-epochs 80 --finetune-lr 1e-4 -- --batch-size 16 --amp

echo "=== 3/3 disagreement on matched / unmatched pixels ==="
"$PY" -m denoiser.evaluate --no-images --out benchmarks/results/eval_split --models \
    baseline=$CK/baseline/best.pt \
    control_l0=$CK/sweep_l0/best.pt \
    stereo=$CK/sweep_l$(sel sweep)/best.pt \
    long_l0=$CK/long_l0/best.pt \
    fusion_l0=$CK/fusion_l0/best.pt \
    fusion=$CK/fusion_l$(sel fusion)/best.pt

echo "done. Results: benchmarks/results/lambda_sweep_outloss.csv, lambda_sweep_long.csv, eval_split/summary.csv"

#!/usr/bin/env bash
# Phase 9: scripted NVIDIA Nsight runs.
#
#   bash benchmarks/nsight_profile_runs.sh <torchscript_model.pt> [scene]
#
# 1. Nsight Systems timeline of the full stereo pipeline (path tracing kernels,
#    LibTorch inference, host<->device copies), with NVTX ranges
#    frame / pathtrace_left / pathtrace_right / denoise / readback, plus CSV
#    summaries of kernel and NVTX time.
# 2. Nsight Systems runs of the renderer alone at low and high spp.
# 3. Nsight Compute (if its CLI is installed) on the renderer at low and high
#    spp. Open the resulting reports in the Nsight Compute GUI.
# Outputs: benchmarks/results/nsight/
set -euo pipefail

MODEL="${1:?usage: nsight_profile_runs.sh <torchscript_model.pt> [scene]}"
SCENE="${2:-builtin:cornell_glossy}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/benchmarks/results/nsight"
PT="${PATHTRACER_BIN:-$ROOT/build/renderer/pathtracer}"
PIPE="${RUN_PIPELINE_BIN:-$ROOT/build/integration/run_pipeline}"
RES="--width 512 --height 512"
mkdir -p "$OUT"

if ! command -v nsys >/dev/null; then
  echo "nsys (Nsight Systems CLI) not found on PATH - install Nsight Systems or add its bin/ folder" >&2
  exit 1
fi

nsys_run () {  # name, command...
  local name="$1"; shift
  echo "== nsys: $name"
  nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true --force-overwrite=true \
       --output="$OUT/$name" "$@"
  nsys stats --force-export=true --format=csv --output="$OUT/$name" \
       --report=cuda_gpu_kern_sum --report=nvtx_sum --report=cuda_api_sum --report=cuda_gpu_mem_time_sum \
       "$OUT/$name.nsys-rep" || echo "   (some nsys reports are unavailable in this nsys version)"
}

# 1. full pipeline timeline (low spp + denoiser), readback on to expose transfer cost
nsys_run pipeline_spp4 "$PIPE" --scene "$SCENE" $RES --spp 4 --clamp 8 --model "$MODEL" \
  --frames 60 --warmup 10 --readback --no-save --config nsight_pipeline

# 2. renderer only, low vs high spp
for SPP in 1 64; do
  nsys_run pathtrace_spp$SPP "$PT" --scene "$SCENE" $RES --spp $SPP --frames 20 --warmup 3 --no-save --quiet
done

# 3. Nsight Compute. The CLI is `ncu` in recent releases, `nv-nsight-cu-cli` in older ones.
NCU=""
for c in ncu nv-nsight-cu-cli; do command -v $c >/dev/null && NCU=$c && break; done
if [ -n "$NCU" ]; then
  for SPP in 1 64; do
    echo "== $NCU: pathtrace_spp$SPP"
    "$NCU" --report-file "$OUT/ncompute_spp$SPP" \
      "$PT" --scene "$SCENE" $RES --spp $SPP --frames 5 --no-save --quiet \
      || echo "   Nsight Compute run failed (it may need to be run from the GUI on this platform)"
  done
else
  echo "Nsight Compute CLI not found; open $PT in the Nsight Compute GUI with the same arguments instead."
fi

# Hardware memory-bus utilization as a cross-check for the bandwidth estimate
if command -v nvidia-smi >/dev/null; then
  echo "== nvidia-smi utilization during a 64 spp run"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv -lms 100 \
    > "$OUT/smi_spp64.csv" &
  SMI=$!
  "$PT" --scene "$SCENE" $RES --spp 64 --frames 30 --no-save --quiet
  kill $SMI
fi

echo "done - reports in $OUT (open .nsys-rep files in the Nsight Systems GUI)"

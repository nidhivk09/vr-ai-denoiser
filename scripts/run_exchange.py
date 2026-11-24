"""Stereo-consistency experiments with stronger cross-eye sharing (GPU machine).

Builds on the plain fusion models: every new model is fine-tuned from the same
Phase 6 baseline with the fusion budget (80 epochs, lr 1e-4), so it is directly
comparable with fusion_l<lambda> and with the single-view control sweep_l0.

  xchg        fusion + multi-scale cross-eye feature exchange      (--exchange)
  blend       fusion + learned stereo blending of the outputs      (--blend)
  xchg_blend  both

    python scripts/run_exchange.py all                 # everything below, in order
    python scripts/run_exchange.py check               # prerequisites only
    python scripts/run_exchange.py train eval          # some steps
    python scripts/run_exchange.py all --quick         # 5-minute smoke test of the whole chain
    python scripts/run_exchange.py all --variants xchg_blend --seeds 0   # a cheaper run

Steps:
  check      dataset, baseline, sweep_l0 and fusion_l0 must exist (or run them: --make-missing)
  train      one lambda sweep per variant (runs <variant>_l<lambda>[_s<seed>], reuses finished runs)
  eval       all models together on the test scenes -> benchmarks/results/eval_exchange/
  timing     TorchScript export, C++ pipeline per model (FP16) and C++ vs Python check
             -> benchmarks/results/exchange_timing.csv, outputs/exchange/<model>/
  nsight     Nsight Systems (+ Nsight Compute if installed) of the pipeline per model
             -> benchmarks/results/nsight_exchange/ (skipped if nsys is not installed)
  summarize  the tables to read first -> benchmarks/results/exchange_summary.md
  pack       results_exchange.zip in the repo root, to send back

Disagreement is reported against the control (sweep_l0) and the fusion control
(fusion_l0) at equal noise level; see the summary for how to read it.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from common.paths import CHECKPOINT_DIR, RAW_DIR, RESULTS_DIR, find_pipeline  # noqa: E402

PY = sys.executable
OUT = ROOT / "outputs" / "exchange"
VARIANTS = {"xchg": ["--exchange"], "blend": ["--blend"], "xchg_blend": ["--exchange", "--blend"]}
# Set in main(). --quick writes everything under quick_* names, so a smoke test
# can never be mistaken for (or reused as) a real run.
EVAL_DIR = TIMING_CSV = NSIGHT_DIR = SUMMARY = None
TAG = ""


def run_prefix(v: str) -> str:
    return f"{TAG}{v}"


def model_config(ckpt: Path) -> dict:
    return torch.load(ckpt, map_location="cpu", weights_only=False)["model_config"]


def sh(*cmd, check=True):
    cmd = [str(c) for c in cmd]
    print("\n>>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=check)


def selected(prefix: str) -> Path | None:
    path = CHECKPOINT_DIR / f"{prefix}_selected.json"
    if path.exists():
        lam = json.loads(path.read_text()).get("lambda")
        if lam is not None:
            ckpt = CHECKPOINT_DIR / f"{prefix}_l{lam:g}" / "best.pt"
            return ckpt if ckpt.exists() else None
    return None


def selected_lambda(prefix: str):
    path = CHECKPOINT_DIR / f"{prefix}_selected.json"
    return json.loads(path.read_text()).get("lambda") if path.exists() else None


def reference_models() -> dict[str, Path]:
    refs = {"baseline": CHECKPOINT_DIR / "baseline/best.pt", "control_l0": CHECKPOINT_DIR / "sweep_l0/best.pt",
            "stereo": selected("sweep"), "fusion_l0": CHECKPOINT_DIR / "fusion_l0/best.pt",
            "fusion": selected("fusion")}
    return {k: v for k, v in refs.items() if v is not None and v.exists()}


def variant_models(variants) -> dict[str, Path]:
    out = {}
    for v in variants:
        l0 = CHECKPOINT_DIR / f"{run_prefix(v)}_l0/best.pt"
        if l0.exists():
            out[f"{v}_l0"] = l0
        pick = selected(run_prefix(v))
        if pick is not None:
            out[v] = pick
    return out


# ----------------------------------------------------------------------------- steps

def step_check(a):
    problems = []
    if not (RAW_DIR / "manifest.json").exists():
        problems.append("dataset/raw/manifest.json missing -> python scripts/run_phase.py 5")
    if not (ROOT / "dataset/patches/train.npy").exists():
        problems.append("dataset/patches missing -> python scripts/run_phase.py 5")
    if not (CHECKPOINT_DIR / "baseline/best.pt").exists():
        problems.append("baseline checkpoint missing -> python scripts/run_phase.py 6")
    if problems:
        raise SystemExit("prerequisites missing:\n  " + "\n  ".join(problems))
    sweeps = {"sweep": [], "fusion": ["--fusion", "--finetune-epochs", a.epochs, "--finetune-lr", a.lr]}
    for prefix, extra in sweeps.items():
        if (CHECKPOINT_DIR / f"{prefix}_l0/best.pt").exists() and (CHECKPOINT_DIR / f"{prefix}_selected.json").exists():
            print(f"ok: {prefix} sweep found ({prefix}_l0, selected lambda {selected_lambda(prefix)})")
            continue
        if a.quick:
            print(f"warning: no '{prefix}' reference sweep - the quick run continues without it")
            continue
        if not a.make_missing:
            raise SystemExit(f"the '{prefix}' reference sweep is missing (denoiser/checkpoints/{prefix}_l0). Re-run with "
                             "--make-missing to train it now with the same lambdas/seeds, or run "
                             f"`python scripts/run_phase.py 7{' --fusion' if prefix == 'fusion' else ''}`.")
        sh(PY, "-m", "denoiser.sweep_lambda", "--lambdas", a.lambdas, "--seeds", a.seeds, *extra,
           "--", *train_tail(a))
    print("prerequisites ok")


def train_tail(a):
    if a.quick:
        return ["--batch-size", 8, "--max-steps-per-epoch", 5]
    return ["--batch-size", a.batch_size, "--amp"]


def step_train(a):
    for v in a.variants:
        epochs = 2 if a.quick else a.epochs
        sh(PY, "-m", "denoiser.sweep_lambda", "--prefix", run_prefix(v), *VARIANTS[v], "--cons-loss", "error",
           "--lambdas", a.lambdas, "--seeds", a.seeds, "--finetune-epochs", epochs, "--finetune-lr", a.lr,
           "--", *train_tail(a))


def step_eval(a):
    models = {**reference_models(), **variant_models(a.variants)}
    sh(PY, "-m", "denoiser.evaluate", "--models", *(f"{k}={v}" for k, v in models.items()), "--out", EVAL_DIR,
       *(["--no-images"] if a.quick else []))


def step_timing(a):
    manifest = json.loads((RAW_DIR / "manifest.json").read_text())
    scene = next(s for s in manifest["scenes"] if s["split"] == "test")
    models = {k: v for k, v in {**reference_models(), **variant_models(a.variants)}.items()
              if k in ("baseline", "fusion") or k in a.variants}
    pipe = find_pipeline()
    rows = []
    for name, ckpt in models.items():
        meta = model_config(ckpt)
        fusion = ["--fusion", "--ipd", manifest["ipd"]] if meta.get("fusion") else []
        dims = ["--width", manifest["width"], "--height", manifest["height"]]
        ts32, ts16 = ckpt.with_name("best_ts.pt"), ckpt.with_name("best_ts_fp16.pt")
        sh(PY, "-m", "denoiser.export_model", "--checkpoint", ckpt, "--device", "cuda", *dims, "--out", ts32)
        sh(PY, "-m", "denoiser.export_model", "--checkpoint", ckpt, "--device", "cuda", "--half", *dims, "--out", ts16)
        common = [pipe, "--scene", ROOT / scene["scene_file"], *dims, "--spp", 4, "--clamp", manifest["clamp"], *fusion]
        # FP32 pass: the C++ output must match PyTorch on the same buffers
        sh(*common, "--model", ts32, "--frames", 10, "--out", OUT / name / "fp32")
        sh(PY, "-m", "integration.compare_with_python", "--pipeline-out", OUT / name / "fp32", "--checkpoint", ckpt)
        # FP16 pass: the timing that is reported
        frames = 20 if a.quick else 200
        sh(*common, "--model", ts16, "--half", "--frames", frames, "--warmup", 20, "--config", name,
           "--out", OUT / name / "fp16")
        t = json.loads((OUT / name / "fp16" / "timing.json").read_text())
        cmp_path = OUT / name / "fp32" / "python_comparison.json"
        cmp = json.loads(cmp_path.read_text()) if cmp_path.exists() else {}
        rows.append({"model": name, "exchange": bool(meta.get("exchange")), "blend": bool(meta.get("blend")),
                     "render_ms": t["render_left_ms"] + t["render_right_ms"], "denoise_ms": t["denoise_ms"],
                     "total_ms": t["total_ms"], "total_p95_ms": t["total_p95_ms"],
                     "budget_met": t["total_ms"] <= 1000.0 / 90.0,
                     "cpp_matches_python": cmp.get("match", ""),
                     "cpp_max_abs_diff": max((v["max_abs_diff"] for k, v in cmp.items() if k != "match"), default="")})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TIMING_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {TIMING_CSV}")


def step_nsight(a):
    nsys = shutil.which("nsys")
    if nsys is None:
        print("nsys (Nsight Systems CLI) not on PATH - skipping. See GPU_TESTING.md, step 8, to install it.")
        return
    manifest = json.loads((RAW_DIR / "manifest.json").read_text())
    scene = next(s for s in manifest["scenes"] if s["split"] == "test")
    pipe = find_pipeline()
    NSIGHT_DIR.mkdir(parents=True, exist_ok=True)
    models = {k: v for k, v in {**reference_models(), **variant_models(a.variants)}.items()
              if k in ("baseline", "fusion") or k in a.variants}
    ncu = shutil.which("ncu") or shutil.which("nv-nsight-cu-cli")
    for name, ckpt in models.items():
        ts16 = ckpt.with_name("best_ts_fp16.pt")
        if not ts16.exists():
            print(f"{ts16} missing - run the timing step first")
            continue
        meta = model_config(ckpt)
        fusion = ["--fusion", "--ipd", manifest["ipd"]] if meta.get("fusion") else []
        cmd = [pipe, "--scene", ROOT / scene["scene_file"], "--width", manifest["width"], "--height",
               manifest["height"], "--spp", 4, "--clamp", manifest["clamp"], "--model", ts16, "--half", *fusion,
               "--no-save", "--config", name]
        rep = NSIGHT_DIR / f"pipeline_{name}"
        sh(nsys, "profile", "--trace=cuda,nvtx,osrt", "--force-overwrite=true", f"--output={rep}",
           *cmd, "--frames", 60, "--warmup", 10, check=False)
        sh(nsys, "stats", "--force-export=true", "--format=csv", f"--output={rep}", "--report=cuda_gpu_kern_sum",
           "--report=nvtx_sum", f"{rep}.nsys-rep", check=False)
        if ncu:
            # Kernel-level metrics of the denoiser kernels (NVTX range "denoise", nested in "frame").
            out = NSIGHT_DIR / f"ncu_denoise_{name}"
            for flt in ("frame/denoise/", "denoise/"):  # the nesting syntax differs between ncu versions
                sh(ncu, "--nvtx", "--nvtx-include", flt, "--set", "basic", "--launch-count", 300,
                   "--force-overwrite", "--export", out, *cmd, "--frames", 1, "--warmup", 1, check=False)
                if out.with_suffix(".ncu-rep").exists():
                    break
            if out.with_suffix(".ncu-rep").exists():
                with open(out.with_suffix(".csv"), "w") as f:
                    subprocess.run([ncu, "--import", str(out.with_suffix(".ncu-rep")), "--csv", "--page", "raw",
                                    "--metrics", "gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,"
                                    "dram__throughput.avg.pct_of_peak_sustained_elapsed"], stdout=f, check=False)
            else:
                print("Nsight Compute produced no report (see GPU_TESTING.md, troubleshooting: ERR_NVGPUCTRPERM)")
    if not ncu:
        print("Nsight Compute CLI (ncu) not found - Nsight Systems profiles only.")


def _mean_ci(vals):
    vals = [float(v) for v in vals]
    m = sum(vals) / len(vals)
    return m, min(vals), max(vals)


def step_summarize(a):
    """Mean over seeds (and the test scenes) per variant and lambda, against the controls."""
    lines = ["# Cross-eye exchange / blending results", "",
             "Disagreement in units of 1e-3 (display scale, lower = eyes agree more). PSNR in dB.",
             "Ranges are min-max over seeds. dis_matched = pixels where the reference pair agrees "
             "(diffuse); dis_unmatched = where it legitimately differs (highlights, silhouettes).", ""]
    rows_by_spp = defaultdict(list)
    ref_rows = {}
    for v in a.variants:
        path = RESULTS_DIR / f"lambda_sweep_{run_prefix(v)}.csv"
        if not path.exists():
            lines.append(f"(no {path.name} - variant {v} not trained)")
            continue
        groups = defaultdict(list)
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["lambda"] == "":
                    ref_rows[(r["method"], r["spp"])] = r
                else:
                    groups[(float(r["lambda"]), r["spp"])].append(r)
        for (lam, spp), rs in groups.items():
            rows_by_spp[spp].append((f"{v}, lambda={lam:g}", rs))
    for spp in sorted(rows_by_spp, key=int):
        ctrl = ref_rows.get(("nofusion_l0", spp)) or ref_rows.get(("default_l0", spp))
        fus0 = ref_rows.get(("fusion_l0", spp))
        lines += [f"## {spp} spp", "",
                  "| model | PSNR | disagreement | vs control | vs fusion_l0 | matched | unmatched | seeds |",
                  "|---|---|---|---|---|---|---|---|"]
        for name in ("baseline", "nofusion_l0", "nofusion_stereo", "fusion_l0", "fusion_stereo", "ground_truth"):
            r = ref_rows.get((name, spp))
            if r:
                lines.append(f"| {name} | {float(r['psnr']):.2f} | {1e3 * float(r['disagreement']):.3f} | "
                             f"{_rel(r, ctrl)} | {_rel(r, fus0)} | {_k(r, 'dis_matched')} | {_k(r, 'dis_unmatched')} | 1 |")
        for label, rs in sorted(rows_by_spp[spp]):
            p, plo, phi = _mean_ci([r["psnr"] for r in rs])
            d, dlo, dhi = _mean_ci([r["disagreement"] for r in rs])
            mean_row = {"psnr": p, "disagreement": d,
                        "dis_matched": _avg(rs, "dis_matched"), "dis_unmatched": _avg(rs, "dis_unmatched")}
            lines.append(f"| {label} | {p:.2f} [{plo:.2f}, {phi:.2f}] | {1e3 * d:.3f} [{1e3 * dlo:.3f}, {1e3 * dhi:.3f}] | "
                         f"{_rel(mean_row, ctrl)} | {_rel(mean_row, fus0)} | {_k(mean_row, 'dis_matched')} | "
                         f"{_k(mean_row, 'dis_unmatched')} | {len(rs)} |")
        lines.append("")
    picks = {v: selected_lambda(run_prefix(v)) for v in a.variants}
    lines += ["## Selected on the validation split", "",
              *(f"- {v}: lambda = {lam}" for v, lam in picks.items()),
              f"- fusion (reference): lambda = {selected_lambda('fusion')}",
              f"- single-view (reference): lambda = {selected_lambda('sweep')}", ""]
    if TIMING_CSV.exists():
        lines += ["## Frame time (C++ pipeline, FP16, 4 spp, both eyes)", "",
                  "| model | render ms | denoise ms | total ms | p95 ms | 90 FPS budget |", "|---|---|---|---|---|---|"]
        with open(TIMING_CSV) as f:
            for r in csv.DictReader(f):
                lines.append(f"| {r['model']} | {float(r['render_ms']):.2f} | {float(r['denoise_ms']):.2f} | "
                             f"{float(r['total_ms']):.2f} | {float(r['total_p95_ms']):.2f} | "
                             f"{'met' if r['budget_met'] == 'True' else 'MISSED'} |")
        lines.append("")
    lines += ["## How to read this", "",
              "- `vs control`: change in PSNR (dB) and relative change in disagreement against the single-view "
              "control sweep_l0 (same baseline, fine-tuned without consistency term).",
              "- `vs fusion_l0`: the same against plain fusion without consistency term, i.e. what the new "
              "mechanism adds over the fusion input alone.",
              "- The previous best (paper, Table IV): fusion lambda=1, -7.2% disagreement at -0.08 dB (4 spp).",
              "- A clear improvement is a larger disagreement reduction at the same PSNR (inside the seed range), "
              "or the same reduction at a smaller PSNR cost. Check `matched`: that is where disagreement is a "
              "pure denoising artefact.", ""]
    SUMMARY.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {SUMMARY}")


def _avg(rs, k):
    vals = [float(r[k]) for r in rs if r.get(k) not in (None, "")]
    return sum(vals) / len(vals) if vals else ""


def _k(r, key):
    v = r.get(key)
    return "" if v in (None, "") else f"{1e3 * float(v):.3f}"


def _rel(r, ref):
    if not ref:
        return ""
    dp = float(r["psnr"]) - float(ref["psnr"])
    dd = float(r["disagreement"]) / float(ref["disagreement"]) - 1.0
    return f"{dp:+.2f} dB, {100 * dd:+.1f}%"


def step_pack(a):
    dest = ROOT / f"results_exchange{'_quick' if TAG else ''}.zip"
    ev, ns, out = (p.relative_to(ROOT).as_posix() for p in (EVAL_DIR, NSIGHT_DIR, OUT))
    patterns = ["benchmarks/results/lambda_sweep*.csv", "benchmarks/results/exchange_*", f"{ev}/*",
                f"{ns}/*.csv", f"{ns}/*.nsys-rep", f"{ns}/*.ncu-rep",
                "denoiser/checkpoints/*_selected.json", "denoiser/checkpoints/*/done.json",
                "denoiser/checkpoints/*/log.csv", f"{out}/*/*/timing.json",
                f"{out}/*/*/python_comparison.json", f"{out}/*/fp16/anaglyph.png",
                f"{out}/*/fp16/*_denoised.png", "run_logs/*.log", "benchmarks/results/eval_split/*.csv",
                "benchmarks/results/lambda_sweep_*/summary.csv"]
    if a.with_checkpoints:
        for p in variant_models(a.variants).values():
            patterns.append(str(p.relative_to(ROOT)))
    n = 0
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for pat in patterns:
            for p in sorted(ROOT.glob(pat)):
                if p.is_file():
                    z.write(p, p.relative_to(ROOT))
                    n += 1
        z.writestr("ENVIRONMENT.txt", environment())
    print(f"wrote {dest} ({n} files, {dest.stat().st_size / 1e6:.1f} MB) - send this file back")


def environment() -> str:
    out = []
    for cmd in (["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
                ["nvcc", "--version"], [PY, "-c", "import torch;print('torch', torch.__version__, torch.version.cuda)"],
                ["git", "rev-parse", "HEAD"], ["git", "status", "--short"]):
        try:
            out.append("$ " + " ".join(cmd) + "\n" + subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout)
        except OSError as e:
            out.append(f"$ {' '.join(cmd)}\n({e})")
    return "\n".join(out)


STEPS = {"check": step_check, "train": step_train, "eval": step_eval, "timing": step_timing, "nsight": step_nsight,
         "summarize": step_summarize, "pack": step_pack}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="+", help=f"'all' or any of {', '.join(STEPS)}")
    ap.add_argument("--variants", default=",".join(VARIANTS), help="comma-separated subset of " + ", ".join(VARIANTS))
    ap.add_argument("--lambdas", default="0,1,2")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=80, help="fine-tuning epochs (same as the fusion sweep)")
    ap.add_argument("--lr", type=float, default=1e-4, help="fine-tuning peak lr (same as the fusion sweep)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--quick", action="store_true", help="tiny settings: checks the chain works, results meaningless")
    ap.add_argument("--make-missing", action="store_true", help="train missing reference sweeps (sweep_*, fusion_*)")
    ap.add_argument("--with-checkpoints", action="store_true", help="pack: include the selected new models' best.pt")
    a = ap.parse_args()
    global EVAL_DIR, TIMING_CSV, NSIGHT_DIR, SUMMARY, TAG, OUT
    TAG = "quick_" if a.quick else ""
    sfx = "_quick" if a.quick else ""
    EVAL_DIR = RESULTS_DIR / f"eval_exchange{sfx}"
    TIMING_CSV = RESULTS_DIR / f"exchange_timing{sfx}.csv"
    NSIGHT_DIR = RESULTS_DIR / f"nsight_exchange{sfx}"
    SUMMARY = RESULTS_DIR / f"exchange_summary{sfx}.md"
    OUT = ROOT / "outputs" / f"exchange{sfx}"
    if a.quick:
        a.lambdas, a.seeds = "0,1", "0"
    a.variants = [v for v in a.variants.split(",") if v]
    bad = [v for v in a.variants if v not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variants {bad}; choose from {list(VARIANTS)}")
    steps = list(STEPS) if a.steps == ["all"] else a.steps
    for s in steps:
        if s not in STEPS:
            raise SystemExit(f"unknown step {s}; choose from {list(STEPS)} or 'all'")
    for s in steps:
        print(f"\n{'=' * 70}\n{s.upper()}\n{'=' * 70}", flush=True)
        STEPS[s](a)


if __name__ == "__main__":
    main()

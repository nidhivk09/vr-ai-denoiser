"""Phase 7: consistency-weight (lambda) sweep.

Trains one stereo model per lambda (lambda=0 is the paired-data control with
no consistency term), evaluates them together with the Phase 6 baseline, and
writes benchmarks/results/lambda_sweep.csv.

    python -m denoiser.sweep_lambda --lambdas 0,0.1,0.5,1,2,5 --epochs 150
Extra arguments after `--` are passed to train.py, e.g. `-- --amp --batch-size 32`.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import CHECKPOINT_DIR, RESULTS_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lambdas", default="0,0.1,0.5,1,2,5")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--baseline", default=str(CHECKPOINT_DIR / "baseline" / "best.pt"))
    ap.add_argument("--retrain", action="store_true", help="retrain even if a checkpoint exists")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "lambda_sweep")
    args, train_extra = ap.parse_known_args()
    train_extra = [a for a in train_extra if a != "--"]

    lambdas = [float(x) for x in args.lambdas.split(",")]
    models = []
    for lam in lambdas:
        run = f"sweep_l{lam:g}"
        ckpt = CHECKPOINT_DIR / run / "best.pt"
        if args.retrain or not ckpt.exists():
            cmd = [sys.executable, "-m", "denoiser.train", "--mode", "stereo", "--lambda-cons", str(lam),
                   "--epochs", str(args.epochs), "--run-name", run, *train_extra]
            print(">>", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        models.append(f"lambda_{lam:g}={ckpt}")
    if Path(args.baseline).exists():
        models.insert(0, f"baseline={args.baseline}")
    else:
        print(f"note: baseline checkpoint {args.baseline} not found; sweep evaluated without it")

    subprocess.run([sys.executable, "-m", "denoiser.evaluate", "--out", str(args.out), "--no-images",
                    "--models", *models], check=True)

    # Collapse to one row per method (averaged over noise levels) + per-spp rows.
    per = defaultdict(list)
    with open(args.out / "summary.csv") as f:
        for r in csv.DictReader(f):
            per[r["method"]].append(r)
    out_rows = []
    for method, rows in per.items():
        lam = float(method.split("_", 1)[1]) if method.startswith("lambda_") else None
        for r in rows:
            out_rows.append({"method": method, "lambda": "" if lam is None else lam, "spp": r["spp"],
                             "psnr": r["psnr"], "ssim": r["ssim"], "disagreement": r["disagreement"]})
    dest = RESULTS_DIR / "lambda_sweep.csv"
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()

"""Phase 7: consistency-weight (lambda) sweep.

Every model in the sweep is fine-tuned from the same converged checkpoint
(the Phase 6 baseline by default) on paired stereo patches, for the same
number of epochs, and differs only in lambda and the seed. lambda=0 is the
control: same fine-tuning, no consistency term. This keeps the comparison
about lambda - training each lambda from scratch made the result depend on
which runs happened to leave a long loss plateau before the learning rate
decayed.

    python -m denoiser.sweep_lambda --lambdas 0,0.1,0.5,1,2,5 --seeds 0,1,2
    python -m denoiser.sweep_lambda --from-scratch --epochs 150      # old behaviour
Extra arguments after `--` are passed to train.py, e.g. `-- --amp --batch-size 32`.

With --fusion the sweep models are cross-eye fusion models (each eye also sees
the other eye's reprojected noisy color, see denoiser/model.py). They are still
fine-tuned from the same non-fusion baseline, which they reproduce exactly at
the start. Their runs are named fusion_l<lambda>/, results go to
lambda_sweep_fusion.csv and fusion_selected.json, and the non-fusion sweep's
control and selected model are evaluated alongside when they exist.

Outputs: benchmarks/results/lambda_sweep.csv (one row per model and noise
level) and denoiser/checkpoints/sweep_selected.json (the lambda chosen on the
validation split as "the" stereo model). The first seed of each lambda lives
in sweep_l<lambda>/, the others in sweep_l<lambda>_s<seed>/. A finished run is
reused only if its settings still match (see done.json).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import CHECKPOINT_DIR, RESULTS_DIR  # noqa: E402

def run_name(lam: float, seed: int, first_seed: int, prefix: str = "sweep") -> str:
    return f"{prefix}_l{lam:g}" if seed == first_seed else f"{prefix}_l{lam:g}_s{seed}"


def selected_checkpoint(prefix: str = "sweep") -> Path | None:
    """The checkpoint a finished sweep picked as 'the' stereo model, if any."""
    path = CHECKPOINT_DIR / f"{prefix}_selected.json"
    if not path.exists():
        return None
    lam = json.loads(path.read_text()).get("lambda")
    return None if lam is None else CHECKPOINT_DIR / f"{prefix}_l{lam:g}" / "best.pt"


def finished_run(run_dir: Path, expected: dict) -> dict | None:
    """The run's done.json if it completed with the expected settings, else None."""
    done = run_dir / "done.json"
    if not done.exists() or not (run_dir / "best.pt").exists():
        return None
    info = json.loads(done.read_text())
    if any(info["args"].get(k) != v for k, v in expected.items()):
        return None
    return info


def select_lambda(stats: dict[float, list[dict]], l1_tol: float) -> dict:
    """Picks the stereo model on validation data (never the test scenes):
    the lambda > 0 with the lowest output-form consistency whose per-eye L1 is
    within l1_tol of the lambda=0 control. Values are means over seeds."""
    mean = {lam: {k: sum(s[k] for s in runs) / len(runs) for k in ("val_l1", "val_cons")}
            for lam, runs in stats.items()}
    positive = [lam for lam in mean if lam > 0]
    if not positive:
        return {"lambda": None, "reason": "no lambda > 0 in the sweep"}
    if 0.0 in mean:
        limit = mean[0.0]["val_l1"] * (1.0 + l1_tol)
        ok = [lam for lam in positive if mean[lam]["val_l1"] <= limit]
        reason = f"lowest val_cons among lambda > 0 with val_l1 within {l1_tol:.0%} of lambda=0"
        if not ok:
            ok = [min(positive, key=lambda lam: mean[lam]["val_l1"])]
            reason = f"no lambda > 0 kept val_l1 within {l1_tol:.0%} of lambda=0; took the one with the lowest val_l1"
    else:
        ok, reason = positive, "lowest val_cons (no lambda=0 control in the sweep)"
    best = min(ok, key=lambda lam: mean[lam]["val_cons"])
    return {"lambda": best, "reason": reason, "validation_means": {f"{lam:g}": mean[lam] for lam in sorted(mean)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lambdas", default="0,0.1,0.5,1,2,5")
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated training seeds per lambda")
    ap.add_argument("--cons-loss", choices=["error", "output"], default="error")
    ap.add_argument("--fusion", action="store_true", help="sweep cross-eye fusion models (runs fusion_l*)")
    ap.add_argument("--init", default=str(CHECKPOINT_DIR / "baseline" / "best.pt"),
                    help="checkpoint every sweep model is fine-tuned from")
    ap.add_argument("--finetune-epochs", type=int, default=40)
    ap.add_argument("--finetune-lr", type=float, default=5e-5)
    ap.add_argument("--from-scratch", action="store_true",
                    help="train each model from random weights for --epochs instead of fine-tuning")
    ap.add_argument("--epochs", type=int, default=150, help="epochs per model with --from-scratch")
    ap.add_argument("--baseline", default=str(CHECKPOINT_DIR / "baseline" / "best.pt"))
    ap.add_argument("--select-l1-tol", type=float, default=0.03,
                    help="max relative val_l1 increase over lambda=0 allowed for the selected stereo model")
    ap.add_argument("--retrain", action="store_true", help="retrain even if a matching finished run exists")
    ap.add_argument("--out", type=Path, default=None)
    args, train_extra = ap.parse_known_args()
    train_extra = [a for a in train_extra if a != "--"]
    prefix = "fusion" if args.fusion else "sweep"
    suffix = "_fusion" if args.fusion else ""
    args.out = args.out or RESULTS_DIR / f"lambda_sweep{suffix}"
    selected_path = CHECKPOINT_DIR / f"{prefix}_selected.json"
    if args.fusion:
        train_extra.append("--fusion")

    lambdas = [float(x) for x in args.lambdas.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    if args.from_scratch:
        schedule = {"epochs": args.epochs, "init": None}
        schedule_cmd = ["--epochs", str(args.epochs)]
    else:
        init = Path(args.init)
        if not init.exists():
            raise SystemExit(f"{init} not found - run Phase 6 first (or pass --init / --from-scratch)")
        schedule = {"epochs": args.finetune_epochs, "lr": args.finetune_lr, "init": str(init),
                    "init_mtime": init.stat().st_mtime_ns}
        schedule_cmd = ["--epochs", str(args.finetune_epochs), "--lr", str(args.finetune_lr), "--init", str(init)]

    models = []
    info = {}  # method -> (lambda, seed)
    stats = defaultdict(list)
    for lam in lambdas:
        for seed in seeds:
            run = run_name(lam, seed, seeds[0], prefix)
            expected = {"mode": "stereo", "lambda_cons": lam, "cons_loss": args.cons_loss, "seed": seed, **schedule}
            if args.fusion:  # (not checked otherwise, so runs made before --fusion existed stay reusable)
                expected["fusion"] = True
            done = None if args.retrain else finished_run(CHECKPOINT_DIR / run, expected)
            if done is None:
                cmd = [sys.executable, "-m", "denoiser.train", "--mode", "stereo", "--lambda-cons", str(lam),
                       "--cons-loss", args.cons_loss, "--seed", str(seed), "--run-name", run,
                       *train_extra, *schedule_cmd]
                print(">>", " ".join(cmd), flush=True)
                subprocess.run(cmd, check=True)
                done = json.loads((CHECKPOINT_DIR / run / "done.json").read_text())
            else:
                print(f"reusing finished run {run}")
            stats[lam].append(done["best"])
            method = f"lambda_{lam:g}" if seed == seeds[0] else f"lambda_{lam:g}_s{seed}"
            info[method] = (lam, seed)
            models.append(f"{method}={CHECKPOINT_DIR / run / 'best.pt'}")

    selected = select_lambda(stats, args.select_l1_tol)
    if selected["lambda"] is not None:
        selected["checkpoint"] = str(CHECKPOINT_DIR / run_name(selected["lambda"], seeds[0], seeds[0], prefix) / "best.pt")
    selected.update(cons_loss=args.cons_loss, seeds=seeds, from_scratch=args.from_scratch, fusion=args.fusion)
    selected_path.write_text(json.dumps(selected, indent=2))
    print(f"selected stereo model: lambda = {selected['lambda']} ({selected['reason']}) -> {selected_path}")

    if args.fusion:  # the non-fusion sweep's control and pick, for a direct comparison
        refs = {"nofusion_l0": CHECKPOINT_DIR / "sweep_l0" / "best.pt", "nofusion_stereo": selected_checkpoint("sweep")}
        for name, ckpt in reversed(refs.items()):
            if ckpt is not None and ckpt.exists():
                models.insert(0, f"{name}={ckpt}")
    if Path(args.baseline).exists():
        models.insert(0, f"baseline={args.baseline}")
    else:
        print(f"note: baseline checkpoint {args.baseline} not found; sweep evaluated without it")

    subprocess.run([sys.executable, "-m", "denoiser.evaluate", "--out", str(args.out), "--no-images",
                    "--models", *models], check=True)

    out_rows = []
    with open(args.out / "summary.csv") as f:
        for r in csv.DictReader(f):
            lam, seed = info.get(r["method"], ("", ""))
            out_rows.append({"method": r["method"], "lambda": lam, "seed": seed, "spp": r["spp"],
                             "psnr": r["psnr"], "ssim": r["ssim"], "disagreement": r["disagreement"]})
    dest = RESULTS_DIR / f"lambda_sweep{suffix}.csv"
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {dest}")

    # Mean over seeds and scenes: the table to read first.
    floor = {r["spp"]: float(r["disagreement"]) for r in out_rows if r["method"] == "ground_truth"}
    groups = defaultdict(list)
    for r in out_rows:
        label = f"{'fusion ' if args.fusion else ''}lambda {r['lambda']:g}" if r["lambda"] != "" else r["method"]
        groups[(label, int(r["spp"]))].append(r)
    print(f"\n{'model':>18} {'spp':>4} {'PSNR':>7} {'disagree':>9} {'excess over GT':>15} {'runs':>5}")
    for (label, spp), rows in groups.items():
        psnr = sum(float(r["psnr"]) for r in rows) / len(rows)
        dis = sum(float(r["disagreement"]) for r in rows) / len(rows)
        print(f"{label:>18} {spp:>4} {psnr:7.2f} {dis:9.5f} {dis - floor.get(str(spp), 0.0):15.5f} {len(rows):>5}")


if __name__ == "__main__":
    main()

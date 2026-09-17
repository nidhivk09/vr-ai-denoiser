"""Phases 6 & 7: train the U-Net denoiser.

  Baseline (Phase 6, single image):
      python -m denoiser.train --mode mono --run-name baseline
  Stereo-consistent (Phase 7):
      python -m denoiser.train --mode stereo --lambda-cons 0.5 --run-name stereo_l0.5

Both modes are validated on the same stereo validation patches, reporting
per-eye reconstruction error (val_l1) and cross-eye consistency (val_cons),
so their curves are directly comparable. Logs go to
denoiser/checkpoints/<run>/log.csv; the best model is best.pt.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import CHECKPOINT_DIR, PATCH_DIR  # noqa: E402
from denoiser.dataset import PatchDataset  # noqa: E402
from denoiser.losses import hdr_l1, stereo_consistency_loss  # noqa: E402
from denoiser.model import Denoiser, count_parameters  # noqa: E402


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stereo_losses(model, batch, device, lambda_cons: float, amp_ctx):
    li = batch["left_input"].to(device, non_blocking=True)
    ri = batch["right_input"].to(device, non_blocking=True)
    lt = batch["left_target"].to(device, non_blocking=True)
    rt = batch["right_target"].to(device, non_blocking=True)
    ld = batch["left_disp"].to(device, non_blocking=True)
    rd = batch["right_disp"].to(device, non_blocking=True)
    with amp_ctx():
        pred = model(torch.cat([li, ri], dim=0))
    pred = pred.float()
    lp, rp = pred.chunk(2, dim=0)
    l1 = hdr_l1(lp, lt) + hdr_l1(rp, rt)
    cons = stereo_consistency_loss(lp, rp, ld, rd)
    total = l1 + lambda_cons * cons if lambda_cons > 0 else l1
    return total, l1, cons


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["mono", "stereo"], default="mono")
    ap.add_argument("--lambda-cons", type=float, default=0.5, help="consistency weight (stereo mode)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--patches", type=Path, default=PATCH_DIR)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="stereo pairs per step (mono mode uses 2x this many single-eye images, so both see the same image count)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=32, help="U-Net base channel count")
    ap.add_argument("--levels", type=int, default=3, help="U-Net downsampling levels")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps-per-epoch", type=int, default=0, help="limit steps (smoke tests)")
    ap.add_argument("--patience", type=int, default=0, help="early stop after N epochs without improvement (0 = off)")
    args = ap.parse_args()

    if args.mode == "mono":
        args.lambda_cons = 0.0
    run = args.run_name or (f"baseline" if args.mode == "mono" else f"stereo_l{args.lambda_cons:g}")
    out_dir = CHECKPOINT_DIR / run
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    train_set = PatchDataset("train", args.mode, args.patches, augment=True)
    val_set = PatchDataset("val", "stereo", args.patches, augment=False)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size if args.mode == "stereo" else 2 * args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.workers, pin_memory=pin,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=pin)
    if len(train_loader) == 0:
        raise SystemExit("not enough training patches for one batch - lower --batch-size")

    model = Denoiser(args.base, args.levels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps_per_epoch = len(train_loader) if args.max_steps_per_epoch <= 0 else min(len(train_loader), args.max_steps_per_epoch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.05)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def amp_ctx():
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)

    print(f"run '{run}' | mode {args.mode} | lambda {args.lambda_cons} | device {device} | "
          f"{count_parameters(model) / 1e6:.2f}M params")
    print(f"train samples {len(train_set)} ({steps_per_epoch} steps/epoch), val stereo patches {len(val_set)}")
    (out_dir / "args.json").write_text(json.dumps({**vars(args), "patches": str(args.patches), "run": run}, indent=2))

    log_path = out_dir / "log.csv"
    fields = ["epoch", "train_loss", "train_l1", "train_cons", "val_l1", "val_cons", "val_objective", "lr", "epoch_sec"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(fields)

    best = math.inf
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        sums = np.zeros(3)
        n = 0
        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            opt.zero_grad(set_to_none=True)
            if args.mode == "mono":
                x = batch["input"].to(device, non_blocking=True)
                y = batch["target"].to(device, non_blocking=True)
                with amp_ctx():
                    pred = model(x)
                l1 = hdr_l1(pred.float(), y)
                loss, cons = l1, torch.zeros((), device=device)
            else:
                loss, l1, cons = stereo_losses(model, batch, device, args.lambda_cons, amp_ctx)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            sums += [loss.item(), l1.item(), cons.item()]
            n += 1

        model.eval()
        vsum = np.zeros(2)
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                _, l1, cons = stereo_losses(model, batch, device, 0.0, amp_ctx)
                vsum += [0.5 * l1.item(), cons.item()]  # l1 reported per eye
                vn += 1
        val_l1, val_cons = vsum / max(vn, 1)
        objective = 2.0 * val_l1 + args.lambda_cons * val_cons
        dt = time.perf_counter() - t0
        row = [epoch, *(sums / max(n, 1)), val_l1, val_cons, objective, sched.get_last_lr()[0], dt]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])

        ckpt = {
            "model_state": model.state_dict(),
            "model_config": model.config(),
            "epoch": epoch,
            "val_l1": val_l1,
            "val_cons": val_cons,
            "val_objective": objective,
            "mode": args.mode,
            "lambda_cons": args.lambda_cons,
            "run": run,
        }
        torch.save(ckpt, out_dir / "last.pt")
        improved = objective < best
        if improved:
            best = objective
            stale = 0
            torch.save(ckpt, out_dir / "best.pt")
        else:
            stale += 1
        print(f"epoch {epoch:4d} | train {row[1]:.4f} (l1 {row[2]:.4f} cons {row[3]:.4f}) | "
              f"val l1 {val_l1:.4f} cons {val_cons:.4f} | {dt:5.1f}s{' *' if improved else ''}", flush=True)
        if args.patience and stale >= args.patience:
            print(f"early stop: no improvement for {args.patience} epochs")
            break

    print(f"best val objective {best:.5f}; checkpoints in {out_dir}")


if __name__ == "__main__":
    main()

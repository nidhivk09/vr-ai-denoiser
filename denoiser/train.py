"""Phases 6 & 7: train the U-Net denoiser.

  Baseline (Phase 6, single image):
      python -m denoiser.train --mode mono --run-name baseline
  Stereo-consistent (Phase 7):
      python -m denoiser.train --mode stereo --lambda-cons 0.5 --run-name stereo_l0.5
  Fine-tune a converged model with the consistency term (what the sweep does):
      python -m denoiser.train --mode stereo --lambda-cons 0.5 --init denoiser/checkpoints/baseline/best.pt \
          --epochs 40 --lr 5e-5 --run-name sweep_l0.5

Stronger cross-eye sharing (fusion models only, see denoiser/model.py):
--exchange adds the multi-scale feature exchange, --blend the learned output
blending; either implies --fusion. Parameters the --init checkpoint does not
have start at an identity initialisation and are trained at --new-lr-mult
times the learning rate, so a short fine-tune is enough for them to move.

--cons-loss picks the consistency term: "error" (default) compares the two
eyes' errors against ground truth and is zero for a perfect prediction;
"output" compares the predictions directly (the original form).

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
from denoiser.model import Denoiser, count_parameters, pair_input  # noqa: E402


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stereo_losses(model, batch, device, lambda_cons: float, amp_ctx, cons_loss: str = "error"):
    """Returns (total, l1, cons_output, cons_term). cons_output is always the
    output-form consistency (comparable across runs and with the evaluation
    metric); cons_term is the form selected by cons_loss, the one in the total."""
    li = batch["left_input"].to(device, non_blocking=True)
    ri = batch["right_input"].to(device, non_blocking=True)
    lt = batch["left_target"].to(device, non_blocking=True)
    rt = batch["right_target"].to(device, non_blocking=True)
    ld = batch["left_disp"].to(device, non_blocking=True)
    rd = batch["right_disp"].to(device, non_blocking=True)
    with amp_ctx():
        pred = model(pair_input(model, li, ri, ld, rd))
    pred = pred.float()
    lp, rp = pred.chunk(2, dim=0)
    l1 = hdr_l1(lp, lt) + hdr_l1(rp, rt)
    cons_output = stereo_consistency_loss(lp, rp, ld, rd)
    if cons_loss == "error":
        cons_term = stereo_consistency_loss(lp, rp, ld, rd, left_target=lt, right_target=rt)
    else:
        cons_term = cons_output
    total = l1 + lambda_cons * cons_term if lambda_cons > 0 else l1
    return total, l1, cons_output, cons_term


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["mono", "stereo"], default="mono")
    ap.add_argument("--lambda-cons", type=float, default=0.5, help="consistency weight (stereo mode)")
    ap.add_argument("--cons-loss", choices=["error", "output"], default="error",
                    help="consistency term: 'error' = eyes' errors vs ground truth (zero at a perfect "
                         "prediction), 'output' = predictions compared directly")
    ap.add_argument("--fusion", action="store_true",
                    help="cross-eye fusion: each eye also sees the other eye's reprojected noisy color (stereo mode)")
    ap.add_argument("--exchange", action="store_true",
                    help="multi-scale cross-eye feature exchange inside the U-Net (implies --fusion)")
    ap.add_argument("--blend", action="store_true",
                    help="learned stereo blending of the two eyes' outputs (implies --fusion)")
    ap.add_argument("--new-lr-mult", type=float, default=10.0,
                    help="learning-rate multiplier for parameters the --init checkpoint does not have")
    ap.add_argument("--init", type=Path, default=None,
                    help="start from this checkpoint's weights (fine-tuning); its model config overrides --base/--levels")
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

    if args.exchange or args.blend:
        args.fusion = True
    if args.mode == "mono":
        args.lambda_cons = 0.0
        if args.fusion:
            raise SystemExit("--fusion needs stereo pairs: use --mode stereo")
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

    init_mtime = None
    if args.init is not None:
        init_ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
        args.base, args.levels = init_ckpt["model_config"]["base"], init_ckpt["model_config"]["levels"]
        init_mtime = args.init.stat().st_mtime_ns
        for key, flag in (("fusion", "--fusion"), ("exchange", "--exchange"), ("blend", "--blend")):
            if init_ckpt["model_config"].get(key) and not getattr(args, key):
                raise SystemExit(f"{args.init} has {key}: pass {flag}")
    model = Denoiser(args.base, args.levels, args.fusion, args.exchange, args.blend).to(device)
    new_params: set[str] = set()
    if args.init is not None:
        new_params = model.new_parameter_names(init_ckpt["model_state"])
        model.load_weights(init_ckpt["model_state"])  # a simpler init starts the model as its exact copy
    # Parameters the init does not have (the zero-initialised exchange/blend blocks; the new input
    # channels of the first convolution share their tensor with old ones and stay in the base group)
    fresh = [p for n, p in model.named_parameters() if n in new_params and n != "unet.enc.0.body.0.weight"]
    fresh_ids = {id(p) for p in fresh}
    groups = [{"params": [p for p in model.parameters() if id(p) not in fresh_ids], "lr": args.lr}]
    max_lrs = [args.lr]
    if fresh:
        groups.append({"params": fresh, "lr": args.lr * args.new_lr_mult})
        max_lrs.append(args.lr * args.new_lr_mult)
    opt = torch.optim.Adam(groups, lr=args.lr)
    steps_per_epoch = len(train_loader) if args.max_steps_per_epoch <= 0 else min(len(train_loader), args.max_steps_per_epoch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lrs, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.05)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def amp_ctx():
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)

    if args.init is not None:
        print(f"initialised from {args.init} (epoch {init_ckpt.get('epoch')}, val_l1 {init_ckpt.get('val_l1')})")
    extras = "".join(f" + {k}" for k in ("fusion", "exchange", "blend") if getattr(args, k))
    if fresh:
        print(f"{sum(p.numel() for p in fresh)} new parameters at {args.new_lr_mult:g}x lr")
    print(f"run '{run}' | mode {args.mode}{extras} | lambda {args.lambda_cons} "
          f"({args.cons_loss}) | device {device} | "
          f"{count_parameters(model) / 1e6:.2f}M params")
    print(f"train samples {len(train_set)} ({steps_per_epoch} steps/epoch), val stereo patches {len(val_set)}")
    run_args = {**vars(args), "patches": str(args.patches), "init": str(args.init) if args.init else None,
                "init_mtime": init_mtime, "run": run}
    (out_dir / "args.json").write_text(json.dumps(run_args, indent=2))
    (out_dir / "done.json").unlink(missing_ok=True)  # written again only when this run finishes

    log_path = out_dir / "log.csv"
    # train_cons / val_cons_term: the consistency term being optimised (--cons-loss).
    # val_cons: always the output form, so curves of different runs are comparable.
    fields = ["epoch", "train_loss", "train_l1", "train_cons", "val_l1", "val_cons", "val_objective", "lr",
              "epoch_sec", "val_cons_term"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(fields)

    best = math.inf
    best_stats = {}
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
                loss, l1, _, cons = stereo_losses(model, batch, device, args.lambda_cons, amp_ctx, args.cons_loss)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            sums += [loss.item(), l1.item(), cons.item()]
            n += 1

        model.eval()
        vsum = np.zeros(3)
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                _, l1, cons, cons_term = stereo_losses(model, batch, device, 0.0, amp_ctx, args.cons_loss)
                vsum += [0.5 * l1.item(), cons.item(), cons_term.item()]  # l1 reported per eye
                vn += 1
        val_l1, val_cons, val_cons_term = (float(v) for v in vsum / max(vn, 1))
        objective = 2.0 * val_l1 + args.lambda_cons * val_cons_term  # same form as the training loss
        dt = time.perf_counter() - t0
        row = [epoch, *(sums / max(n, 1)), val_l1, val_cons, objective, sched.get_last_lr()[0], dt, val_cons_term]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])

        ckpt = {
            "model_state": model.state_dict(),
            "model_config": model.config(),
            "epoch": epoch,
            "val_l1": val_l1,
            "val_cons": val_cons,
            "val_cons_term": val_cons_term,
            "val_objective": objective,
            "mode": args.mode,
            "lambda_cons": args.lambda_cons,
            "cons_loss": args.cons_loss,
            "seed": args.seed,
            "init": str(args.init) if args.init else None,
            "run": run,
        }
        torch.save(ckpt, out_dir / "last.pt")
        improved = objective < best
        if improved:
            best = objective
            best_stats = {"epoch": epoch, "val_l1": val_l1, "val_cons": val_cons, "val_cons_term": val_cons_term,
                          "val_objective": objective}
            stale = 0
            torch.save(ckpt, out_dir / "best.pt")
        else:
            stale += 1
        print(f"epoch {epoch:4d} | train {row[1]:.4f} (l1 {row[2]:.4f} cons {row[3]:.4f}) | "
              f"val l1 {val_l1:.4f} cons {val_cons:.4f} | {dt:5.1f}s{' *' if improved else ''}", flush=True)
        if args.patience and stale >= args.patience:
            print(f"early stop: no improvement for {args.patience} epochs")
            break

    # Marks the run as complete. sweep_lambda.py reuses a checkpoint only if this
    # file exists and its settings match, so interrupted or outdated runs are redone.
    (out_dir / "done.json").write_text(json.dumps({"args": run_args, "best": best_stats}, indent=2))
    print(f"best val objective {best:.5f}; checkpoints in {out_dir}")


if __name__ == "__main__":
    main()

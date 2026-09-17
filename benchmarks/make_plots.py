"""Phase 9: turn the CSVs in benchmarks/results/ into paper figures.

Every figure is optional - a plot is skipped (with a note) when its CSV has
not been produced yet. Output: benchmarks/results/figures/*.png and *.pdf.

    python -m benchmarks.make_plots
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import CHECKPOINT_DIR, RESULTS_DIR  # noqa: E402

# Categorical slots in fixed order (validated reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
VR_BUDGET_MS = 1000.0 / 90.0

# Fixed entity -> color, so a config keeps its color in every figure.
CONFIG_ORDER = ["highspp_pathtrace", "lowspp_pathtrace", "lowspp_baseline", "lowspp_stereo"]
CONFIG_SHORT = {
    "highspp_pathtrace": "High spp",
    "lowspp_pathtrace": "Low spp",
    "lowspp_baseline": "Low spp\n+ baseline",
    "lowspp_stereo": "Low spp\n+ stereo",
}
CONFIG_LABEL = {
    "highspp_pathtrace": "High-spp path tracing",
    "lowspp_pathtrace": "Low-spp path tracing",
    "lowspp_baseline": "Low-spp + baseline denoiser",
    "lowspp_stereo": "Low-spp + stereo denoiser",
}
METHOD_COLOR = {"noisy": SERIES[1], "baseline": SERIES[2], "stereo": SERIES[3], "ground_truth": SERIES[0]}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2.0, "lines.markersize": 7,
})


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  skip: {path.name} not found")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"  wrote figures/{name}.png")


def mean(vals) -> float:
    vals = [float(v) for v in vals if v not in ("", None)]
    return sum(vals) / len(vals) if vals else float("nan")


def bar_labels(ax, bars, fmt="{:.1f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK_2)


def plot_frame_times(res: Path, out: Path) -> None:
    rows = read_csv(res / "frame_time_summary.csv")
    if not rows:
        return
    rows = sorted(rows, key=lambda r: CONFIG_ORDER.index(r["config"]) if r["config"] in CONFIG_ORDER else 99)
    stages = [("render_left_ms", "Path trace (left)"), ("render_right_ms", "Path trace (right)"),
              ("denoise_ms", "Denoise (both eyes)"), ("readback_ms", "Readback")]
    stages = [s for s in stages if any(float(r[s[0]]) > 0 for r in rows)]
    labels = [CONFIG_LABEL.get(r["config"], r["config"]) for r in rows]
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bottom = [0.0] * len(rows)
    for i, (key, name) in enumerate(stages):
        vals = [float(r[key]) for r in rows]
        # 2px surface gap between stacked segments
        ax.bar(x, vals, bottom=bottom, color=SERIES[i], label=name, width=0.6, edgecolor=SURFACE, linewidth=1.5)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for xi, total in zip(x, bottom):
        ax.annotate(f"{total:.1f} ms", (xi, total), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    ax.axhline(VR_BUDGET_MS, color=INK, linewidth=1.2, linestyle="--")
    ax.text(0.99, VR_BUDGET_MS * 1.04, "90 FPS budget (11.1 ms)", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color=INK)
    if max(bottom) > 20 * VR_BUDGET_MS:
        ax.set_yscale("log")
        ax.set_ylabel("Frame time, ms (log scale)")
    else:
        ax.set_ylabel("Frame time (ms)")
    ax.set_xticks(list(x), labels, rotation=12, ha="right")
    ax.set_title("Stereo frame time vs. VR budget")
    ax.legend(loc="upper right", ncols=1, fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, out, "frame_time_vs_budget")


def plot_quality_by_config(res: Path, out: Path) -> None:
    rows = read_csv(res / "quality_by_config.csv")
    if not rows:
        return
    rows = sorted(rows, key=lambda r: CONFIG_ORDER.index(r["config"]) if r["config"] in CONFIG_ORDER else 99)
    labels = [CONFIG_SHORT.get(r["config"], r["config"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (key, title, fmt) in zip(axes, [("psnr", "PSNR (dB, higher is better)", "{:.1f}"),
                                           ("ssim", "SSIM (higher is better)", "{:.3f}"),
                                           ("disagreement", "Stereo disagreement (lower is better)", "{:.4f}")]):
        bars = ax.bar(range(len(rows)), [float(r[key]) for r in rows], color=SERIES[0], width=0.6)
        bar_labels(ax, bars, fmt)
        ax.set_xticks(range(len(rows)), labels, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
    save(fig, out, "quality_by_config")


def plot_eval_summary(res: Path, out: Path) -> None:
    rows = read_csv(res / "eval" / "summary.csv")
    if not rows:
        return
    by = defaultdict(dict)
    for r in rows:
        by[r["method"]][int(r["spp"])] = r
    spps = sorted({int(r["spp"]) for r in rows})
    methods = [m for m in by]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    width = 0.8 / len(methods)
    for ax, key, title in [(axes[0], "psnr", "PSNR vs. ground truth (dB)"),
                           (axes[1], "disagreement", "Stereo disagreement (lower is better)")]:
        for i, m in enumerate(methods):
            if key == "psnr" and m == "ground_truth":
                continue
            xs = [j + (i - len(methods) / 2 + 0.5) * width for j in range(len(spps))]
            color = METHOD_COLOR.get(m, SERIES[(4 + i) % len(SERIES)])
            ax.bar(xs, [float(by[m][s][key]) for s in spps], width=width * 0.92, color=color, label=m)
        ax.set_xticks(range(len(spps)), [f"{s} spp" for s in spps])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[1].legend(fontsize=8)
    save(fig, out, "denoiser_eval_by_spp")


def plot_lambda_sweep(res: Path, out: Path) -> None:
    rows = [r for r in read_csv(res / "lambda_sweep.csv") if r["lambda"] != ""]
    if not rows:
        return
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        lam = float(r["lambda"])
        for k in ("psnr", "ssim", "disagreement"):
            agg[lam][k].append(float(r[k]))
    lams = sorted(agg)
    baseline = [r for r in read_csv(res / "lambda_sweep.csv") if r["method"] == "baseline"]
    # Two panels sharing the x axis instead of one dual-axis chart.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, title in [(axes[0], "psnr", "Per-eye quality: PSNR (dB)"),
                           (axes[1], "disagreement", "Stereo disagreement")]:
        ys = [mean(agg[l][key]) for l in lams]
        ax.plot(lams, ys, marker="o", color=SERIES[0], label="stereo-trained")
        if baseline:
            b = mean(r[key] for r in baseline)
            ax.axhline(b, color=INK_2, linestyle="--", linewidth=1.2)
            ax.annotate("baseline (single-image)", (lams[-1], b), xytext=(0, 4), textcoords="offset points",
                        ha="right", fontsize=8, color=INK_2)
        ax.set_xlabel("consistency weight λ")
        ax.set_title(title, fontsize=10)
    save(fig, out, "lambda_sweep")


def plot_divergence(res: Path, out: Path) -> None:
    rows = read_csv(res / "divergence.csv")
    if not rows:
        return
    spp = [int(r["spp"]) for r in rows]
    blocks = sorted(int(k.split("_b")[1]) for k in rows[0] if k.startswith("divergence_b"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, b in enumerate(blocks):
        axes[0].plot(spp, [100 * float(r[f"divergence_b{b}"]) for r in rows], marker="o", color=SERIES[i],
                     label=f"{b}x{b} block")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("samples per pixel")
    axes[0].set_ylabel("idle thread capacity (%)")
    axes[0].set_title("Traversal-work divergence", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].plot(spp, [float(r["traversal_gb_per_sec"]) for r in rows], marker="o", color=SERIES[0])
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("samples per pixel")
    axes[1].set_ylabel("GB/s")
    axes[1].set_title("BVH + primitive memory throughput", fontsize=10)
    save(fig, out, "divergence_vs_spp")


def plot_block_sweep(res: Path, out: Path) -> None:
    rows = read_csv(res / "block_size_sweep.csv")
    if not rows:
        return
    by_spp = defaultdict(list)
    for r in rows:
        by_spp[int(r["spp"])].append(r)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    groups = sorted(by_spp)
    blocks = sorted({int(r["block_x"]) for r in rows})
    width = 0.8 / len(blocks)
    for i, b in enumerate(blocks):
        xs = [j + (i - len(blocks) / 2 + 0.5) * width for j in range(len(groups))]
        ys = [mean(r["mean_ms"] for r in by_spp[g] if int(r["block_x"]) == b) for g in groups]
        bars = ax.bar(xs, ys, width=width * 0.92, color=SERIES[i], label=f"{b}x{b}")
        bar_labels(ax, bars, "{:.1f}")
    ax.set_xticks(range(len(groups)), [f"{g} spp" for g in groups])
    ax.set_ylabel("ms per frame (one eye)")
    ax.set_title("CUDA thread block size")
    ax.legend(title="block", fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis="x", visible=False)
    save(fig, out, "block_size_sweep")


def plot_cpu_vs_gpu(res: Path, out: Path) -> None:
    rows = read_csv(res / "cpu_vs_gpu.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bars = ax.bar([r["config"].replace("_", " ") for r in rows], [float(r["speedup_vs_1_thread"]) for r in rows],
                  color=SERIES[0], width=0.55)
    bar_labels(ax, bars, "x{:.1f}")
    ax.set_yscale("log")
    ax.set_ylabel("speed-up vs. 1 CPU thread (log)")
    ax.set_title(f"Renderer speed-up ({rows[0]['width']}x{rows[0]['height']}, {rows[0]['spp']} spp)")
    ax.grid(axis="x", visible=False)
    save(fig, out, "cpu_vs_gpu_speedup")


def plot_stage_share(res: Path, out: Path) -> None:
    """Irregular (path tracing) vs. regular (neural) compute share of a denoised frame."""
    rows = [r for r in read_csv(res / "frame_time_summary.csv") if r["config"] in ("lowspp_baseline", "lowspp_stereo")]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 2.4))
    for yi, r in enumerate(rows):
        parts = [("Path tracing (irregular)", float(r["render_left_ms"]) + float(r["render_right_ms"])),
                 ("Neural denoising (regular)", float(r["denoise_ms"])),
                 ("Transfer / other", max(0.0, float(r["total_ms"]) - float(r["render_left_ms"])
                                          - float(r["render_right_ms"]) - float(r["denoise_ms"])))]
        left = 0.0
        total = sum(p[1] for p in parts)
        for i, (name, v) in enumerate(parts):
            ax.barh(yi, v, left=left, color=SERIES[i], edgecolor=SURFACE, linewidth=1.5, height=0.55,
                    label=name if yi == 0 else None)
            if v / total > 0.08:
                ax.text(left + v / 2, yi, f"{100 * v / total:.0f}%", ha="center", va="center", fontsize=8, color="white")
            left += v
    ax.set_yticks(range(len(rows)), [CONFIG_LABEL[r["config"]] for r in rows], fontsize=8)
    ax.set_xlabel("ms per stereo frame")
    ax.set_title("Where the frame time goes")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncols=3)
    ax.grid(axis="y", visible=False)
    save(fig, out, "stage_breakdown")


def plot_training_curves(out: Path) -> None:
    runs = sorted(p.parent for p in CHECKPOINT_DIR.glob("*/log.csv"))
    runs = [r for r in runs if not r.name.startswith("smoke")]
    if not runs:
        print("  skip: no training logs")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, run in enumerate(runs[:len(SERIES)]):
        rows = read_csv(run / "log.csv")
        ep = [int(r["epoch"]) for r in rows]
        axes[0].plot(ep, [float(r["val_l1"]) for r in rows], color=SERIES[i], label=run.name, linewidth=1.6)
        axes[1].plot(ep, [float(r["val_cons"]) for r in rows], color=SERIES[i], label=run.name, linewidth=1.6)
    axes[0].set_title("Validation reconstruction L1 (log space, per eye)", fontsize=10)
    axes[1].set_title("Validation stereo consistency loss", fontsize=10)
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[1].legend(fontsize=8)
    save(fig, out, "training_curves")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()
    out = args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for fn in (plot_frame_times, plot_stage_share, plot_quality_by_config, plot_eval_summary, plot_lambda_sweep,
               plot_divergence, plot_block_sweep, plot_cpu_vs_gpu):
        print(fn.__name__)
        fn(args.results, out)
    print("plot_training_curves")
    plot_training_curves(out)


if __name__ == "__main__":
    main()

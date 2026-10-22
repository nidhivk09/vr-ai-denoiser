"""Builds the images and numbers for the demo page (outputs/demo/site/).

Expects the renders made by the demo run in outputs/demo/ (convergence series,
reference, no-NEE, stereo, stats), a dataset in dataset/raw and the
demo_baseline / demo_stereo checkpoints. Writes JPEGs to outputs/demo/site/img
and all metrics to outputs/demo/site/data.json.

    python scripts/make_demo_assets.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.divergence_analysis import block_divergence, colormap  # noqa: E402
from benchmarks.quality_metrics import display, stereo_metrics  # noqa: E402
from common.exr import read_exr, read_exr_channels  # noqa: E402
from common.imageops import to_uint8, tonemap  # noqa: E402
from common.paths import CHECKPOINT_DIR, RAW_DIR  # noqa: E402
from dataset import layout  # noqa: E402
from denoiser.dataset import load_scene_pair, scenes_in_split  # noqa: E402
from denoiser.losses import visibility_masks, warp_right_to_left  # noqa: E402
from denoiser.model import denoise_full, load_checkpoint  # noqa: E402
from denoiser.train import pick_device  # noqa: E402

DEMO = ROOT / "outputs/demo"
SITE = DEMO / "site"
IMG = SITE / "img"
CONV_SPP = [1, 4, 16, 64, 256, 1024]


def save_jpg(name: str, img01: np.ndarray, size: int | None = None, quality: int = 90) -> str:
    im = Image.fromarray(to_uint8(img01))
    if size:
        im = im.resize((size, size), Image.LANCZOS)
    IMG.mkdir(parents=True, exist_ok=True)
    im.save(IMG / name, quality=quality, subsampling=0)
    return f"img/{name}"


def copy_png(src: Path, name: str, **kw) -> str:
    return save_jpg(name, np.asarray(Image.open(src).convert("RGB"), dtype=np.float32) / 255.0, **kw)


def psnr_display(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((tonemap(a) - tonemap(b)) ** 2))
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


def render_ms(csv_path: Path) -> list[float]:
    with open(csv_path) as f:
        return [float(r["render_ms"]) for r in csv.DictReader(f)]


def t_img(t: torch.Tensor) -> np.ndarray:
    return t[0].permute(1, 2, 0).cpu().numpy()


def path_tracer_section(data: dict) -> None:
    ref = read_exr(DEMO / "reference/color.exr")
    data["reference"] = {"img": copy_png(DEMO / "reference/color.png", "reference.jpg", quality=94), "spp": 4096}
    times = render_ms(DEMO / "conv_times.csv")
    conv = []
    for spp, ms in zip(CONV_SPP, times):
        img = read_exr(DEMO / f"conv/spp{spp}/color.exr")
        conv.append({"spp": spp, "ms": ms, "psnr": psnr_display(img, ref),
                     "rmse": float(np.sqrt(np.mean((np.clip(img, 0, 4) - np.clip(ref, 0, 4)) ** 2))),
                     "img": copy_png(DEMO / f"conv/spp{spp}/color.png", f"conv_{spp}.jpg")})
    data["convergence"] = conv

    for buf in ("normal", "albedo", "depth"):
        data.setdefault("buffers", {})[buf] = copy_png(DEMO / f"reference/{buf}.png", f"buf_{buf}.jpg", size=384)

    nonee = read_exr(DEMO / "nonee/color.exr")
    nee = read_exr(DEMO / "conv/spp16/color.exr")
    data["nee"] = {"with": {"img": "img/conv_16.jpg", "psnr": psnr_display(nee, ref)},
                   "without": {"img": copy_png(DEMO / "nonee/color.png", "nonee_16.jpg"),
                               "psnr": psnr_display(nonee, ref)}}

    one = render_ms(DEMO / "thread1.csv")[0]
    data["threads"] = {"one_thread_ms": one, "all_threads_ms": times[CONV_SPP.index(16)], "spp": 16}

    st = read_exr_channels(DEMO / "stats/stats.exr")
    work = st["nodes"] + st["prims"]
    data["workload"] = {
        "img": save_jpg("work_heatmap.jpg", colormap(work / np.percentile(work, 99.5))),
        "mean_work": float(work.mean()), "max_work": float(work.max()),
        "divergence": {str(b): block_divergence(work, b) for b in (8, 16, 32)},
        "mean_bounces": float(st["bounces"].mean()),
    }


def stereo_views_section(data: dict) -> None:
    """Stereo convergence series (both eyes) and the stereo camera orbit."""
    sc = DEMO / "sconv"
    times = render_ms(DEMO / "sconv_times.csv")  # left, right per spp
    ref = {eye: read_exr(sc / f"spp{CONV_SPP[-1]}/{eye}_color.exr") for eye in ("left", "right")}
    series = []
    for k, spp in enumerate(CONV_SPP):
        row = {"spp": spp, "ms": times[2 * k] + times[2 * k + 1]}
        for eye in ("left", "right"):
            row[eye] = copy_png(sc / f"spp{spp}/{eye}_color.png", f"s{spp}_{eye}.jpg", quality=86)
        row["anaglyph"] = copy_png(sc / f"spp{spp}/anaglyph.png", f"s{spp}_anaglyph.jpg", quality=86)
        if spp != CONV_SPP[-1]:
            row["psnr"] = 0.5 * sum(psnr_display(read_exr(sc / f"spp{spp}/{e}_color.exr"), ref[e]) for e in ref)
        series.append(row)
    data["stereo_conv"] = series

    frames = sorted((DEMO / "orbit").glob("f*"), key=lambda d: int(d.name[1:]))
    data["orbit"] = {"yaw": [-20 + 40 * i / (len(frames) - 1) for i in range(len(frames))], "spp": 192, "frames": [
        {eye: copy_png(f / f"{eye}.png", f"o{f.name[1:]}_{eye}.jpg", quality=84)
         for eye in ("left_color", "right_color", "anaglyph")} for f in frames]}


def stereo_section(data: dict) -> None:
    s = DEMO / "stereo"
    meta = json.loads((s / "meta.json").read_text())
    disp = read_exr(s / "left_disparity.exr")[..., 0]
    data["stereo"] = {
        "left": copy_png(s / "left_color.png", "stereo_left.jpg"),
        "right": copy_png(s / "right_color.png", "stereo_right.jpg"),
        "anaglyph": copy_png(s / "anaglyph.png", "anaglyph.jpg"),
        "disparity": copy_png(s / "left_disparity.png", "disparity.jpg", size=384),
        "disp_min": float(np.percentile(disp[disp > 1], 1)), "disp_max": float(np.percentile(disp, 99.9)),
        "ipd": meta.get("ipd", 0.063), "spp": meta.get("spp", 512),
    }


def denoiser_section(data: dict) -> None:
    device = pick_device("auto")
    manifest = layout.load_manifest(RAW_DIR)
    models = {name: load_checkpoint(CHECKPOINT_DIR / f"demo_{name}/best.pt", device)[0]
              for name in ("baseline", "stereo")}
    h, w = manifest["height"], manifest["width"]
    warm = torch.zeros(2, 9, h, w, device=device)
    for m in models.values():
        denoise_full(m, warm)

    scenes = scenes_in_split("test")
    rows = []
    for sid in scenes:
        for spp in manifest["noisy_spp"]:
            s = {k: v.to(device) for k, v in load_scene_pair(sid, spp).items()}
            gl, gr, ld, rd = s["left_target"], s["right_target"], s["left_disp"], s["right_disp"]
            outs = {"noisy": (s["left_input"][:, :3], s["right_input"][:, :3], 0.0)}
            for name, m in models.items():
                if device.type == "mps":
                    torch.mps.synchronize()
                t0 = time.perf_counter()
                pred = denoise_full(m, torch.cat([s["left_input"], s["right_input"]], dim=0))
                if device.type == "mps":
                    torch.mps.synchronize()
                outs[name] = (pred[0:1], pred[1:2], (time.perf_counter() - t0) * 1000.0)
            outs["ground_truth"] = (gl, gr, 0.0)
            for method, (l, r, ms) in outs.items():
                rows.append({"scene": sid, "spp": spp, "method": method, "ms": ms,
                             **{k: float(v) for k, v in stereo_metrics(l, r, gl, gr, ld, rd).items()
                                if isinstance(v, (int, float))}})
            if sid == scenes[0]:
                lm, _ = visibility_masks(ld, rd)
                for method, (l, r, _) in outs.items():
                    save_jpg(f"dn_{sid}_spp{spp}_{method}.jpg", t_img(display(l)))
                    warped, _ = warp_right_to_left(display(r), ld)
                    err = ((display(l) - warped).abs().mean(dim=1, keepdim=True) * lm * 5).clamp(0, 1)
                    save_jpg(f"err_{sid}_spp{spp}_{method}.jpg", colormap(t_img(err)[..., 0]))

    summary = {}
    for r in rows:
        key = f"{r['method']}@{r['spp']}"
        summary.setdefault(key, []).append(r)
    data["denoiser"] = {
        "device": str(device), "res": [w, h], "showcase_scene": scenes[0], "test_scenes": scenes,
        "noisy_spp": manifest["noisy_spp"], "gt_spp": manifest["gt_spp"],
        "summary": {k: {m: float(np.mean([r[m] for r in v])) for m in ("psnr", "ssim", "disagreement", "ms")}
                    for k, v in summary.items()},
    }
    for name in ("baseline", "stereo"):
        with open(CHECKPOINT_DIR / f"demo_{name}/log.csv") as f:
            data["denoiser"][f"log_{name}"] = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def dataset_section(data: dict) -> None:
    manifest = layout.load_manifest(RAW_DIR)
    data["dataset"] = {
        "scenes": [{"id": s["id"], "split": s["split"],
                    "img": copy_png(RAW_DIR / s["id"] / "gt/left_color.png", f"scene_{s['id']}.jpg", size=192)}
                   for s in manifest["scenes"]],
        "res": [manifest["width"], manifest["height"]], "gt_spp": manifest["gt_spp"],
        "noisy_spp": manifest["noisy_spp"],
    }
    stats = ROOT / "dataset/patches/stats.json"
    if stats.exists():
        data["dataset"]["patch_stats"] = json.loads(stats.read_text())


def pipeline_section(data: dict) -> None:
    """C++ LibTorch pipeline run (outputs/demo/pipeline), if one was made."""
    out = DEMO / "pipeline"
    if not (out / "timing.json").exists():
        return
    meta = json.loads((out / "run.json").read_text()) if (out / "run.json").exists() else {}
    cmp = json.loads((out / "python_comparison.json").read_text())
    data["pipeline"] = {
        "timing": json.loads((out / "timing.json").read_text()),
        "max_abs_diff": max(cmp["left"]["max_abs_diff"], cmp["right"]["max_abs_diff"]),
        "match": cmp["match"], "res": meta.get("res", data["denoiser"]["res"]), "spp": meta.get("spp", 4),
    }


def write_pages(data: dict) -> None:
    """index.html is a standalone page for offline use; artifact.html is the same body without the skeleton."""
    body = (ROOT / "scripts/demo_template.html").read_text()
    body = body.replace("__DEMO_DATA__", json.dumps(data).replace("</", "<\\/"))
    (SITE / "artifact.html").write_text(body)
    (SITE / "index.html").write_text('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
                                     '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                                     f"</head><body>\n{body}\n</body></html>\n")


def main() -> None:
    data: dict = {"generated": time.strftime("%d %b %Y")}
    for section in (path_tracer_section, stereo_views_section, stereo_section, dataset_section, denoiser_section, pipeline_section):
        section(data)
        print(f"done: {section.__name__}", flush=True)
    (SITE / "data.json").write_text(json.dumps(data, indent=1))
    write_pages(data)
    print(f"wrote {SITE / 'index.html'}")


if __name__ == "__main__":
    main()

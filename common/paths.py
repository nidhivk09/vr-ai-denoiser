"""Repository paths and helpers to locate the compiled binaries."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw"
PATCH_DIR = DATASET_DIR / "patches"
CHECKPOINT_DIR = REPO_ROOT / "denoiser" / "checkpoints"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

_EXE = ".exe" if sys.platform == "win32" else ""


def _find_binary(name: str, env_var: str, subdirs: list[str]) -> Path:
    override = os.environ.get(env_var)
    if override:
        p = Path(override)
        if p.is_file():
            return p
        raise FileNotFoundError(f"{env_var}={override} does not exist")
    candidates = []
    for build in ("build", "build-release", "out/build"):
        for sub in subdirs:
            for cfg in ("", "Release", "RelWithDebInfo", "Debug"):
                candidates.append(REPO_ROOT / build / sub / cfg / f"{name}{_EXE}")
    for c in candidates:
        if c.is_file():
            return c
    on_path = shutil.which(name)
    if on_path:
        return Path(on_path)
    raise FileNotFoundError(
        f"could not find the '{name}' binary. Build it first (see README) or set {env_var}=/path/to/{name}{_EXE}"
    )


def find_renderer() -> Path:
    return _find_binary("pathtracer", "PATHTRACER_BIN", ["renderer", "."])


def find_pipeline() -> Path:
    return _find_binary("run_pipeline", "RUN_PIPELINE_BIN", ["integration", "."])

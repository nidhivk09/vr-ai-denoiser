# GPU testing: cross-eye feature exchange and learned stereo blending

This file is written to be handed to **Claude Code on the NVIDIA machine** (or followed
by hand). It takes a fresh laptop to a single zip file of results to send back.

> **To the person running this:** open a terminal in an empty folder, start `claude`, and
> say: *"Follow GPU_TESTING.md from https://github.com/nidhivk09/vr-ai-denoiser
> (branch `stereo-exchange`) step by step. Ask me before anything that needs sudo or
> admin rights."* Keep the laptop **plugged in** and stop it from sleeping
> (Windows: Settings → System → Power → Screen and sleep → *Never* while plugged in).
> The full run takes several hours.

> **Already did the full run (steps 0-12) once?** Then only the *Final run* below is
> needed. Skip steps 0-12.

## Final run (about 3 h, mostly unattended)

The first run is finished. This last one, in the same repository folder as before:

1. **Re-evaluates** the trained models with a new metric (stereo error against the
   ground-truth pair). Finished training runs are reused, not retrained.
2. **Trains two cheaper exchange variants** (`xchg123`, `xchg23`: exchange only at the
   coarser U-Net levels), 9 short runs each, because `xchg` missed the 90 FPS budget.
3. **Re-times** every model in the C++ pipeline. The export now pads convolution inputs to
   multiples of 8 channels (the output is the same), so the timing changes.
4. **Benchmarks** the denoiser variants in Python with a CUDA kernel table.

The rules below still apply (no pushing, no source edits, log everything).

```bash
cd ~/vr-ai-denoiser && source .venv/bin/activate
git stash                      # only if git reports local changes (they are result files)
git fetch origin && git checkout stereo-exchange && git pull
for f in tests/test_*.py; do python -m unittest tests.$(basename $f .py) 2>&1 | tail -1; done   # Expected: OK x4
mkdir -p run_logs
tmux new -s final              # reattach later with: tmux attach -t final
python scripts/run_exchange.py train eval timing bench summarize 2>&1 | tee run_logs/final.log
python scripts/run_exchange.py pack --with-checkpoints
```

**Expected:**
- For `xchg`, `blend` and `xchg_blend`, every run prints `reusing finished run ...`. Only
  `xchg123_*` and `xchg23_*` runs train (lines like `run 'xchg23_l0' | ... + exchange`).
  If an `xchg_l*`, `blend_l*` or `xchg_blend_l*` run starts training, stop and record it
  in `run_logs/NOTES.md`.
- The timing step prints `traced vs eager max abs diff` values near 0 for every model and
  no `C++ and Python outputs differ` warning.
- The printed `exchange_summary.md` has numbers in the `stereo error` column, rows for
  `xchg123` and `xchg23`, the frame-time table (now 7 models) and the
  *Denoiser alone* table.
- `results_exchange.zip` exists (a few hundred MB; it includes the trained models).

The `nsight` step is not needed this time.

Send back `results_exchange.zip`, plus `run_logs/NOTES.md` if anything went wrong (the
zip includes `run_logs/` anyway).

---

## Rules for Claude Code

1. Work only in the cloned repository. **Do not push, commit, or open pull requests.**
   The results go back as a zip file (step 12).
2. **Do not edit the source code to make a step pass.** If a step fails, look in
   *Troubleshooting* at the end. If the fix is not there, stop and write down the exact
   command, the full error and what you tried in `run_logs/NOTES.md`. Settings changes
   that the steps allow (for example `--batch-size`) are fine, but record them in
   `run_logs/NOTES.md` too.
3. Log every long command with `2>&1 | tee run_logs/<name>.log`. The `run_logs/`
   folder goes into the result zip.
4. Run long steps inside `tmux` (step 7) so a closed terminal does not kill them.
5. After each step, check its **Expected** line before moving on.

---

## Step 0: Platform

- **Windows laptop:** use **WSL2 with Ubuntu 24.04**. The earlier GPU runs used it, and
  everything below assumes Linux. In an *administrator* PowerShell:
  ```powershell
  wsl --install -d Ubuntu-24.04
  ```
  Reboot if asked, open "Ubuntu 24.04" from the Start menu and create a user. Run every
  later command **inside Ubuntu**, in the Linux home folder (`~`), **not** under
  `/mnt/c/...` (that is about 10x slower).
- **Native Linux:** skip this step.

**Expected:** a Linux shell.

## Step 1: NVIDIA driver and CUDA toolkit

- **Windows + WSL2:** install or update the normal **Windows** NVIDIA driver (GeForce
  Experience / NVIDIA App, or nvidia.com/drivers). **Do not install a Linux NVIDIA driver
  inside WSL**, because WSL uses the Windows one.
- **Native Linux:** install the distribution's recommended NVIDIA driver
  (`sudo ubuntu-drivers install`) and reboot.

Then, inside Linux, install the CUDA toolkit (12.8 supports every GPU from GTX 10xx up to
RTX 50xx):

```bash
nvidia-smi                       # must list the GPU. If not, fix the driver first
# WSL2 users: use the WSL-Ubuntu repo. Native Ubuntu: replace "wsl-ubuntu" with "ubuntu2404"
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8 build-essential cmake git python3-venv python3-dev tmux zip
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
nvcc --version                   # release 12.8
```

**Expected:** `nvidia-smi` shows the GPU, and `nvcc --version` prints 12.x. Write the GPU
name and VRAM into `run_logs/NOTES.md` once the repo is cloned.

## Step 2: Clone the code

```bash
cd ~
git clone https://github.com/nidhivk09/vr-ai-denoiser.git
cd vr-ai-denoiser
git checkout stereo-exchange
mkdir -p run_logs
```

**If this machine already has `vr-ai-denoiser/` from an earlier run** (look for
`denoiser/checkpoints/baseline/best.pt`), use that folder instead and keep its data:
```bash
cd ~/vr-ai-denoiser && git fetch origin && git checkout stereo-exchange && git pull
ls denoiser/checkpoints/        # note what exists: baseline, sweep_l*, fusion_l*
```
The earlier runs' checkpoints are still valid; the new code loads them unchanged.

## Step 3: Python environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128   # cu128 works for all current GPUs
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Expected:** `... True <GPU name>`. Run `source .venv/bin/activate` again in every new
terminal or tmux window.

## Step 4: Build the renderer and the C++ pipeline

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build -j
./build/renderer/pathtracer --info
ls build/integration/run_pipeline
```

**Expected:** the CMake output shows `CUDA backend: ON` and no "LibTorch not found"
warning, `--info` prints the GPU, and `run_pipeline` exists.

## Step 5: Tests (about 1 s, no GPU needed)

```bash
ctest --test-dir build --output-on-failure
python -m unittest discover -s tests -v 2>&1 | tail -5
```

**Expected:** all C++ tests pass, and the Python run ends in `Ran 80 tests ... OK`. If
anything fails, stop and record it (rule 2).

## Step 6: Dataset and reference models (skip what already exists)

The new models are compared against the single-view control (`sweep_l0`) and plain fusion
(`fusion_l0`), all fine-tuned from the same baseline. Check what exists:

```bash
ls dataset/patches/train.npy denoiser/checkpoints/baseline/best.pt \
   denoiser/checkpoints/sweep_selected.json denoiser/checkpoints/fusion_selected.json 2>&1
```

Run only the parts that are missing:

```bash
# dataset (30-90 min); only if dataset/patches/train.npy is missing
python scripts/run_phase.py 5 2>&1 | tee run_logs/phase5.log
# baseline (2-6 h); only if baseline/best.pt is missing
python scripts/run_phase.py 6 2>&1 | tee run_logs/phase6.log
```

The single-view sweep and the cross-eye fusion sweep are trained in step 9.

> If a new dataset or baseline is made here, all numbers will differ slightly from the
> paper's. That is fine: everything in the result zip is compared within the same run.

## Step 7: Smoke test (about 5-10 min)

```bash
tmux new -s gpu                  # detach: Ctrl-b then d; reattach: tmux attach -t gpu
source .venv/bin/activate
python scripts/run_exchange.py all --quick --make-missing 2>&1 | tee run_logs/exchange_quick.log
```

This uses tiny settings, and its numbers mean nothing. Everything it writes has
`quick` in its name, so it cannot mix with the real run. It checks that training, export,
the C++ pipeline (`--fusion`), the C++ vs Python comparison, Nsight and packing all work.

**Expected:** it ends with `wrote .../results_exchange_quick.zip`. In the log, each
`C++ and Python outputs MATCH` line should appear (one per model).

## Step 8: Nsight (free; optional but wanted)

NVIDIA Nsight Systems (`nsys`) and Nsight Compute (`ncu`) are **free**. On Linux they come
with the CUDA toolkit from step 1:

```bash
which nsys ncu || ls /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys \
                     /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null
# if found only under /opt/..., add those folders to PATH, e.g.
# echo 'export PATH=$(ls -d /opt/nvidia/nsight-systems/*/bin | tail -1):$(ls -d /opt/nvidia/nsight-compute/* | tail -1):$PATH' >> ~/.bashrc
# if missing entirely:
sudo apt-get install -y cuda-nsight-systems-12-8 cuda-nsight-compute-12-8
nsys --version && ncu --version
```

`ncu` needs access to GPU performance counters. If it fails with **`ERR_NVGPUCTRPERM`**:
- **WSL2 / Windows:** in Windows, open NVIDIA Control Panel → *Desktop* menu → enable
  *Developer Settings* → *Developer* → *Manage GPU Performance Counters* → *Allow access to
  the GPU performance counters to all users*. Then rerun.
- **Native Linux:** run the nsight step with `sudo -E env PATH=$PATH .venv/bin/python scripts/run_exchange.py nsight`,
  or set `options nvidia NVreg_RestrictProfilingToAdminUsers=0` in
  `/etc/modprobe.d/nvidia-profiling.conf` and reboot.

If `ncu` still does not work, carry on: the script then only records Nsight Systems, and
that is enough. If `nsys` is missing too, the step is skipped; write that in `NOTES.md`.

## Step 9: Cross-eye fusion, phases 7-9 with `--fusion` (2-5 h)

Cross-eye fusion (each eye also sees the other eye's reprojected noisy samples) has not
been run on this machine yet. It is the reference for everything in step 10, so run it
first, in tmux.

**9a. Single-view sweep (only if `denoiser/checkpoints/sweep_selected.json` is missing).**
This is the control that fusion is compared against (3-8 h with the paper's λ values):

```bash
python scripts/run_phase.py 7 2>&1 | tee run_logs/phase7.log
# if time is short: python scripts/run_phase.py 7 --lambdas 0,1,2,5   (write it in NOTES.md)
```

**9b. Fusion sweep, C++ pipeline and benchmarks.** Look up the GPU's memory bandwidth in
GB/s on its spec sheet (`nvidia-smi --query-gpu=name --format=csv` gives the model; for
example RTX 3060 Laptop 336, RTX 3070 Laptop 448, RTX 4060/4070 Laptop 256, RTX 4080 Laptop
432, RTX 4090 Laptop 576). It only scales a percentage, so use 0 if unsure.

```bash
python scripts/run_phase.py 7 8 9 --fusion --lambdas 0,1,2 --gpu-bandwidth <GB/s> 2>&1 | tee run_logs/fusion.log
```

- Phase 7 `--fusion`: λ ∈ {0, 1, 2} × 3 seeds, each fine-tuned from the baseline for 80
  epochs at lr 1e-4 (runs `fusion_l*`), then all models are evaluated together.
- Phase 8 `--fusion`: TorchScript export, the C++ pipeline with `run_pipeline --fusion`
  (it builds the disparity channel on the GPU), and a C++ vs Python check.
- Phase 9 `--fusion`: frame times and quality of the four configurations with the fusion
  model as the stereo model, workload divergence, Nsight (if `nsys` is installed, step 8),
  and the plots.

**Expected:**
- `benchmarks/results/lambda_sweep_fusion.csv` and `denoiser/checkpoints/fusion_selected.json` exist.
- `benchmarks/results/eval_fusion/summary.csv` has rows `fusion_l0`, `fusion` and (if 9a ran) `control_l0`.
- `outputs/phase8_fusion/python_comparison.json` says `"match": true`.
- `outputs/phase8_fusion/timing.json` exists, and its `denoise_ms` compared with `outputs/phase8/timing.json` is the cost of the cross-eye input.

Phase 9 writes to `benchmarks/results/` directly, so keep a copy before step 10 (the copy goes into the zip):

```bash
mkdir -p benchmarks/results/phase9_fusion
for f in frame_time_summary.csv frame_times.csv quality_by_config.csv quality_per_scene.csv \
         divergence.csv figures nsight; do
  [ -e benchmarks/results/$f ] && cp -r benchmarks/results/$f benchmarks/results/phase9_fusion/
done
ls benchmarks/results/phase9_fusion
```

## Step 10: Exchange / blending, the full run (several hours)

Inside tmux:

```bash
python scripts/run_exchange.py all --make-missing 2>&1 | tee run_logs/exchange.log
```

What it does (each step can also be run alone, e.g. `python scripts/run_exchange.py eval`):

| step | what | time (rough, laptop GPU) |
|---|---|---|
| check | finds the `sweep_*` / `fusion_*` sweeps from step 9 (trains them only if missing) | seconds |
| train | 3 variants (`xchg`, `blend`, `xchg_blend`) × λ 0,1,2 × 3 seeds, 80 epochs each, from the baseline | 2-6 h |
| eval | every model on the 4 test scenes at 1/4/8 spp | 5 min |
| timing | TorchScript export (FP32, FP16), C++ pipeline at 4 spp, C++ vs Python check | 5 min |
| nsight | `nsys` timeline + `ncu` denoiser kernels per model | 5-10 min |
| summarize | `benchmarks/results/exchange_summary.md` | seconds |
| pack | `results_exchange.zip` | seconds |

The run can be resumed: re-running the same command reuses every finished training run
(see `done.json` in each run folder).

**If time is short**, run the strongest variant only, with one seed first:
```bash
python scripts/run_exchange.py all --make-missing --variants xchg_blend --seeds 0
```
and add the other seeds later with `--seeds 0,1,2`, which reuses seed 0.

**If CUDA runs out of memory**, add `--batch-size 8` to the command, keep it for every
later command, and record it in `NOTES.md`. If the reference sweeps already exist with
batch 16, they are not retrained, so the comparison is then less clean. Write that down too.

**Expected:** `benchmarks/results/exchange_summary.md` is printed at the end, and
`results_exchange.zip` exists.

## Step 11: Optional follow-up experiments for the paper (2-4 h)

These are separate pending items: the output-form loss under the controlled protocol, the
single-view sweep with the fusion budget, and the matched/unmatched disagreement split.
Run them only if steps 1-10 are done and there is time:

```bash
bash scripts/run_followup.sh 2>&1 | tee run_logs/followup.log
python scripts/run_exchange.py pack          # re-pack so the zip includes them
```

## Step 12: Send back

1. Fill in `run_logs/NOTES.md`: the GPU, the driver (`nvidia-smi`), any deviations, any
   errors, and anything that looked odd.
2. Re-pack, so the zip includes NOTES.md:
   ```bash
   python scripts/run_exchange.py pack
   ```
   Add `--with-checkpoints` if the trained models are wanted as well (about 10-50 MB more).
3. Send **`results_exchange.zip`** (from the repo folder; in Windows Explorer it is at
   `\\wsl$\Ubuntu-24.04\home\<user>\vr-ai-denoiser\`).

It contains the summary table, all sweep CSVs, the evaluation images, the timing CSV, the
Nsight reports, the training logs and an `ENVIRONMENT.txt` with the GPU, driver, CUDA,
PyTorch and git commit.

**Opening Nsight reports on a Mac:** Nsight Systems and Nsight Compute have free macOS
viewers (they cannot profile, only view). Download them from
developer.nvidia.com/nsight-systems and developer.nvidia.com/nsight-compute (a free NVIDIA
developer account is needed), then open the `.nsys-rep` / `.ncu-rep` files from the zip.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is False | The CPU wheel was installed. `pip uninstall -y torch`, then reinstall it with the `--index-url .../cu128` line. On WSL, `nvidia-smi` must work first. |
| `no kernel image is available` / `sm_120 is not compatible` | The GPU is newer than the wheel. Use the cu128 wheel (RTX 50xx needs CUDA 12.8+). |
| CMake: `CUDA backend: OFF` | `nvcc` is not on PATH: `export PATH=/usr/local/cuda/bin:$PATH`, delete `build/`, and configure again. |
| `Unsupported gpu architecture` or `no kernel image` from the renderer | Delete `build/` and configure again with `-DCMAKE_CUDA_ARCHITECTURES=<n>`: 61 GTX 10xx, 75 RTX 20xx/GTX 16xx, 86 RTX 30xx, 89 RTX 40xx, 120 RTX 50xx. |
| CMake: LibTorch not found | The `-DCMAKE_PREFIX_PATH=$(python -c ...)` part ran outside the venv. Activate `.venv` and configure again. |
| `run_pipeline: error while loading shared libraries: libtorch...` | `export LD_LIBRARY_PATH=$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib:$LD_LIBRARY_PATH` |
| `could not find the 'run_pipeline' binary` | Step 4 did not finish, or set `RUN_PIPELINE_BIN=/full/path/to/run_pipeline`. |
| `CUDA out of memory` during training | `--batch-size 8` (see step 10). |
| `WARNING: C++ and Python outputs differ` | Do not change the code. Record the printed diff values in NOTES.md and continue. |
| `traced model does not match eager model` during export | Record it in NOTES.md, then rerun that one step (`python scripts/run_exchange.py timing`). If it happens again, stop. |
| `ncu`: `ERR_NVGPUCTRPERM` | See step 8. |
| `ncu`: "No kernels were profiled" | Fine to skip. The `nsys` report covers the timeline. |
| WSL runs out of RAM / gets killed | Make a `C:\Users\<you>\.wslconfig` with `[wsl2]` and `memory=12GB` (about 3/4 of the RAM), then `wsl --shutdown` in PowerShell. Rerun; finished runs are reused. |
| Laptop throttles or the run is very slow | Plug it in, set the Windows power mode to *Best performance*, and close games or browsers. Training times are not reported, so throttling only costs time, but the *timing* step should run on a cool, idle machine. |

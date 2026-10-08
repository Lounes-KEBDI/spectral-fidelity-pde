"""Autoregressive inference of every finished DPOT-Ti fine-tuning run on the PhysBiasBench test grid.
Predictions only: no metric and no evaluation. The ground truth is not saved.

Protocol (Chu et al., arXiv:2605.29283, Figure 1 and Section 3.3): for every test trajectory the model gets
frames 0-4 and predicts frames 5-19, i.e. 10 in-horizon frames, then 5 extra OOD rollout frames. The models
were trained for one step only (DPOT-Ti/finetune.py), so the 15 frames are predicted autoregressively:
each prediction is appended to the 5-frame input window and the oldest frame is dropped. Ground truth is
never used after frame 4. Choices that are not in the paper are marked "NOT IN PAPER" (NOT_IN_PAPER.md,
section 6).

Models: every run whose training is finished, i.e. checkpoint/DPOT-Ti/{variant}/{mix}/{pde}/best.pt
exists and last.pt does not (finetune.py's is_finished). Unfinished runs are listed and skipped. Each model
runs only on its own PDE's grid: 25 cells x 50 trajectories.

Outputs, in predictions/DPOT-Ti/ (predictions/DPOT-Ti_smoke/ with SMOKE_TEST):
  {variant}/{mix}/{pde}/{dynamic}__{ic}.npy   float32 [50, 15, C, 64, 64], raw physical units.
                                              Row k is test trajectory k; frame j is sequence frame 5 + j
                                              (j = 0-9 in-horizon, 10-14 OOD rollout).
  manifest.csv                                one row per .npy, columns in MANIFEST_COLUMNS
Resume: cells whose .npy already exists are skipped, so the job can simply be submitted again.

Layout: only the "DPOT-Ti specific" section depends on the model. Everything else is meant to be copied
verbatim into the inference scripts of the other models.

finetune.py is loaded as a module, as in MPP-Ti/debug_nan.py. Its main() only runs under
__name__ == "__main__", so nothing trains. This script therefore uses the exact model code and one-step
function of training.

Usage (from any directory): python DPOT-Ti/infer.py
"""

import csv
import importlib.util
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Lets ops missing from Apple's MPS backend fall back to the CPU. Must be set before importing torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "data"))
from benchmark_api import COMPLEXITY_SOURCES, DYNAMIC_TEST, IC_TEST, iter_test_grid  # noqa: E402


# ============================================================================= settings
VARIANTS = ["pretrained", "scratch"]
MIXES = ["Mix-simple", "Mix-balance", "Mix-complex"]
PDES = [
    "gray_scott",
    "wave",
    "burgers",
    "fisher_kpp",
    "swift_hohenberg",
    "decay",
    "kolmogorov",
    "kuramoto_sivashinsky",
]
# True: the first finished run only (1 variant, 1 mix, 1 PDE) and the first SMOKE_CELLS cells of its grid,
# saved under predictions/<model>_smoke/ so smoke outputs are never mistaken for real ones.
SMOKE_TEST = False
SMOKE_CELLS = 2


# ============================================================================= DPOT-Ti specific
# The only part of this file that changes between models: the folder name, the device, how a best.pt becomes
# a model, and the one-step function.
MODEL = "DPOT-Ti"


def pick_device():
    # cuda, else cpu, never MPS: DPOT's complex FFT aborts the process on Apple GPUs, so finetune.py's MPS test
    # (DPOT-Ti/finetune.py:206-216) cannot fall back to the CPU. On CUDA this is training's choice
    # (DPOT-Ti/finetune.py:204-205).
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt, device):
    """The fine-tuned model, rebuilt as finetune.py's docstring says (DPOT-Ti/finetune.py:12-14): DPOTNet with
    the run's own config, then the run's weights. The config must be training's unchanged DPOT-Ti
    architecture (DPOT-Ti/finetune.py:117-135), the one finetune.py's predict pads and resamples for."""
    assert ckpt["dpot_config"] == ft.DPOT_CONFIG, "dpot_config in best.pt differs from DPOT-Ti/finetune.py"
    assert ckpt["dpot_commit"] == ft.DPOT_COMMIT, "best.pt was trained with another DPOT commit"
    model = ft.DPOTNet(**ckpt["dpot_config"])
    model.load_state_dict(ckpt["model_state_dict"])  # strict: every key and every shape must match
    assert 1 <= ckpt["n_channels"] <= ft.MODEL_CHANNELS
    assert all(p.dtype == torch.float32 for p in model.parameters())
    return model.to(device).eval()  # eval mode, as finetune.py's evaluate


def make_step(model, ckpt):
    """One step in normalised space: window (N, INPUT_FRAMES, C, 64, 64) -> next frame (N, C, 64, 64).

    NOT IN PAPER: exactly the step of training, finetune.py's predict (DPOT-Ti/finetune.py:380-399), on all 5
    frames of the current window. At every step it:
      - Fourier-upsamples the window 64 -> 128 (DPOT-Ti/finetune.py:167-200, 382);
      - pads it to 10 frames by repeating its first (oldest) frame (DPOT-Ti/finetune.py:140-146, 383-384);
      - pads it to 4 channels with 1.0, built anew at every step (DPOT-Ti/finetune.py:148-151, 385-387). The
        window only holds the C real channels, so the predicted padding channels are never fed back;
      - runs DPOTNet with noise_scale 0, i.e. no input noise: the model adds none itself
        (DPOT/models/dpot.py:364-403), DPOT's scripts add it to the inputs before the call
        (DPOT/finetune.py:218), and training added none (DPOT-Ti/finetune.py:491-494);
      - keeps pred[..., 0, :C], the single output step and the C real channels, ignores cls_pred, and
        Fourier-truncates the prediction 128 -> 64 (DPOT-Ti/finetune.py:391-399).
    The adaptation saved in best.pt must be the one finetune.py's predict applies."""
    for key, value in [
        ("frames_used", ft.FRAMES_USED),
        ("time_padding", ft.TIME_PADDING),
        ("channel_pad_value", ft.CHANNEL_PAD_VALUE),
        ("resampling", ft.RESAMPLING),
    ]:
        assert ckpt[key] == value, f"{key} is {ckpt[key]!r} in best.pt, {value!r} in DPOT-Ti/finetune.py"
    frames = ckpt["frames_used"]
    assert frames == list(range(INPUT_FRAMES)), f"best.pt uses input frames {frames}, not all {INPUT_FRAMES}"

    def step(window):
        return ft.predict(model, window[:, frames])

    return step


# ============================================================================= generic (same for every model)
DATA_ROOT = REPO_ROOT / "data" / "Data"
CKPT_ROOT = REPO_ROOT / "checkpoint" / MODEL  # the real runs, also in a smoke test
OUT_ROOT = REPO_ROOT / "predictions" / (f"{MODEL}_smoke" if SMOKE_TEST else MODEL)
MANIFEST = OUT_ROOT / "manifest.csv"

# From the paper and the dataset (do not change): 5 input frames, then 10 in-horizon + 5 OOD rollout frames
# (paper, Figure 1), on the 50 test trajectories of every cell (paper, Appendix A).
INPUT_FRAMES = 5
TARGET_FRAMES = 15
N_TRAJECTORIES = 50
RESOLUTION = 64

MANIFEST_COLUMNS = [
    "model",
    "variant",
    "mix",
    "pde",
    "dynamic",
    "ic",
    "shift_group",
    "n_trajectories",
    "n_frames",
    "n_channels",
    "sample_index_ok",
    "n_nonfinite",
    "path",
]


def load_finetune():
    # Definitions and setup only: finetune.py's main() is behind __name__ == "__main__".
    spec = importlib.util.spec_from_file_location("finetune", HERE / "finetune.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ft = load_finetune()
assert ft.INPUT_FRAMES == INPUT_FRAMES


# Shift group of a test cell (paper, Section 3.3 and Table 2 caption). The train-seen cells are the 3
# (dynamic, IC) pairs the train mixes are built from (data/benchmark_api.py:49-53). "OOD" in a name marks a
# level outside the training range (data/benchmark_api.py:25-39).
TRAIN_SEEN = set(COMPLEXITY_SOURCES.values())


def shift_group(dynamic, ic):
    if (dynamic, ic) in TRAIN_SEEN:
        return "train-seen"
    dynamic_ood, ic_ood = "OOD" in dynamic, "OOD" in ic
    if dynamic_ood and ic_ood:
        return "joint-OOD"
    if dynamic_ood:
        return "dynamic-OOD"
    if ic_ood:
        return "IC-OOD"
    return "compositional-ID"


assert Counter(shift_group(d, i) for d in DYNAMIC_TEST for i in IC_TEST) == {
    "train-seen": 3,
    "compositional-ID": 6,
    "dynamic-OOD": 6,
    "IC-OOD": 6,
    "joint-OOD": 4,
}


def set_strict_fp32():
    # NOT IN PAPER: strict fp32, as in training: no autocast and no TF32 (TF32 only exists on Ampere and newer).
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def find_runs():
    """Splits the (variant, mix, PDE) runs into finished ones and skipped ones, with the reason."""
    finished, skipped = [], []
    for variant in VARIANTS:
        for mix in MIXES:
            for pde in PDES:
                run_dir = CKPT_ROOT / variant / mix / pde
                if ft.is_finished(run_dir):
                    finished.append((variant, mix, pde))
                elif (run_dir / "last.pt").exists():
                    skipped.append((variant, mix, pde, "training not finished (last.pt exists)"))
                else:
                    skipped.append((variant, mix, pde, "no checkpoint"))
    return finished, skipped


def load_checkpoint(run_dir, variant, mix, pde):
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    assert (ckpt["variant"], ckpt["mix"], ckpt["pde"]) == (variant, mix, pde), f"{run_dir}: best.pt is another run"
    print(
        f"  best.pt: epoch {ckpt['epoch']}, val loss {ckpt['val_loss']:.4e}, {ckpt['n_channels']} channels, "
        f"mean {np.round(ckpt['norm_mean'], 4).tolist()}, std {np.round(ckpt['norm_std'], 4).tolist()}"
    )
    return ckpt


def norm_constants(ckpt):
    # Per channel, from the run's train windows; model space = (field - mean) / std (NOT_IN_PAPER.md, 1.4).
    return np.asarray(ckpt["norm_mean"], dtype=np.float64), np.asarray(ckpt["norm_std"], dtype=np.float64)


def normalise(x, ckpt):
    # Exactly as finetune.py's load_windows: float64 arithmetic, then float32.
    mean, std = norm_constants(ckpt)
    return torch.from_numpy((x - mean[:, None, None]) / std[:, None, None]).float()


def denormalise(pred, ckpt):
    # The inverse, also in float64, then float32 for saving. Non-finite values go through unchanged.
    mean, std = norm_constants(ckpt)
    return (pred.astype(np.float64) * std[:, None, None] + mean[:, None, None]).astype(np.float32)


def check_test_dataset(ds, dynamic, ic):
    """One window per trajectory, starting at frame 0, and trajectories in file order (sample_index 0..49),
    so row k of the saved array is test trajectory k. Only the entry list is read, not the arrays."""
    assert ds.starts == [0], f"{dynamic}/{ic}: window starts {ds.starts}"
    assert len(ds) == N_TRAJECTORIES, f"{dynamic}/{ic}: {len(ds)} trajectories"
    sample_index_ok = [entry.sample_index for entry in ds.entries] == list(range(N_TRAJECTORIES))
    assert sample_index_ok, f"{dynamic}/{ic}: trajectories are not in order 0..{N_TRAJECTORIES - 1}"
    return sample_index_ok


def load_inputs(ds, dynamic, ic, n_channels):
    """Frames 0-4 of every trajectory, (N, INPUT_FRAMES, C, 64, 64) float32 in raw units. Only "x" is kept:
    the target frames 5-19 never reach the rollout."""
    inputs = []
    for k in range(len(ds)):
        item = ds[k]
        assert item["sample_index"] == k and item["window_start"] == 0
        assert (item["dynamic_name"], item["ic_name"]) == (dynamic, ic)
        x = item["x"]
        assert x.shape == (INPUT_FRAMES, n_channels, RESOLUTION, RESOLUTION) and x.dtype == np.float32
        inputs.append(x)
    return np.stack(inputs)


@torch.no_grad()
def rollout(step, x, device):
    """x: normalised frames 0-4, (N, INPUT_FRAMES, C, H, W), on the CPU. Returns the normalised predictions of
    frames 5-19, (N, TARGET_FRAMES, C, H, W), as a NumPy array.

    NOT IN PAPER: autoregressive rollout. Each predicted frame is appended to the window and the oldest frame
    is dropped, so after frame 4 the model only sees its own predictions. Outputs are never clipped or
    changed, and non-finite values are fed back as they are."""
    outputs = []
    for i in range(0, len(x), ft.BATCH_SIZE):  # NOT IN PAPER: batches of 16 trajectories, the training batch size
        window = x[i : i + ft.BATCH_SIZE].to(device)
        frames = []
        for _ in range(TARGET_FRAMES):
            next_frame = step(window)
            assert next_frame.shape == window[:, -1].shape and next_frame.dtype == torch.float32
            frames.append(next_frame)
            window = torch.cat([window[:, 1:], next_frame[:, None]], dim=1)
        outputs.append(torch.stack(frames, dim=1).cpu())
    return torch.cat(outputs).numpy()


def atomic_save_npy(array, path):
    # Write to a temporary file first, so a crash never leaves a half-written .npy that a resume would skip.
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:  # through a file object, so np.save does not add ".npy" to the name
        np.save(f, array)
    os.replace(tmp, path)


def manifest_row(variant, mix, pde, dynamic, ic, predictions, sample_index_ok, path):
    n_trajectories, n_frames, n_channels = predictions.shape[:3]
    return {
        "model": MODEL,
        "variant": variant,
        "mix": mix,
        "pde": pde,
        "dynamic": dynamic,
        "ic": ic,
        "shift_group": shift_group(dynamic, ic),
        "n_trajectories": n_trajectories,
        "n_frames": n_frames,
        "n_channels": n_channels,
        "sample_index_ok": sample_index_ok,
        "n_nonfinite": int((~np.isfinite(predictions)).sum()),
        "path": path.relative_to(REPO_ROOT).as_posix(),
    }


def read_manifest():
    """Rows of an earlier manifest.csv whose .npy still exists, keyed by path."""
    if not MANIFEST.exists():
        return {}
    with open(MANIFEST, newline="") as f:
        rows = {row["path"]: row for row in csv.DictReader(f)}
    return {path: row for path, row in rows.items() if (REPO_ROOT / path).exists()}


def write_manifest(rows):
    def order(row):
        return row["variant"], row["mix"], row["pde"], DYNAMIC_TEST.index(row["dynamic"]), IC_TEST.index(row["ic"])

    tmp = MANIFEST.with_name(MANIFEST.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=order))
    os.replace(tmp, MANIFEST)


def infer_run(variant, mix, pde, device, manifest, totals):
    run_dir, out_dir = CKPT_ROOT / variant / mix / pde, OUT_ROOT / variant / mix / pde
    print(f"\n=== {variant} / {mix} / {pde}")
    cells = list(iter_test_grid(DATA_ROOT, pde, input_frames=INPUT_FRAMES, target_frames=TARGET_FRAMES, count=None))
    assert [(d, i) for d, i, _ in cells] == [(d, i) for d in DYNAMIC_TEST for i in IC_TEST]
    if SMOKE_TEST:
        cells = cells[:SMOKE_CELLS]

    ckpt = step = None  # loaded only when a cell still needs predictions
    for dynamic, ic, ds in cells:
        label = f"  {dynamic:17s} {ic:14s} {shift_group(dynamic, ic):16s}"
        sample_index_ok = check_test_dataset(ds, dynamic, ic)
        path = out_dir / f"{dynamic}__{ic}.npy"
        key = path.relative_to(REPO_ROOT).as_posix()
        if path.exists():
            if key not in manifest:  # saved by a job that stopped before it updated the manifest
                predictions = np.load(path, mmap_mode="r")
                manifest[key] = manifest_row(variant, mix, pde, dynamic, ic, predictions, sample_index_ok, path)
                write_manifest(manifest)
            totals["existing"] += 1
            print(f"{label} already saved, skipping")
            continue

        if step is None:
            ckpt = load_checkpoint(run_dir, variant, mix, pde)
            step = make_step(load_model(ckpt, device), ckpt)
        start = time.time()
        n_channels = ckpt["n_channels"]
        x = normalise(load_inputs(ds, dynamic, ic, n_channels), ckpt)
        predictions = denormalise(rollout(step, x, device), ckpt)
        assert predictions.shape == (N_TRAJECTORIES, TARGET_FRAMES, n_channels, RESOLUTION, RESOLUTION)
        assert predictions.dtype == np.float32

        # NOT IN PAPER: non-finite values are saved as they are and only counted.
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_save_npy(predictions, path)
        manifest[key] = row = manifest_row(variant, mix, pde, dynamic, ic, predictions, sample_index_ok, path)
        write_manifest(manifest)
        totals["written"] += 1
        totals["nonfinite_values"] += row["n_nonfinite"]
        totals["nonfinite_cells"] += row["n_nonfinite"] > 0
        nonfinite = f"  {row['n_nonfinite']} non-finite values" if row["n_nonfinite"] else ""
        print(f"{label} {time.time() - start:5.1f}s{nonfinite}")


def print_skipped(skipped):
    if skipped:
        print(f"Skipped runs ({len(skipped)}), training not finished:")
        for variant, mix, pde, reason in skipped:
            print(f"  {variant} / {mix} / {pde}: {reason}")


def main():
    set_strict_fp32()
    device = pick_device()
    finished, skipped = find_runs()
    print(f"Device: {device}" + ("  (SMOKE TEST)" if SMOKE_TEST else ""))
    print(f"Runs: {len(finished)} finished of {len(finished) + len(skipped)}, predictions in {OUT_ROOT}")
    print_skipped(skipped)
    if SMOKE_TEST:
        assert finished, "smoke test: no finished run"
        finished = finished[:1]

    start = time.time()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest()
    totals = Counter()
    for variant, mix, pde in finished:
        infer_run(variant, mix, pde, device, manifest, totals)
    write_manifest(manifest)

    print(
        f"\nDone in {(time.time() - start) / 60:.1f} min: {totals['written']} cells predicted, "
        f"{totals['existing']} already saved. Non-finite values: {totals['nonfinite_values']} "
        f"in {totals['nonfinite_cells']} of the {totals['written']} new cells."
    )
    print(f"Manifest: {MANIFEST} ({len(manifest)} rows)")
    print_skipped(skipped)


if __name__ == "__main__":
    main()

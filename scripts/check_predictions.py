"""Read-only numerical check of the saved predictions against the ground truth. No model, no GPU; nothing is
written.

For the 4 models in predictions/, variant pretrained, mix Mix-balance, all 8 PDEs and the 3 train-seen cells,
it computes the paper's relative L2 error (Chu et al., arXiv:2605.29283, eq. 1) at prediction step 1 and
prints it next to the paper's value (Figure 23: S variant, Mix-balance, mean over the 3 train-seen cells,
frame 1), with the persistence baseline (the last input frame used as the prediction).

Alignment check: the error of prediction step 1 against target step 2 (one frame later). If the predictions
were one frame off, it would be smaller than the error against target step 1.

Error of trajectory i at frame t: ||pred_i,t - true_i,t||_2 / ||true_i,t||_2 over the whole field (all
channels and grid points), in float64, then averaged over the 50 trajectories of each of the 3 cells, i.e.
over 150 values. A trajectory whose step-1 prediction has a non-finite value is skipped and counted.

Ground truth: read with iter_test_grid as the infer.py scripts read it (same data root, same frames, same
order, sample_index 0..49 checked).

Usage, from the repo root on Lab-IA (CPU only):
    sbatch --time=01:00:00 --cpus-per-task=4 --job-name=check-pred --output=logs/%x-%j.out \
        --wrap='. /mnt/beegfs/projects/ftctinfer/conda_stagiaires/etc/profile.d/conda.sh && conda activate "$SLURM_SUBMIT_DIR/.venv-labia" && cd "$SLURM_SUBMIT_DIR" && export PYTHONUNBUFFERED=1 && python scripts/check_predictions.py'
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data"))
from benchmark_api import COMPLEXITY_SOURCES, DYNAMIC_TEST, IC_TEST, iter_test_grid  # noqa: E402

DATA_ROOT = REPO_ROOT / "data" / "Data"
PRED_ROOT = REPO_ROOT / "predictions"
VARIANT, MIX = "pretrained", "Mix-balance"

# As in the infer.py scripts.
INPUT_FRAMES = 5
TARGET_FRAMES = 15
N_TRAJECTORIES = 50
RESOLUTION = 64
TRAIN_SEEN = set(COMPLEXITY_SOURCES.values())  # (Dynamic-small, IC-simple), (medium, medium), (large, complex)

# Paper name -> folder in predictions/, in the column order of the paper's Figure 23.
MODELS = {"DPOT": "DPOT-Ti", "Poseidon": "Poseidon-T", "GPhyT": "GPhyT-S", "MPP": "MPP-Ti"}
# (paper label, our PDE name), in the row order of Figure 23.
PDES = [
    ("F-KPP", "fisher_kpp"),
    ("G-S", "gray_scott"),
    ("S-H", "swift_hohenberg"),
    ("Burgers", "burgers"),
    ("Kolm", "kolmogorov"),
    ("K-S", "kuramoto_sivashinsky"),
    ("Decay", "decay"),
    ("Wave", "wave"),
]
# Paper, Figure 23: raw relative L2 error at prediction frame 1, S variant, Mix-balance, averaged over the 3
# train-seen cells. Same order as PDES.
PAPER = {
    "DPOT": [0.003, 0.008, 0.005, 0.031, 0.048, 0.010, 0.016, 0.027],
    "Poseidon": [0.012, 0.036, 0.033, 0.085, 0.054, 0.043, 0.052, 0.13],
    "GPhyT": [0.003, 0.020, 0.016, 0.062, 0.087, 0.061, 0.059, 0.086],
    "MPP": [0.008, 0.030, 0.035, 0.12, 0.11, 0.094, 0.079, 0.20],
}


def relative_l2(pred, true):
    """(N, C, H, W) pairs -> (N,) ||pred - true||_2 / ||true||_2 over the whole field, in float64 (eq. 1)."""
    pred, true = pred.astype(np.float64), true.astype(np.float64)
    return np.sqrt(((pred - true) ** 2).sum(axis=(1, 2, 3)) / (true**2).sum(axis=(1, 2, 3)))


def load_ground_truth(pde):
    """The 3 train-seen cells, in grid order: {(dynamic, ic): (last input frame (50, C, H, W), target frames 1-2
    (50, 2, C, H, W))}, raw units. Read and checked as the infer.py scripts read the test grid."""
    cells = list(iter_test_grid(DATA_ROOT, pde, input_frames=INPUT_FRAMES, target_frames=TARGET_FRAMES, count=None))
    assert [(d, i) for d, i, _ in cells] == [(d, i) for d in DYNAMIC_TEST for i in IC_TEST]
    truth = {}
    for dynamic, ic, ds in cells:
        if (dynamic, ic) not in TRAIN_SEEN:
            continue
        # The checks of the infer.py scripts (check_test_dataset, load_inputs).
        assert ds.starts == [0], f"{pde} {dynamic}/{ic}: window starts {ds.starts}"
        assert len(ds) == N_TRAJECTORIES, f"{pde} {dynamic}/{ic}: {len(ds)} trajectories"
        assert [entry.sample_index for entry in ds.entries] == list(range(N_TRAJECTORIES))
        last_inputs, targets = [], []
        for k in range(len(ds)):
            item = ds[k]
            assert item["sample_index"] == k and item["window_start"] == 0
            assert (item["dynamic_name"], item["ic_name"]) == (dynamic, ic)
            x, y = item["x"], item["y"]
            assert x.shape[0] == INPUT_FRAMES and y.shape[0] == TARGET_FRAMES
            assert x.shape[2:] == y.shape[2:] == (RESOLUTION, RESOLUTION)
            last_inputs.append(x[-1])
            targets.append(y[:2])
        last_inputs, targets = np.stack(last_inputs), np.stack(targets)
        assert np.isfinite(last_inputs).all() and np.isfinite(targets).all(), f"{pde} {dynamic}/{ic}"
        assert (np.abs(targets).max(axis=(2, 3, 4)) > 0).all(), f"{pde} {dynamic}/{ic}: a target frame is all zero"
        truth[(dynamic, ic)] = (last_inputs, targets)
    assert len(truth) == len(TRAIN_SEEN)
    return truth


def model_errors(model_dir, pde, truth, missing):
    """Step-1 errors of one model on one PDE, pooled over the 3 cells: (mean vs target 1, mean vs target 2,
    trajectories skipped). None if a prediction file is missing (its path is added to missing)."""
    step1 = {}
    for (dynamic, ic), (_, targets) in truth.items():
        path = PRED_ROOT / model_dir / VARIANT / MIX / pde / f"{dynamic}__{ic}.npy"
        if not path.exists():
            missing.append(path.relative_to(REPO_ROOT).as_posix())
            continue
        pred = np.load(path, mmap_mode="r")
        expected = (N_TRAJECTORIES, TARGET_FRAMES, targets.shape[2], RESOLUTION, RESOLUTION)
        assert pred.shape == expected and pred.dtype == np.float32, f"{path}: {pred.shape} {pred.dtype}"
        step1[(dynamic, ic)] = np.array(pred[:, 0])  # prediction step 1 = sequence frame 5
    if len(step1) < len(truth):
        return None
    errors_t1, errors_t2, finite = [], [], []
    for cell, (_, targets) in truth.items():
        errors_t1.append(relative_l2(step1[cell], targets[:, 0]))
        errors_t2.append(relative_l2(step1[cell], targets[:, 1]))
        finite.append(np.isfinite(step1[cell]).all(axis=(1, 2, 3)))
    errors_t1, errors_t2, finite = np.concatenate(errors_t1), np.concatenate(errors_t2), np.concatenate(finite)
    if not finite.any():
        return np.nan, np.nan, len(finite)
    return errors_t1[finite].mean(), errors_t2[finite].mean(), int((~finite).sum())


def main():
    results, persistence, missing = {}, {}, []
    for _, pde in PDES:
        print(f"reading {pde} ...")
        truth = load_ground_truth(pde)
        persistence[pde] = np.concatenate([relative_l2(last, targets[:, 0]) for last, targets in truth.values()]).mean()
        for name, model_dir in MODELS.items():
            results[name, pde] = model_errors(model_dir, pde, truth, missing)

    n_total = len(TRAIN_SEEN) * N_TRAJECTORIES
    print(
        f"\nRelative L2 error at prediction step 1 (eq. 1), {VARIANT}, {MIX}, mean over the {len(TRAIN_SEEN)} "
        f"train-seen cells x {N_TRAJECTORIES} trajectories.\n"
        "ours = our predictions; paper = Figure 23 (S variant); persistence = last input frame as the prediction.\n"
    )
    header = f"{'PDE':9s}" + "".join(f"{name:>18s}" for name in MODELS) + f"{'persistence':>14s}"
    print(header)
    print(f"{'':9s}" + f"{'ours':>10s}{'paper':>8s}" * len(MODELS))
    print("-" * len(header))
    for k, (label, pde) in enumerate(PDES):
        row = f"{label:9s}"
        for name in MODELS:
            result = results[name, pde]
            ours = "missing" if result is None else f"{result[0]:.4f}"
            row += f"{ours:>10s}{PAPER[name][k]:>8g}"
        print(row + f"{persistence[pde]:>14.4f}")

    print(
        "\nAlignment: prediction step 1 against target step 1 (t1) and against target step 2 (t2, one frame later),"
        f"\nand trajectories skipped because their step-1 prediction is non-finite (of {n_total}).\n"
    )
    header = f"{'PDE':9s}" + "".join(f"{name:>24s}" for name in MODELS)
    print(header)
    print(f"{'':9s}" + f"{'t1':>10s}{'t2':>8s}{'skip':>6s}" * len(MODELS))
    print("-" * len(header))
    for label, pde in PDES:
        row = f"{label:9s}"
        for name in MODELS:
            result = results[name, pde]
            if result is None:
                row += f"{'missing':>24s}"
            else:
                row += f"{result[0]:>10.4f}{result[1]:>8.4f}{result[2]:>6d}"
        print(row)

    if missing:
        print(f"\nMissing prediction files ({len(missing)}):")
        for path in missing:
            print(f"  {path}")


if __name__ == "__main__":
    main()

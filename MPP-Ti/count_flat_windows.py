"""Counts, for every PDE x mix, the train windows MPP-Ti/finetune.py leaves out of training as flat, and, for
information only, the val windows that are flat too (validation keeps every window) and the train windows whose
flatness score is below 1e-3, 3e-3, 1e-2 and 3e-2. No training.

Usage, from the repo root on Lab-IA (CPU only; loads the 24 (PDE, mix) data sets one after the other):
    sbatch --time=02:00:00 --cpus-per-task=4 --job-name=count-flat --output=logs/%x-%j.out \
        --wrap='. /mnt/beegfs/projects/ftctinfer/conda_stagiaires/etc/profile.d/conda.sh && conda activate "$SLURM_SUBMIT_DIR/.venv-labia" && cd "$SLURM_SUBMIT_DIR" && export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 && python MPP-Ti/count_flat_windows.py'

finetune.py is loaded as a module, as in debug_nan.py, so its own load_windows (data and normalisation) and
flat_windows (criterion and FLAT_STD_THRESHOLD) are used. main() only runs under __main__: nothing trains.
"""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_finetune():
    spec = importlib.util.spec_from_file_location("mpp_finetune", HERE / "finetune.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # definitions and setup only: main() is behind __name__ == "__main__"
    return module


ft = load_finetune()
assert not ft.SMOKE_TEST, "set SMOKE_TEST = False in MPP-Ti/finetune.py: with it, only 1 PDE and 48 windows load"


SCORE_THRESHOLDS = [1e-3, 3e-3, 1e-2, 3e-2]  # extra train-window counts by flatness score, for information


def count(n_flat, n_total):
    return f"{n_flat:5d} / {n_total:4d} ({100 * n_flat / n_total:5.2f}%)"


def short(n_flat, n_total):
    return f"{n_flat:4d} ({100 * n_flat / n_total:4.1f}%)"


def main():
    print(
        f"Flat window: one of its {ft.INPUT_FRAMES} input frames has a spatial std below {ft.FLAT_STD_THRESHOLD:g} "
        f"in every channel at once, in our normalised units (per PDE x mix, from the train windows).\n"
        f"Flatness score: min over the input frames of the max over channels of the frame's spatial std, so a "
        f"window is flat when its score is below {ft.FLAT_STD_THRESHOLD:g}. The last columns count the train "
        f"windows whose score is below each value.\n"
    )
    score_cols = "".join(f"{'score < ' + format(t, 'g'):>14s}" for t in SCORE_THRESHOLDS)
    header = f"{'PDE':22s} {'mix':12s} {'train windows excluded':>26s} {'val windows flat (kept)':>26s}{score_cols}"
    print(header)
    print("-" * len(header))
    totals = [0, 0, 0, 0]  # train flat, train total, val flat, val total
    score_totals = [0] * len(SCORE_THRESHOLDS)
    for pde in ft.PDES:
        for mix in ft.MIXES:
            data = ft.load_windows(pde, mix)
            scores = ft.flatness_scores(data["x_train"])
            train_flat = int(ft.flat_windows(data["x_train"]).sum())
            val_flat = int(ft.flat_windows(data["x_val"]).sum())
            n_train, n_val = len(data["x_train"]), len(data["x_val"])
            below = [int((scores < t).sum()) for t in SCORE_THRESHOLDS]
            print(f"{pde:22s} {mix:12s} {count(train_flat, n_train):>26s} {count(val_flat, n_val):>26s}"
                  + "".join(f"{short(n, n_train):>14s}" for n in below))
            for k, value in enumerate([train_flat, n_train, val_flat, n_val]):
                totals[k] += value
            for k, n in enumerate(below):
                score_totals[k] += n
            del data
    print("-" * len(header))
    print(f"{'all':22s} {'':12s} {count(totals[0], totals[1]):>26s} {count(totals[2], totals[3]):>26s}"
          + "".join(f"{short(n, totals[1]):>14s}" for n in score_totals))


if __name__ == "__main__":
    main()

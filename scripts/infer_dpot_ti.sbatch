#!/bin/bash
# Autoregressive inference of every finished DPOT-Ti run on the PhysBiasBench test grid (predictions
# only, no evaluation), with DPOT-Ti/infer.py.
#
# Submit from the repo root:
#   sbatch scripts/infer_dpot_ti.sbatch
# Runs whose training is not finished are listed in the log and skipped. If the job hits the 12 h limit, or
# once more runs have finished, submit the same command again: cells whose .npy already exists are skipped.
# Smoke test: set SMOKE_TEST = True in DPOT-Ti/infer.py, then submit with
#   sbatch --qos=debug --partition=testing --time=00:30:00 scripts/infer_dpot_ti.sbatch
# Predictions go to predictions/DPOT-Ti/ (predictions/DPOT-Ti_smoke/ for the smoke test).
# The log goes to logs/<job name>-<job id>.out.
#
#SBATCH --job-name=infer-dpot-ti
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# conda's shell functions read variables that may be unset, so -u is off while they run.
set +u
source /mnt/beegfs/projects/ftctinfer/conda_stagiaires/etc/profile.d/conda.sh
conda activate "$SLURM_SUBMIT_DIR/.venv-labia"
set -u

cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1  # nothing is downloaded during jobs
export PYTHONUNBUFFERED=1  # print progress to the log as it happens

nvidia-smi
python DPOT-Ti/infer.py

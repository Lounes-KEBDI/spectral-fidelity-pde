#!/bin/bash
# Fine-tunes Poseidon-T on PhysBiasBench (48 runs) on Lab-IA, with Poseidon-T/finetune.py.
#
# Submit from the repo root:
#   sbatch scripts/finetune_poseidon_t.sbatch
# If the job hits the 24 h limit, submit the same command again: finished runs are skipped and the
# unfinished one resumes from its last.pt.
# Smoke test: set SMOKE_TEST = True in Poseidon-T/finetune.py, then submit with
#   sbatch --qos=debug --partition=testing --time=00:30:00 scripts/finetune_poseidon_t.sbatch
# The log goes to logs/<job name>-<job id>.out.
#
#SBATCH --job-name=ft-poseidon-t
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
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
python Poseidon-T/finetune.py

#!/usr/bin/env bash
# Creates the Python environment for the 4 fine-tuning scripts. Run once on the Lab-IA login node,
# from anywhere:   bash scripts/setup_env.sh
# The environment lives inside the repo, at <repo root>/.venv-labia. It is never committed (.gitignore)
# and never copied back to the Mac (scripts/from_cluster.sh excludes .venv*/).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="$REPO_ROOT/.venv-labia"
# Shared conda install: never install anything into its base, never create environments inside it.
CONDA_BASE="/mnt/beegfs/projects/ftctinfer/conda_stagiaires"

# Internet on Lab-IA goes through this proxy. It is normally already set on the login node.
if [ -z "${http_proxy:-}" ]; then
    export http_proxy="http://webproxy.lab-ia.fr:8080"
    export https_proxy="$http_proxy"
    export HTTP_PROXY="$http_proxy"
    export HTTPS_PROXY="$http_proxy"
fi

# Keep conda's package cache in your home rather than in the shared install.
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$HOME/.conda/pkgs}"

# conda's shell functions read variables that may be unset, so -u is off while they run.
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
if [ ! -d "$ENV_PREFIX" ]; then
    conda create -p "$ENV_PREFIX" python=3.9 -y
fi
conda activate "$ENV_PREFIX"
set -u

python -m pip install --no-cache-dir --prefer-binary -r "$REPO_ROOT/requirements.txt"

# Check every import the 4 finetune scripts need.
python - <<'EOF'
import importlib

for name in ["torch", "torchvision", "numpy", "einops", "timm", "transformers", "safetensors",
             "huggingface_hub", "scOT.model"]:
    module = importlib.import_module(name)
    print(f"  ok  {name:16s} {getattr(module, '__version__', '')}")

import torch

print(f"torch {torch.__version__}, built for CUDA {torch.version.cuda}")
print(f"torch.cuda.is_available() = {torch.cuda.is_available()} "
      "(False is expected on the login node, which has no GPU; jobs run on GPU nodes)")
EOF

echo
echo "Environment ready. Activate it with:"
echo "  source $CONDA_BASE/etc/profile.d/conda.sh && conda activate $ENV_PREFIX"

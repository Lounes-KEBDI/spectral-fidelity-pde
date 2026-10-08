#!/usr/bin/env bash
# Copies new or changed files (logs, ...) from Lab-IA back to the Mac. Run on the Mac, from
# anywhere:   bash scripts/from_cluster.sh
# --update never overwrites a local file that is newer than the cluster's copy, and nothing is deleted
# locally (no --delete). The cluster environment (.venv-labia) never comes back, and neither do checkpoint/
# and predictions/: they live on Hugging Face.
set -euo pipefail

REMOTE_HOST="labia"
REMOTE_DIR="spectral-fidelity-pde"  # relative to your home folder on the cluster

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rsync -avP --update \
    --exclude='.venv*/' \
    --exclude='/checkpoint/' \
    --exclude='/predictions/' \
    --exclude='__pycache__/' \
    --exclude='.DS_Store' \
    "$REMOTE_HOST:$REMOTE_DIR/" "$REPO_ROOT/"

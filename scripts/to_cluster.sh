#!/usr/bin/env bash
# Copies this repo from the Mac to Lab-IA. Run on the Mac, from anywhere:   bash scripts/to_cluster.sh
# Never deletes anything on the cluster (no --delete). Local environments and checkpoints are not sent.
set -euo pipefail

REMOTE_HOST="labia"
REMOTE_DIR="spectral-fidelity-pde"  # relative to your home folder on the cluster

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rsync -avP \
    --exclude='.venv*/' \
    --exclude='checkpoint/' \
    --exclude='__pycache__/' \
    --exclude='.DS_Store' \
    "$REPO_ROOT/" "$REMOTE_HOST:$REMOTE_DIR/"

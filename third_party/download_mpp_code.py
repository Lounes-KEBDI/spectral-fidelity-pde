# Clone the MPP code (github.com/PolymathicAI/multiple_physics_pretraining) at a fixed commit into third_party/MPP/.
# MPP has no setup.py, so it cannot be installed with pip: MPP-Ti/finetune.py puts third_party/MPP/models/
# on sys.path instead. Requires git.

import subprocess
from pathlib import Path

REPO_URL = "https://github.com/PolymathicAI/multiple_physics_pretraining.git"
COMMIT = "e751fc25ae5274c3ddc869bf73afd6fcd4163cb5"  # latest commit on main (2024-12-06)
DEST = Path(__file__).resolve().parent / "MPP"

if not DEST.exists():
    subprocess.run(["git", "clone", REPO_URL, str(DEST)], check=True)
subprocess.run(["git", "-C", str(DEST), "-c", "advice.detachedHead=false", "checkout", COMMIT], check=True)

print(f"MPP code at commit {COMMIT[:7]} in: {DEST}")

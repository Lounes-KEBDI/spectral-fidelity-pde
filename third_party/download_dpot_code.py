# Clone the DPOT code (github.com/thu-ml/DPOT) at a fixed commit into third_party/DPOT/.
# DPOT has no setup.py, so it cannot be installed with pip: DPOT-Ti/finetune.py loads
# third_party/DPOT/models/dpot.py directly. Requires git.

import subprocess
from pathlib import Path

REPO_URL = "https://github.com/thu-ml/DPOT.git"
COMMIT = "dcd2f9a9359765e19ad63e2f3f879a2a8ce1aa17"  # latest commit on main (2024-06-10)
DEST = Path(__file__).resolve().parent / "DPOT"

if not DEST.exists():
    subprocess.run(["git", "clone", REPO_URL, str(DEST)], check=True)
subprocess.run(["git", "-C", str(DEST), "-c", "advice.detachedHead=false", "checkout", COMMIT], check=True)

print(f"DPOT code at commit {COMMIT[:7]} in: {DEST}")

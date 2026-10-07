# Clone the GPhyT code (github.com/FloWsnr/General-Physics-Transformer) at a fixed commit into third_party/GPhyT/.
# Its pyproject.toml asks for Python >= 3.12 and lists no dependencies, so it is not pip-installed:
# GPhyT-S/finetune.py puts third_party/GPhyT/ on sys.path instead. Requires git.

import subprocess
from pathlib import Path

REPO_URL = "https://github.com/FloWsnr/General-Physics-Transformer.git"
COMMIT = "4374116e8c96db08f5b0691b04485959749d6374"  # 2025-10-08, the commit that published the weights
DEST = Path(__file__).resolve().parent / "GPhyT"

if not DEST.exists():
    subprocess.run(["git", "clone", REPO_URL, str(DEST)], check=True)
subprocess.run(["git", "-C", str(DEST), "-c", "advice.detachedHead=false", "checkout", COMMIT], check=True)

print(f"GPhyT code at commit {COMMIT[:7]} in: {DEST}")

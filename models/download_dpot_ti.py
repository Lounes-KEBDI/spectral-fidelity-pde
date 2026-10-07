# Download the DPOT-Ti pretrained weights from Hugging Face (hzk17/DPOT).
# Requires: pip install huggingface_hub

from pathlib import Path

from huggingface_hub import hf_hub_download

DEST = Path(__file__).resolve().parent / "DPOT-Ti"

hf_hub_download(repo_id="hzk17/DPOT", filename="model_Ti.pth", local_dir=DEST)

print(f"Weights downloaded to: {DEST}")

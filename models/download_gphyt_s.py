# Download the GPhyT-S pretrained weights from Hugging Face (flwi/Physics-Foundation-Model).
# Requires: pip install huggingface_hub

from pathlib import Path

from huggingface_hub import hf_hub_download

DEST = Path(__file__).resolve().parent / "GPhyT-S"

hf_hub_download(repo_id="flwi/Physics-Foundation-Model", filename="gphyt-S.pth", local_dir=DEST)

print(f"Weights downloaded to: {DEST}")

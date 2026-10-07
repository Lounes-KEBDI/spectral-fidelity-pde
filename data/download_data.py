# Download the PhysBiasBench dataset from Hugging Face
# (paper: "Do Physics Foundation Models Learn Generalizable Physics?", arXiv:2605.29283).
# The dataset is about 19.8 GB. Requires: pip install huggingface_hub

from pathlib import Path

from huggingface_hub import snapshot_download

DEST = Path(__file__).resolve().parent

snapshot_download(repo_id="90879c/PhysBiasBench", repo_type="dataset", local_dir=DEST)

print(f"Dataset downloaded to: {DEST}")

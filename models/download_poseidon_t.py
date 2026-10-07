# Download the Poseidon-T pretrained weights from Hugging Face (camlab-ethz/Poseidon-T).
# Requires: pip install huggingface_hub

from pathlib import Path

from huggingface_hub import hf_hub_download

DEST = Path(__file__).resolve().parent / "Poseidon-T"

for filename in ["config.json", "model.safetensors"]:
    hf_hub_download(repo_id="camlab-ethz/Poseidon-T", filename=filename, local_dir=DEST)

print(f"Weights downloaded to: {DEST}")

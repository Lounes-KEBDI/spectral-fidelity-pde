"""Fine-tune Poseidon-T on PhysBiasBench (training only: no inference, no evaluation).

Follows Chu et al., "Do Physics Foundation Models Learn Generalizable Physics?" (arXiv:2605.29283):
one-step prediction with teacher forcing on the dataset's own train/val windows
(5 input frames -> 1 target frame, taken from the first 15 frames of each 20-frame sequence).

48 independent runs = 2 variants x 3 mixtures x 8 PDEs:
  pretrained  Poseidon-T weights, embedding/recovery layers replaced for C channels, then fine-tuned
  scratch     the same adapted architecture with random weights, trained the same way

Outputs per run, in checkpoint/Poseidon-T/{variant}/{mix}/{pde}/:
  best.pt         best epoch (lowest val loss), enough to rebuild the model on its own:
                      cfg = ScOTConfig.from_dict(ckpt["scot_config"])
                      model = ScOT(cfg); model.load_state_dict(ckpt["model_state_dict"])
  last.pt         latest epoch + optimizer, scheduler and RNG states, for resuming (deleted at the end)
  train_log.csv   epoch, train_loss, val_loss, lr, seconds

Code references in comments:
  scOT/...          github.com/camlab-ethz/poseidon at commit b8fa28f59bd7f7673323f28d11a12c6f3a215c61
  transformers/...  huggingface/transformers v4.29.2 (the version Poseidon's pyproject.toml pins)
  benchmark_api.py  data/benchmark_api.py (shipped with the PhysBiasBench dataset)
Choices that are not in the paper are marked "NOT IN PAPER".

Usage (from any directory): python Poseidon-T/finetune.py
"""

import csv
import hashlib
import math
import os
import random
import sys
import time
from pathlib import Path

# Lets ops missing from Apple's MPS backend fall back to the CPU. Must be set before importing torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from scOT.model import ScOT, ScOTConfig
from torch.optim.lr_scheduler import LambdaLR

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data"))
from benchmark_api import make_train_val_datasets  # noqa: E402


# ============================================================================= settings
PDES = [
    "gray_scott",
    "wave",
    "burgers",
    "fisher_kpp",
    "swift_hohenberg",
    "decay",
    "kolmogorov",
    "kuramoto_sivashinsky",
]
MIXES = ["Mix-simple", "Mix-balance", "Mix-complex"]
# NOT IN PAPER: the paper's scratch baseline is M-sized; we only have Poseidon-T, so scratch is T-sized.
VARIANTS = ["pretrained", "scratch"]
# True: 1 PDE, 1 mix, 1 epoch, a few batches. Saved under checkpoint/Poseidon-T_smoke/ so a smoke
# run is never mistaken for a finished real run.
SMOKE_TEST = False
SMOKE_BATCHES = 3

DATA_ROOT = REPO_ROOT / "data" / "Data"
POSEIDON_DIR = REPO_ROOT / "models" / "Poseidon-T"  # config.json + model.safetensors
POSEIDON_COMMIT = "b8fa28f59bd7f7673323f28d11a12c6f3a215c61"
OUT_ROOT = REPO_ROOT / "checkpoint" / ("Poseidon-T_smoke" if SMOKE_TEST else "Poseidon-T")

# From the paper and the dataset (do not change).
INPUT_FRAMES = 5
TARGET_FRAMES = 1
TRAIN_CONTEXT_FRAMES = 15
WINDOW_STRIDE = 1
N_TRAIN_WINDOWS = 3000
N_VAL_WINDOWS = {"Mix-simple": 800, "Mix-balance": 750, "Mix-complex": 800}
RESOLUTION = 64

# NOT IN PAPER: one training recipe for all 48 runs.
LR = 1e-4  # a single learning rate for every parameter
WEIGHT_DECAY = 1e-4
WARMUP_FRACTION = 0.05
BATCH_SIZE = 16
MAX_EPOCHS = 50
PATIENCE = 10
SEED = 0

# NOT IN PAPER: scOT takes one frame (B, C, H, W) plus a time (B,) (scOT/model.py:1318-1321).
# We give it the last of the 5 input frames...
FRAME_USED = INPUT_FRAMES - 1
# NOT IN PAPER: ...with a constant lead time of one frame step. scOT's time is
# "frames elapsed / index of the trajectory's last frame", so a whole trajectory spans [0, 1]:
# t/20 for 21 snapshots (scOT/problems/fluids/incompressible.py:87, normalization_constants.py:6),
# t/19 for 20 snapshots (scOT/problems/reaction_diffusion/allen_cahn.py:23,33).
# PhysBiasBench sequences have 20 frames (benchmark_api.py:87-94), so one step is 1/19.
LEAD_TIME = 1.0 / 19.0

# Poseidon-T runs at 128x128 (config.json: image_size=128). When the input is smaller, scOT itself
# Fourier-upsamples it to 128 (scOT/model.py:1360-1366; zero-padded fft2 with norm="forward", which
# keeps amplitudes, scOT/model.py:1302-1316) and Fourier-truncates the prediction back to the input
# size (scOT/model.py:1416-1420, 1293-1300). So we feed 64x64, get 64x64 back and compute the loss
# there. We do no resampling of our own, so it is not done twice.
RESAMPLING = (
    "scOT internal: input 64->128 by Fourier zero-padding, output 128->64 by Fourier truncation "
    "(scOT/model.py:1293-1316, 1360-1366, 1416-1420)"
)

# The only tensors whose shape depends on the channel count C (scOT/model.py:282-284, 616-630).
# With ignore_mismatched_sizes=True they are skipped when loading (transformers/modeling_utils.py:
# 3010-3036) and keep PyTorch's default init (transformers/modeling_utils.py:122-135, 469-477,
# 2984-2986). embeddings.patch_embeddings.projection.bias has shape (48,) for any C, so it is loaded.
REPLACED_KEYS = {
    "embeddings.patch_embeddings.projection.weight",
    "patch_recovery.projection.weight",
    "patch_recovery.projection.bias",
    "patch_recovery.mixup.weight",
}

if SMOKE_TEST:
    PDES, MIXES, MAX_EPOCHS = PDES[:1], MIXES[:1], 1


# ============================================================================= helpers
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        # torch 2.0.1 (pinned by Poseidon) may not support, on MPS, the FFT/complex ops used by
        # scOT's resampling (scOT/model.py:1293-1316). Try them once and fall back to the CPU.
        try:
            x = torch.randn(1, 1, 8, 8, device="mps")
            ScOT._downsample(None, ScOT._upsample(None, x, 16), 8)
            return torch.device("mps")
        except Exception as err:
            print(f"MPS cannot run scOT's Fourier resampling ({type(err).__name__}), using the CPU.")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device


def get_rng_state(shuffle_gen):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "shuffle": shuffle_gen.get_state(),
    }


def set_rng_state(state, shuffle_gen):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    shuffle_gen.set_state(state["shuffle"])


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(obj, path):
    # Write to a temporary file first, so a crash never leaves a half-written checkpoint.
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def is_finished(run_dir):
    # last.pt exists from before epoch 1 until the run ends, so this only holds for finished runs.
    return (run_dir / "best.pt").exists() and not (run_dir / "last.pt").exists()


def cosine_with_warmup(warmup_steps, total_steps):
    # NOT IN PAPER: linear warmup, then cosine decay to 0 at MAX_EPOCHS (same shape as transformers'
    # get_cosine_schedule_with_warmup). Runs stopped early never reach the end of the cosine.
    def factor(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return factor


def write_log(path, history):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr", "seconds"])
        writer.writeheader()
        writer.writerows(history)


# ============================================================================= data
def load_windows(pde, mix):
    """Builds the dataset's train/val windows with the unmodified API, checks them, and returns the
    normalised (input frame, target frame) pairs as tensors plus the normalisation constants."""
    train_ds, val_ds = make_train_val_datasets(
        DATA_ROOT,
        pde,
        mix,
        input_frames=INPUT_FRAMES,
        target_frames=TARGET_FRAMES,
        train_context_frames=TRAIN_CONTEXT_FRAMES,
        window_stride=WINDOW_STRIDE,
    )
    assert len(train_ds) == N_TRAIN_WINDOWS, f"{pde}/{mix}: {len(train_ds)} train windows"
    assert len(val_ds) == N_VAL_WINDOWS[mix], f"{pde}/{mix}: {len(val_ds)} val windows"
    n_channels = train_ds[0]["x"].shape[1]
    assert n_channels in (1, 2), f"{pde}: {n_channels} channels"

    def collect(ds, n):
        inputs, targets = [], []
        total = np.zeros(n_channels)
        total_sq = np.zeros(n_channels)
        for i in range(n):
            item = ds[i]
            x, y = item["x"], item["y"]
            assert x.shape == (INPUT_FRAMES, n_channels, RESOLUTION, RESOLUTION) and x.dtype == np.float32
            assert y.shape == (TARGET_FRAMES, n_channels, RESOLUTION, RESOLUTION) and y.dtype == np.float32
            frames = np.concatenate([x, y]).astype(np.float64)
            total += frames.sum(axis=(0, 2, 3))
            total_sq += (frames**2).sum(axis=(0, 2, 3))
            inputs.append(x[FRAME_USED])
            targets.append(y[0])
        n_values = n * (INPUT_FRAMES + TARGET_FRAMES) * RESOLUTION * RESOLUTION
        return np.stack(inputs), np.stack(targets), total / n_values, total_sq / n_values

    n_train = SMOKE_BATCHES * BATCH_SIZE if SMOKE_TEST else len(train_ds)
    n_val = SMOKE_BATCHES * BATCH_SIZE if SMOKE_TEST else len(val_ds)
    x_train, y_train, mean, mean_sq = collect(train_ds, n_train)
    x_val, y_val, _, _ = collect(val_ds, n_val)

    # NOT IN PAPER: per-channel mean/std over every frame (x and y) of the TRAIN windows of this
    # (PDE, mix) only. The dataset API does not normalise, and neither does scOT inside the model
    # (Poseidon normalises in its dataset classes, e.g. scOT/problems/reaction_diffusion/
    # allen_cahn.py:46-47), so the data is normalised exactly once, here.
    std = np.sqrt(mean_sq - mean**2)
    assert (std > 0).all(), f"{pde}/{mix}: a channel is constant"

    def normalise(a):
        return torch.from_numpy((a - mean[:, None, None]) / std[:, None, None]).float()

    return {
        "n_channels": n_channels,
        "mean": mean,
        "std": std,
        "x_train": normalise(x_train),
        "y_train": normalise(y_train),
        "x_val": normalise(x_val),
        "y_val": normalise(y_val),
    }


# ============================================================================= model
def build_model(variant, n_channels):
    # The checkpoint's own config with only the channel counts changed (scOT/model.py:80-81). This
    # plays the role of the fresh ScOTConfig that train.py builds (scOT/train.py:247-275). Keyword
    # arguments overwrite the loaded attributes (transformers/configuration_utils.py:716-722).
    # Everything else stays as pretrained, e.g. image_size=128 and use_conditioning=True (time-
    # conditioned LayerNorms, scOT/model.py:143-160). channel_slice_list_normalized_loss only feeds
    # scOT's built-in loss (scOT/model.py:1432-1482), which we do not use, and the checkpoint's value
    # describes 4 channels, so it is cleared.
    cfg = ScOTConfig.from_pretrained(
        str(POSEIDON_DIR),
        num_channels=n_channels,
        num_out_channels=n_channels,
        channel_slice_list_normalized_loss=None,
    )
    if variant == "pretrained":
        # As scOT/train.py:330-333 with --replace_embedding_recovery: every tensor whose shape still
        # matches is loaded, the rest is re-initialised. output_loading_info returns the missing,
        # unexpected and mismatched keys (transformers/modeling_utils.py:2136, 2830-2838).
        model, info = ScOT.from_pretrained(
            str(POSEIDON_DIR), config=cfg, ignore_mismatched_sizes=True, output_loading_info=True
        )
        check_pretrained_load(model, info)
    else:
        model = ScOT(cfg)  # random init, as scOT/train.py:334-335
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,} ({n_params / 1e6:.1f}M, Poseidon-T is ~20.8M)")
    return model


def check_pretrained_load(model, info):
    mismatched = {key for key, _, _ in info["mismatched_keys"]}
    print(f"  missing keys: {info['missing_keys']}")
    print(f"  unexpected keys: {info['unexpected_keys']}")
    for key, ckpt_shape, model_shape in info["mismatched_keys"]:
        print(f"  replaced: {key} {tuple(ckpt_shape)} -> {tuple(model_shape)}")
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["error_msgs"]
    assert mismatched == REPLACED_KEYS, f"unexpected replaced layers: {sorted(mismatched)}"

    # Every other tensor must be exactly the one in model.safetensors.
    ckpt = load_file(str(POSEIDON_DIR / "model.safetensors"))
    state = model.state_dict()
    assert set(state) == set(ckpt)
    differ = {k for k in ckpt if state[k].shape != ckpt[k].shape or not torch.equal(state[k], ckpt[k])}
    assert differ == REPLACED_KEYS, f"tensors differing from the checkpoint: {sorted(differ)}"
    assert all(torch.isfinite(state[k]).all() for k in REPLACED_KEYS)


def predict(model, x):
    # (B, C, 64, 64) + lead time -> (B, C, 64, 64). scOT/model.py:1318-1321, output field: 1490-1492.
    lead_time = torch.full((x.shape[0],), LEAD_TIME, device=x.device)
    return model(pixel_values=x, time=lead_time).output


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    squared_error = 0.0
    for i in range(0, len(x), BATCH_SIZE):
        pred = predict(model, x[i : i + BATCH_SIZE])
        squared_error += F.mse_loss(pred, y[i : i + BATCH_SIZE], reduction="sum").item()
    return squared_error / y.numel()


# ============================================================================= checkpoints
def run_info(variant, mix, pde, data, model, weights_sha256):
    """Everything besides the weights that is needed to rebuild and use the model later."""
    return {
        "variant": variant,
        "pde": pde,
        "mix": mix,
        "n_channels": data["n_channels"],
        "scot_config": model.config.to_dict(),
        "frame_used": FRAME_USED,  # index into x (input_frames, C, H, W): the last input frame
        "lead_time": LEAD_TIME,
        "resampling": RESAMPLING,
        "norm_mean": data["mean"].tolist(),  # per channel; model space = (field - mean) / std
        "norm_std": data["std"].tolist(),
        "pretrained_weights_sha256": weights_sha256,
        "poseidon_commit": POSEIDON_COMMIT,
    }


def model_checkpoint(model, epoch, val_loss, info):
    weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return {"model_state_dict": weights, "epoch": epoch, "val_loss": val_loss, **info}


# ============================================================================= training
def train_run(variant, mix, pde, device, weights_sha256, timing):
    run_dir = OUT_ROOT / variant / mix / pde
    best_path, last_path, log_path = run_dir / "best.pt", run_dir / "last.pt", run_dir / "train_log.csv"
    print(f"\n=== {variant} / {mix} / {pde}")
    if is_finished(run_dir):
        print("  already finished, skipping")
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    data = load_windows(pde, mix)
    print(
        f"  channels: {data['n_channels']}, windows: {len(data['x_train'])} train / {len(data['x_val'])} val, "
        f"mean {np.round(data['mean'], 4).tolist()}, std {np.round(data['std'], 4).tolist()}"
    )
    seed_everything(SEED)  # NOT IN PAPER: seed 0 for every run
    model = build_model(variant, data["n_channels"]).to(device)
    info = run_info(variant, mix, pde, data, model, weights_sha256)

    # NOT IN PAPER: AdamW on all parameters, cosine schedule with a 5% linear warmup.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = math.ceil(len(data["x_train"]) / BATCH_SIZE) * MAX_EPOCHS
    scheduler = LambdaLR(optimizer, cosine_with_warmup(round(WARMUP_FRACTION * total_steps), total_steps))
    shuffle_gen = torch.Generator().manual_seed(SEED)
    progress = {"next_epoch": 1, "best_val": math.inf, "best_epoch": 0, "bad_epochs": 0, "history": []}

    def save_last(epoch, val_loss):
        last = model_checkpoint(model, epoch, val_loss, info)
        last["optimizer_state_dict"] = optimizer.state_dict()
        last["scheduler_state_dict"] = scheduler.state_dict()
        last["rng_state"] = get_rng_state(shuffle_gen)
        last["progress"] = progress
        atomic_save(last, last_path)

    if last_path.exists():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(last["model_state_dict"])
        optimizer.load_state_dict(last["optimizer_state_dict"])
        scheduler.load_state_dict(last["scheduler_state_dict"])
        set_rng_state(last["rng_state"], shuffle_gen)
        progress = last["progress"]
        print(f"  resuming from last.pt at epoch {progress['next_epoch']}")
    else:
        save_last(epoch=0, val_loss=None)  # before epoch 1, so is_finished() stays False until the end

    x_train, y_train = data["x_train"].to(device), data["y_train"].to(device)
    x_val, y_val = data["x_val"].to(device), data["y_val"].to(device)

    for epoch in range(progress["next_epoch"], MAX_EPOCHS + 1):
        if progress["bad_epochs"] >= PATIENCE:  # NOT IN PAPER: early stopping, patience 10
            break
        start = time.time()

        # One-step prediction with teacher forcing: the input is always a ground-truth frame.
        model.train()
        order = torch.randperm(len(x_train), generator=shuffle_gen).to(device)
        loss_sum = 0.0
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i : i + BATCH_SIZE]
            loss = F.mse_loss(predict(model, x_train[idx]), y_train[idx])  # NOT IN PAPER: MSE, normalised space
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(idx)
        train_loss = loss_sum / len(x_train)
        val_loss = evaluate(model, x_val, y_val)
        assert math.isfinite(train_loss) and math.isfinite(val_loss), "loss is not finite"
        seconds = time.time() - start
        lr = optimizer.param_groups[0]["lr"]

        progress["history"].append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr, "seconds": round(seconds, 1)}
        )
        write_log(log_path, progress["history"])
        print(f"  epoch {epoch:2d}/{MAX_EPOCHS}  train {train_loss:.4e}  val {val_loss:.4e}  lr {lr:.2e}  {seconds:.0f}s")

        if val_loss < progress["best_val"]:
            progress.update(best_val=val_loss, best_epoch=epoch, bad_epochs=0)
            atomic_save(model_checkpoint(model, epoch, val_loss, info), best_path)
        else:
            progress["bad_epochs"] += 1
        progress["next_epoch"] = epoch + 1
        save_last(epoch, val_loss)

        if not timing["printed"]:
            timing["printed"] = True
            hours = seconds * MAX_EPOCHS * timing["runs_left"] / 3600
            print(
                f"  [time] {seconds:.0f} s per epoch -> at most {hours:.1f} h for the {timing['runs_left']} "
                f"unfinished runs ({MAX_EPOCHS} epochs each, data loading not included; "
                f"early stopping usually ends runs sooner)"
            )

    if progress["bad_epochs"] >= PATIENCE:
        print(f"  early stopping: no improvement for {PATIENCE} epochs")
    last_path.unlink()
    print(f"  done: best epoch {progress['best_epoch']}, val loss {progress['best_val']:.4e}")


def main():
    # NOT IN PAPER: strict fp32, no autocast and no TF32 (TF32 only exists on Ampere and newer GPUs).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = pick_device()
    weights_sha256 = sha256_of(POSEIDON_DIR / "model.safetensors")

    # NOT IN PAPER: one independent run per (variant, mix, PDE).
    runs = [(variant, mix, pde) for variant in VARIANTS for mix in MIXES for pde in PDES]
    runs_left = sum(not is_finished(OUT_ROOT / v / m / p) for v, m, p in runs)
    print(f"Device: {device}" + ("  (SMOKE TEST)" if SMOKE_TEST else ""))
    print(f"Runs: {len(runs)} ({len(runs) - runs_left} finished, {runs_left} to do), outputs in {OUT_ROOT}")
    timing = {"printed": False, "runs_left": runs_left}
    for variant, mix, pde in runs:
        train_run(variant, mix, pde, device, weights_sha256, timing)


if __name__ == "__main__":
    main()

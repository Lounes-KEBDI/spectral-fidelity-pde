"""Fine-tune MPP-Ti (MPP-AViT-Ti) on PhysBiasBench (training only: no inference, no evaluation).

Follows Chu et al., "Do Physics Foundation Models Learn Generalizable Physics?" (arXiv:2605.29283):
one-step prediction with teacher forcing on the dataset's own train/val windows
(5 input frames -> 1 target frame, taken from the first 15 frames of each 20-frame sequence).

48 independent runs = 2 variants x 3 mixtures x 8 PDEs:
  pretrained  MPP-AViT-Ti weights, projections expanded for our fields as MPP's own fine-tuning does,
              then fine-tuned
  scratch     the same expanded architecture with random weights, trained the same way
MPP only: flat train windows are left out of training (FLAT_STD_THRESHOLD), and a batch with non-finite
gradients is skipped (train_epoch). Validation keeps every window.

Outputs per run, in checkpoint/MPP-Ti/{variant}/{mix}/{pde}/:
  best.pt         best epoch (lowest val loss), enough to rebuild the model on its own:
                      model = build_avit(SimpleNamespace(**ckpt["mpp_config"]))
                      model.expand_projections(ckpt["n_channels"])
                      model.load_state_dict(ckpt["model_state_dict"])
                  (build_avit from third_party/MPP/models/avit.py)
  last.pt         latest epoch + optimizer, scheduler and RNG states, for resuming (deleted at the end)
  train_log.csv   epoch, train_loss, val_loss, lr, seconds

Code references in comments:
  MPP/...           github.com/PolymathicAI/multiple_physics_pretraining at commit
                    e751fc25ae5274c3ddc869bf73afd6fcd4163cb5
                    (cloned into third_party/MPP/ by third_party/download_mpp_code.py)
  config.yaml       MPP/config/mpp_avit_ti_config.yaml
  benchmark_api.py  data/benchmark_api.py (shipped with the PhysBiasBench dataset)
Choices that are not in the paper are marked "NOT IN PAPER".

Usage (from any directory): python MPP-Ti/finetune.py
"""

import csv
import hashlib
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Lets ops missing from Apple's MPS backend fall back to the CPU. Must be set before importing torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data"))
from benchmark_api import make_train_val_datasets  # noqa: E402

MPP_DIR = REPO_ROOT / "third_party" / "MPP"  # created by third_party/download_mpp_code.py
MPP_COMMIT = "e751fc25ae5274c3ddc869bf73afd6fcd4163cb5"


def import_build_avit():
    # MPP is not pip-installable (no setup.py), and its package name "models" would clash with this repo's
    # models/ folder, so MPP/models/ itself goes on sys.path: its modules fall back to plain imports when
    # relative imports fail (MPP/models/avit.py:7-12, MPP/models/mixed_modules.py:9-14,
    # MPP/models/spatial_modules.py:9-12, MPP/models/time_modules.py:10-13). They only import torch, numpy,
    # einops and timm (MPP/models/spatial_modules.py:1-8, MPP/models/time_modules.py:1-7).
    head = MPP_DIR / ".git" / "HEAD"
    if not head.exists():
        raise FileNotFoundError(f"{MPP_DIR} not found: run python third_party/download_mpp_code.py")
    assert head.read_text().strip() == MPP_COMMIT, f"{MPP_DIR} is not at commit {MPP_COMMIT}"
    sys.path.insert(0, str(MPP_DIR / "models"))
    from avit import build_avit

    return build_avit


build_avit = import_build_avit()


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
# NOT IN PAPER: the paper's scratch baseline is M-sized; we only have MPP-Ti, so scratch is Ti-sized.
VARIANTS = ["pretrained", "scratch"]
# True: 1 PDE, 1 mix, 1 epoch, a few batches. Saved under checkpoint/MPP-Ti_smoke/ so a smoke
# run is never mistaken for a finished real run.
SMOKE_TEST = False
SMOKE_BATCHES = 3

DATA_ROOT = REPO_ROOT / "data" / "Data"
MPP_WEIGHTS = REPO_ROOT / "models" / "MPP-Ti" / "MPP_AViT_Ti"  # downloaded by hand, see models/MPP_Ti.md
OUT_ROOT = REPO_ROOT / "checkpoint" / ("MPP-Ti_smoke" if SMOKE_TEST else "MPP-Ti")

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

# NOT IN PAPER: MPP only. A train window is "flat" if one of its 5 input frames has a spatial std below this in
# EVERY channel at once (our normalised units): MPP then gets constant feature maps for that frame and its
# per-frame spatial normalisation divides by ~0. Flat windows are left out of MPP's training set
# (exclude_flat_windows); validation keeps every window (MPP-Ti/debug_nan.py, NOT_IN_PAPER.md 5.11).
FLAT_STD_THRESHOLD = 1e-3
# NOT IN PAPER: MPP only. Safety net: a batch with a non-finite gradient entry is skipped (no optimizer step),
# and the run aborts if more than this fraction of an epoch's steps are skipped (train_epoch).
MAX_SKIPPED_FRACTION = 0.05

# The MPP-AViT-Ti configuration: the values build_avit reads (MPP/models/avit.py:14-27,
# MPP/models/mixed_modules.py:16-24, MPP/models/spatial_modules.py:18-20, MPP/models/time_modules.py:15-20)
# from config.yaml. The released weights use n_states=12 (MPP/README.md:133-135).
MPP_CONFIG = {
    "patch_size": [16, 16],  # config.yaml:47
    "embed_dim": 192,  # config.yaml:44
    "num_heads": 3,  # config.yaml:45
    "processor_blocks": 12,  # config.yaml:46
    "n_states": 12,  # config.yaml:32
    "block_type": "axial",  # config.yaml:40
    "time_type": "attention",  # config.yaml:41
    "space_type": "axial_attention",  # config.yaml:42
    "bias_type": "rel",  # config.yaml:48
    "gradient_checkpointing": False,  # config.yaml:12
}
# Kept: build_avit does not pass config.yaml's drop_path (0.1, config.yaml:21) to AViT, so AViT's default
# applies: stochastic depth rising linearly from 0 to 0.2 over the 12 blocks (MPP/models/avit.py:22-26,
# 40-44, 53-54). That is what MPP's pretraining and fine-tuning ran with.
AVIT_DROP_PATH = 0.2

# NOT IN PAPER: MPP sees all 5 input frames. It was pretrained with 16 (n_steps, config.yaml:35), but its
# time attention has no fixed length: the relative position bias is computed for any number of frames
# (MPP/models/time_modules.py:45-61, MPP/models/shared_modules.py:99-128).
FRAMES_USED = list(range(INPUT_FRAMES))

# NOT IN PAPER: MPP's own fine-tuning procedure (MPP/README.md:133-144), for every PDE. Our C fields are a
# new dataset appended at the bottom of DSET_NAME_TO_OBJECT (MPP/data_utils/datasets.py:21-26). With
# use_all_fields (config.yaml:52), its fields take the state indices that follow the 10 pretraining fields
# (MPP/data_utils/datasets.py:108-114): 0 h (SWE), 1-3 incompressible NS, 4-5 diffusion-reaction, 6-9
# compressible NS (MPP/data_utils/hdf5_datasets.py:220, 242, 267, 331). So our labels are 10 .. 10+C-1, and
# the input/output projections are expanded by C (MPP/models/avit.py:57-72, MPP/train_basic.py:202-207).
# Slots 10 and 11 exist in the checkpoint but were never used in pretraining.
PRETRAINING_FIELDS = 10

# NOT IN PAPER: every PhysBiasBench domain is periodic, so bcs = [1, 1], the value MPP gives its periodic
# datasets (MPP/data_utils/hdf5_datasets.py:361-362). Only bcs[0, 0] (W axis) and bcs[0, 1] (H axis) are
# read (MPP/models/spatial_modules.py:163, 172); 1 wraps the relative position bias
# (MPP/models/shared_modules.py:113-116).
BCS = [1, 1]

# NOT IN PAPER: MPP runs natively at 64x64 with its pretrained weights: it has no positional embedding, its
# patch stem and output are 16x16 convolution stacks (MPP/models/spatial_modules.py:72-120), and its spatial
# attention uses a bucketed relative position bias (MPP/models/spatial_modules.py:160-178). Its pretraining
# already mixed resolutions (128 and 512, config.yaml:59-61), read as is (MPP/data_utils/hdf5_datasets.py:
# 182-209). So there is no resampling: 64x64 in, 64x64 out, loss at 64x64.
RESAMPLING = "none: MPP runs natively at 64x64 (4x4 tokens per frame)"

# The tensors that expand_projections enlarges, and the axis it enlarges (MPP/models/avit.py:57-72).
EXPANDED_AXIS = {"space_bag.weight": 1, "debed.out_kernel": 1, "debed.out_bias": 0}

if SMOKE_TEST:
    PDES, MIXES, MAX_EPOCHS = PDES[:1], MIXES[:1], 1


# ============================================================================= helpers
def state_labels(n_channels):
    """MPP state indices of our C channels: 10 .. 10+C-1 (see PRETRAINING_FIELDS)."""
    return list(range(PRETRAINING_FIELDS, PRETRAINING_FIELDS + n_channels))


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        # torch 2.0.1 may not support, on MPS, the attention with a float mask and the statistics MPP uses
        # (MPP/models/spatial_modules.py:166, MPP/models/avit.py:106). Try them once and fall back to the CPU.
        try:
            q = torch.randn(1, 1, 4, 8, device="mps")
            F.scaled_dot_product_attention(q, q, q, attn_mask=torch.zeros(1, 1, 4, 4, device="mps"))
            torch.std_mean(q, dim=(0, -2, -1), keepdim=True)
            return torch.device("mps")
        except Exception as err:
            print(f"MPS cannot run MPP's attention ops ({type(err).__name__}), using the CPU.")
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


def gradients_finite(model):
    """True if every gradient entry is finite. The total norm is computed in float64, where finite float32
    gradients cannot overflow, so it is non-finite exactly when some entry is."""
    norms = [p.grad.double().norm() for p in model.parameters() if p.grad is not None]
    return bool(torch.isfinite(torch.stack(norms).norm())) if norms else True


# ============================================================================= data
def load_windows(pde, mix):
    """Builds the dataset's train/val windows with the unmodified API, checks them, and returns the
    normalised (input frames, target frame) pairs as tensors plus the normalisation constants."""
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
            inputs.append(x[FRAMES_USED])
            targets.append(y[0])
        n_values = n * (INPUT_FRAMES + TARGET_FRAMES) * RESOLUTION * RESOLUTION
        return np.stack(inputs), np.stack(targets), total / n_values, total_sq / n_values

    n_train = SMOKE_BATCHES * BATCH_SIZE if SMOKE_TEST else len(train_ds)
    n_val = SMOKE_BATCHES * BATCH_SIZE if SMOKE_TEST else len(val_ds)
    x_train, y_train, mean, mean_sq = collect(train_ds, n_train)
    x_val, y_val, _, _ = collect(val_ds, n_val)

    # NOT IN PAPER: per-channel mean/std over every frame (x and y) of the TRAIN windows of this
    # (PDE, mix) only, as for every model. MPP also normalises inside the model: per sample and per
    # channel over (T, H, W), and it de-normalises its output with the same mean/std
    # (MPP/models/avit.py:104-108, 129-130). That per-sample z-score cancels any per-channel affine map
    # applied before it, so ours does not change what the network sees (up to MPP's 1e-7 added to the std,
    # MPP/models/avit.py:107). It only sets the space the prediction comes out in, which is the space our
    # loss uses.
    std = np.sqrt(mean_sq - mean**2)
    assert (std > 0).all(), f"{pde}/{mix}: a channel is constant"

    def normalise(a):
        # mean[:, None, None] lines up with the channel axis of (..., C, H, W).
        return torch.from_numpy((a - mean[:, None, None]) / std[:, None, None]).float()

    return {
        "n_channels": n_channels,
        "mean": mean,
        "std": std,
        "x_train": normalise(x_train),  # (N, 5, C, 64, 64)
        "y_train": normalise(y_train),  # (N, C, 64, 64)
        "x_val": normalise(x_val),
        "y_val": normalise(y_val),
    }


def flatness_scores(x):
    """(N,) flatness score of windows x, (N, 5, C, H, W) in our normalised units: the min over the input frames
    of the frame's largest spatial std over its channels. Low when some frame is flat in every channel."""
    frame_std = torch.cat([chunk.std(dim=(3, 4)) for chunk in x.split(256)])  # (N, 5, C)
    return frame_std.amax(dim=2).amin(dim=1)


def flat_windows(x):
    """(N,) bool for windows x, (N, 5, C, H, W) in our normalised units: True where some input frame has a
    spatial std below FLAT_STD_THRESHOLD in every channel at once, i.e. flatness score below the threshold."""
    return flatness_scores(x) < FLAT_STD_THRESHOLD


def exclude_flat_windows(data):
    """NOT IN PAPER: MPP only. Removes the flat windows from data's train set, in place; val keeps every
    window. Returns the indices of the removed windows among the dataset's train windows, and records in
    data["train_window_ids"] the dataset index of every kept train window (for logging)."""
    flat = flat_windows(data["x_train"])
    data["x_train"], data["y_train"] = data["x_train"][~flat], data["y_train"][~flat]
    data["train_window_ids"] = torch.nonzero(~flat).flatten().tolist()
    data["flat_train_windows"] = torch.nonzero(flat).flatten().tolist()
    return data["flat_train_windows"]


# ============================================================================= model
def build_model(variant, n_channels):
    # build_avit is MPP's own builder (MPP/models/avit.py:14-27), here fed MPP_CONFIG.
    model = build_avit(SimpleNamespace(**MPP_CONFIG))
    pretrained_state = load_pretrained(model) if variant == "pretrained" else None
    # NOT IN PAPER: both variants are expanded by C, as MPP's fine-tuning does right after loading the
    # pretrained weights (MPP/train_basic.py:202-207). The C new input columns and output channels keep
    # PyTorch's default init; the existing ones, biases included, are copied (MPP/models/avit.py:57-72).
    model.expand_projections(n_channels)
    if pretrained_state is not None:
        check_pretrained_load(model, pretrained_state, n_channels)
    # scratch: AViT's own random init. MPP applies no custom init: the layer-scale gammas start at 1e-6
    # (MPP/models/spatial_modules.py:128-131, MPP/models/time_modules.py:31-32), every other layer keeps
    # PyTorch's default.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,} ({n_params / 1e6:.1f}M, MPP-Ti is ~7.3M before expansion)")
    return model


def load_pretrained(model):
    # MPP_AViT_Ti is a bare state dict, with no "model_state" key and no "module." prefix. MPP's own loader
    # accepts both layouts and strips "module." (MPP/train_basic.py:171-186), so this does too.
    # weights_only=False, as in the other scripts.
    ckpt = torch.load(MPP_WEIGHTS, map_location="cpu", weights_only=False)
    if "model_state" in ckpt:
        print(f"  MPP_AViT_Ti: checkpoint with keys {sorted(ckpt)}")
        ckpt = ckpt["model_state"]
    else:
        print(f"  MPP_AViT_Ti: bare state dict with {len(ckpt)} tensors")
    state = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in ckpt.items()}
    # Every tensor is loaded, before the expansion, as MPP/train_basic.py:168-207 does.
    result = model.load_state_dict(state, strict=False)
    print(f"  missing keys: {result.missing_keys}")
    print(f"  unexpected keys: {result.unexpected_keys}")
    assert not result.missing_keys and not result.unexpected_keys
    return state


def check_pretrained_load(model, state, n_channels):
    # After the expansion, every tensor must still be exactly the one in MPP_AViT_Ti, except the 3 expanded
    # ones, whose first 12 slots must be.
    model_state = model.state_dict()
    assert set(model_state) == set(state)
    for key, value in state.items():
        if key in EXPANDED_AXIS:
            axis = EXPANDED_AXIS[key]
            print(f"  expanded: {key} {tuple(value.shape)} -> {tuple(model_state[key].shape)}")
            assert model_state[key].shape[axis] == value.shape[axis] + n_channels
            assert torch.equal(model_state[key].narrow(axis, 0, value.shape[axis]), value), key
        else:
            assert torch.equal(model_state[key], value), f"{key} differs from the checkpoint"


def predict(model, x):
    # (B, 5, C, 64, 64) -> (B, C, 64, 64). MPP takes the frames as (T, B, C, H, W) (MPP/models/avit.py:102-103;
    # its trainer does the same rearrange, MPP/train_basic.py:233), the state labels per sample, (B, C), and
    # the boundary conditions, (B, 2) (MPP/train_basic.py:230, 240). It returns the de-normalised next frame,
    # (B, C, H, W), with no auxiliary output (MPP/models/avit.py:129-131).
    b, _, c = x.shape[:3]
    labels = torch.tensor(state_labels(c), device=x.device).expand(b, c)
    bcs = torch.tensor(BCS, device=x.device).expand(b, 2)
    return model(x.permute(1, 0, 2, 3, 4), labels, bcs)


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    squared_error = 0.0
    for i in range(0, len(x), BATCH_SIZE):
        pred = predict(model, x[i : i + BATCH_SIZE])
        squared_error += F.mse_loss(pred, y[i : i + BATCH_SIZE], reduction="sum").item()
    return squared_error / y.numel()


# ============================================================================= checkpoints
def run_info(variant, mix, pde, data, weights_sha256):
    """Everything besides the weights that is needed to rebuild and use the model later."""
    n_channels = data["n_channels"]
    return {
        "variant": variant,
        "pde": pde,
        "mix": mix,
        "n_channels": n_channels,  # also the expansion of the projections: n_states becomes 12 + n_channels
        "mpp_config": dict(MPP_CONFIG),
        "drop_path": AVIT_DROP_PATH,  # set by AViT itself, not by mpp_config
        "frames_used": FRAMES_USED,  # indices into x (input_frames, C, H, W): all 5 input frames
        "state_labels": state_labels(n_channels),  # our channel i -> MPP state index 10 + i
        "bcs": BCS,
        "resampling": RESAMPLING,
        "flat_std_threshold": FLAT_STD_THRESHOLD,  # train windows flatter than this were left out of training
        "n_flat_excluded": len(data["flat_train_windows"]),
        "flat_train_windows": data["flat_train_windows"],  # their indices among the dataset's train windows
        "norm_mean": data["mean"].tolist(),  # per channel; model space = (field - mean) / std
        "norm_std": data["std"].tolist(),
        "pretrained_weights_sha256": weights_sha256,
        "mpp_commit": MPP_COMMIT,
    }


def model_checkpoint(model, epoch, val_loss, info):
    weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return {"model_state_dict": weights, "epoch": epoch, "val_loss": val_loss, **info}


# ============================================================================= training
def train_epoch(model, optimizer, scheduler, shuffle_gen, x_train, y_train, epoch, window_ids=None):
    """One epoch of one-step prediction with teacher forcing: the inputs are always ground-truth frames.
    window_ids[i] is the dataset index of x_train[i], used only to log skipped batches. Returns (train loss over
    the steps taken, skipped batches, number of steps); each skipped batch is (step, [(window, flatness)])."""
    # model.train() also turns on MPP's own stochastic depth (AVIT_DROP_PATH), kept as MPP builds it.
    model.train()
    order = torch.randperm(len(x_train), generator=shuffle_gen).to(x_train.device)
    n_steps = math.ceil(len(order) / BATCH_SIZE)
    loss_sum, n_used, skipped = 0.0, 0, []
    for step, i in enumerate(range(0, len(order), BATCH_SIZE)):
        idx = order[i : i + BATCH_SIZE]
        loss = F.mse_loss(predict(model, x_train[idx]), y_train[idx])  # NOT IN PAPER: MSE, normalised space
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # NOT IN PAPER: MPP only. Safety net: if a gradient entry is non-finite, the batch is skipped (no
        # optimizer step, gradients cleared, left out of the train loss) and the scheduler still steps.
        if gradients_finite(model):
            optimizer.step()
            loss_sum += loss.item() * len(idx)
            n_used += len(idx)
        else:
            optimizer.zero_grad(set_to_none=True)
            # Logging only: the windows of the skipped batch and their flatness scores, flattest first.
            ids = [window_ids[j] if window_ids is not None else j for j in idx.tolist()]
            windows = sorted(zip(ids, flatness_scores(x_train[idx]).tolist()), key=lambda w: w[1])
            skipped.append((step, windows))
            print(f"  skipped step {step} of epoch {epoch} (non-finite gradient); windows (flatness score): "
                  + ", ".join(f"{w} ({s:.2e})" for w, s in windows))
            if len(skipped) > MAX_SKIPPED_FRACTION * n_steps:
                raise RuntimeError(
                    f"{len(skipped)} of the {n_steps} steps of epoch {epoch} had non-finite gradients and were "
                    f"skipped, more than {MAX_SKIPPED_FRACTION:.0%}: aborting. Resubmitting would hit the same batches."
                )
        scheduler.step()
    return loss_sum / n_used, skipped, n_steps


def train_run(variant, mix, pde, device, weights_sha256, timing):
    run_dir = OUT_ROOT / variant / mix / pde
    best_path, last_path, log_path = run_dir / "best.pt", run_dir / "last.pt", run_dir / "train_log.csv"
    print(f"\n=== {variant} / {mix} / {pde}")
    if is_finished(run_dir):
        print("  already finished, skipping")
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    data = load_windows(pde, mix)
    excluded = exclude_flat_windows(data)  # NOT IN PAPER: MPP only, see FLAT_STD_THRESHOLD
    print(
        f"  flat train windows excluded: {len(excluded)} of {len(excluded) + len(data['x_train'])} "
        f"(an input frame with a spatial std below {FLAT_STD_THRESHOLD:g} in every channel)"
    )
    print(
        f"  channels: {data['n_channels']}, windows: {len(data['x_train'])} train / {len(data['x_val'])} val, "
        f"mean {np.round(data['mean'], 4).tolist()}, std {np.round(data['std'], 4).tolist()}"
    )
    seed_everything(SEED)  # NOT IN PAPER: seed 0 for every run
    model = build_model(variant, data["n_channels"]).to(device)
    info = run_info(variant, mix, pde, data, weights_sha256)

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
        train_loss, skipped, n_steps = train_epoch(
            model, optimizer, scheduler, shuffle_gen, x_train, y_train, epoch, window_ids=data["train_window_ids"]
        )
        val_loss = evaluate(model, x_val, y_val)
        assert math.isfinite(train_loss) and math.isfinite(val_loss), "loss is not finite"
        seconds = time.time() - start
        lr = optimizer.param_groups[0]["lr"]

        progress["history"].append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr, "seconds": round(seconds, 1)}
        )
        write_log(log_path, progress["history"])
        print(
            f"  epoch {epoch:2d}/{MAX_EPOCHS}  train {train_loss:.4e}  val {val_loss:.4e}  lr {lr:.2e}  {seconds:.0f}s"
            f"  skipped steps {len(skipped)}/{n_steps}"
        )

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
    weights_sha256 = sha256_of(MPP_WEIGHTS)

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

"""Fine-tune GPhyT-S on PhysBiasBench (training only: no inference, no evaluation).

Follows Chu et al., "Do Physics Foundation Models Learn Generalizable Physics?" (arXiv:2605.29283):
one-step prediction with teacher forcing on the dataset's own train/val windows
(5 input frames -> 1 target frame, taken from the first 15 frames of each 20-frame sequence).

48 independent runs = 2 variants x 3 mixtures x 8 PDEs:
  pretrained  GPhyT-S weights, architecture unchanged (our data is padded to fit it), then fine-tuned
  scratch     the same architecture with random weights, trained the same way

Outputs per run, in checkpoint/GPhyT-S/{variant}/{mix}/{pde}/:
  best.pt         best epoch (lowest val loss), enough to rebuild the model on its own:
                      model = PhysicsTransformer(**ckpt["gphyt_config"])
                      model.load_state_dict(ckpt["model_state_dict"])
                  (PhysicsTransformer from third_party/GPhyT/gphyt/model/transformer/model.py)
  last.pt         latest epoch + optimizer, scheduler and RNG states, for resuming (deleted at the end)
  train_log.csv   epoch, train_loss, val_loss, lr, seconds

Code references in comments:
  GPhyT/...         github.com/FloWsnr/General-Physics-Transformer at commit
                    4374116e8c96db08f5b0691b04485959749d6374, the commit that published the weights
                    (cloned into third_party/GPhyT/ by third_party/download_gphyt_code.py)
  hf:README.md      the model card at huggingface.co/flwi/Physics-Foundation-Model
  benchmark_api.py  data/benchmark_api.py (shipped with the PhysBiasBench dataset)
Choices that are not in the paper are marked "NOT IN PAPER".

Usage (from any directory): python GPhyT-S/finetune.py
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
from torch.optim.lr_scheduler import LambdaLR

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data"))
from benchmark_api import make_train_val_datasets  # noqa: E402

GPHYT_DIR = REPO_ROOT / "third_party" / "GPhyT"  # created by third_party/download_gphyt_code.py
GPHYT_COMMIT = "4374116e8c96db08f5b0691b04485959749d6374"


def import_physics_transformer():
    # GPhyT's pyproject.toml asks for Python >= 3.12 and lists no dependencies (GPhyT/pyproject.toml:6-7),
    # so it is not pip-installed: the clone is put on sys.path. The model code only imports torch,
    # torchvision and einops (GPhyT/gphyt/model/transformer/attention.py:9-12,
    # GPhyT/gphyt/model/tokenizer/tokenizer.py:8-11, GPhyT/gphyt/model/transformer/derivatives.py:8-12).
    head = GPHYT_DIR / ".git" / "HEAD"
    if not head.exists():
        raise FileNotFoundError(f"{GPHYT_DIR} not found: run python third_party/download_gphyt_code.py")
    assert head.read_text().strip() == GPHYT_COMMIT, f"{GPHYT_DIR} is not at commit {GPHYT_COMMIT}"
    sys.path.insert(0, str(GPHYT_DIR))
    from gphyt.model.transformer.model import PhysicsTransformer

    return PhysicsTransformer


PhysicsTransformer = import_physics_transformer()


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
# NOT IN PAPER: the paper's scratch baseline is M-sized; we only have GPhyT-S, so scratch is S-sized.
VARIANTS = ["pretrained", "scratch"]
# True: 1 PDE, 1 mix, 1 epoch, a few batches. Saved under checkpoint/GPhyT-S_smoke/ so a smoke
# run is never mistaken for a finished real run.
SMOKE_TEST = False
SMOKE_BATCHES = 3

DATA_ROOT = REPO_ROOT / "data" / "Data"
GPHYT_WEIGHTS = REPO_ROOT / "models" / "GPhyT-S" / "gphyt-S.pth"
OUT_ROOT = REPO_ROOT / "checkpoint" / ("GPhyT-S_smoke" if SMOKE_TEST else "GPhyT-S")

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

# The pretrained GPhyT-S configuration: the PhysicsTransformer arguments built by
# GPhyT/gphyt/model/transformer/model.py:17-53 (same as hf:README.md:12-48) from the GPT_S sizes
# (GPhyT/gphyt/model/model_specs.py:5-10) and the training config GPhyT/gphyt/run/scripts/config.yaml
# (cited below as config.yaml). The model card gives no other values. This configuration is the only one
# that fits the 32 MB of gphyt-S.pth (~8.0M fp32 parameters); the strict load in load_pretrained checks it.
# NOT IN PAPER: both variants keep this architecture unchanged (5 fields, 4 input frames, 256x128),
# and our data is padded and resampled to fit it.
GPHYT_CONFIG = {
    "num_fields": 5,  # config.yaml:21
    "hidden_dim": 192,  # model_specs.py:6
    "mlp_dim": 768,  # model_specs.py:7
    "num_heads": 3,  # model_specs.py:8
    "num_layers": 12,  # model_specs.py:9
    "patch_size": (1, 16, 16),  # (time, height, width), config.yaml:26
    "img_size": (4, 256, 128),  # (n_steps_input, *out_shape), config.yaml:86, 88; GPhyT/gphyt/run/train.py:100-102
    "use_derivatives": True,  # config.yaml:28
    "pos_enc_mode": "absolute",  # config.yaml:25
    "att_mode": "full",  # config.yaml:23
    "integrator": "Euler",  # config.yaml:29
    "tokenizer_mode": "linear",  # config.yaml:31
    "detokenizer_mode": "linear",  # config.yaml:32
    "tokenizer_overlap": 0,  # config.yaml:33
    "detokenizer_overlap": 0,  # config.yaml:34
    "tokenizer_net_channels": (64,),  # GPT_S conv_channels (model_specs.py:10), unused by "linear"
    "detokenizer_net_channels": (64,),
    "dropout": 0.0,  # config.yaml:24, kept: no train-time randomness in the model
    "stochastic_depth_rate": 0.0,  # config.yaml:27, kept
}
MODEL_FIELDS = GPHYT_CONFIG["num_fields"]
MODEL_FRAMES, MODEL_HEIGHT, MODEL_WIDTH = GPHYT_CONFIG["img_size"]

# NOT IN PAPER: GPhyT sees the last 4 of our 5 input frames. It was pretrained with 4 (config.yaml:86),
# and its learned positional embedding has one slice per frame, shape (1, 4, 16, 8, 192), added as is
# (GPhyT/gphyt/model/transformer/pos_encodings.py:40, 56; GPhyT/gphyt/model/transformer/model.py:146-148,
# 182-188), so it takes exactly 4 frames.
FRAMES_USED = list(range(INPUT_FRAMES - MODEL_FRAMES, INPUT_FRAMES))  # [1, 2, 3, 4]

# NOT IN PAPER: GPhyT-S has 5 fields, in this order (config.yaml:21; GPhyT/gphyt/data/dataset_utils.py:
# 119-122). For every PDE, our C real channels go into the first C slots, and the other 5 - C fields are
# filled with 0.0 in normalised space (the mean of a z-scored field). The repo shows no fill value of its
# own: its loader reads all 5 fields from every file (GPhyT/gphyt/data/well_dataset.py:458-473). The loss
# uses only the C real output channels.
GPHYT_FIELDS = ["pressure", "density", "temperature", "vel_x", "vel_y"]
FIELD_PAD_VALUE = 0.0

# NOT IN PAPER: GPhyT only runs at its img_size: 16x16 patches of 256x128 give a fixed 16 x 8 grid for the
# positional embedding (GPhyT/gphyt/model/transformer/model.py:146-148, 182-188), and the repo has no
# resizing code. We Fourier-resample our 64x64 fields to 256x128 (x4 along H, x2 along W) and the
# prediction back to 64x64 with the same routine as DPOT-Ti (fourier_resample below), and compute the
# loss at 64x64.
RESAMPLING = (
    "ours: input 64x64 -> 256x128 (H x W) by Fourier zero-padding, output 256x128 -> 64x64 by Fourier "
    "truncation (FFT with norm='forward', Nyquist mode split / summed so that down(up(x)) == x)"
)

if SMOKE_TEST:
    PDES, MIXES, MAX_EPOCHS = PDES[:1], MIXES[:1], 1


# ============================================================================= helpers
def fourier_resize_axis(x, n_out, dim):
    """Band-limited resampling of a periodic signal along one axis: zero-pads (n_out > n_in) or
    truncates (n_out < n_in) its FFT. norm="forward" keeps amplitudes. The Nyquist mode of the smaller
    grid is split equally between +n/2 and -n/2 when upsampling and summed back when downsampling."""
    n_in = x.shape[dim]
    if n_in == n_out:
        return x
    assert n_in % 2 == 0 and n_out % 2 == 0
    spec = torch.fft.fft(x, dim=dim, norm="forward")
    if n_out > n_in:
        half = n_in // 2
        nyquist = spec.narrow(dim, half, 1) / 2
        zeros_shape = list(spec.shape)
        zeros_shape[dim] = n_out - n_in - 1
        parts = [
            spec.narrow(dim, 0, half),  # frequencies 0 .. half-1
            nyquist,  # +half
            spec.new_zeros(zeros_shape),  # new high frequencies
            nyquist,  # -half
            spec.narrow(dim, half + 1, half - 1),  # frequencies -(half-1) .. -1
        ]
    else:
        half = n_out // 2
        parts = [
            spec.narrow(dim, 0, half),
            spec.narrow(dim, half, 1) + spec.narrow(dim, n_in - half, 1),  # +half and -half -> Nyquist
            spec.narrow(dim, n_in - half + 1, half - 1),
        ]
    return torch.fft.ifft(torch.cat(parts, dim=dim), dim=dim, norm="forward")


def fourier_resample(x, height, width):
    """(..., H, W) real periodic fields -> (..., height, width). Never bilinear or F.interpolate."""
    return fourier_resize_axis(fourier_resize_axis(x, height, -2), width, -1).real


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        # torch 2.0.1 may not support, on MPS, the FFT/complex ops of our resampling or the 3D convolutions
        # of GPhyT's tokenizer and derivatives (GPhyT/gphyt/model/tokenizer/tokenizer.py:169-176,
        # GPhyT/gphyt/model/transformer/derivatives.py:91). Try them once and fall back to the CPU.
        try:
            x = torch.randn(1, 1, 8, 8, device="mps")
            fourier_resample(fourier_resample(x, 16, 16), 8, 8)
            F.conv3d(torch.randn(1, 1, 2, 4, 4, device="mps"), torch.randn(1, 1, 1, 2, 2, device="mps"))
            return torch.device("mps")
        except Exception as err:
            print(f"MPS cannot run the FFT / 3D convolution ops ({type(err).__name__}), using the CPU.")
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
    # (PDE, mix) only. The dataset API does not normalise, and neither does GPhyT inside the model
    # (GPhyT/gphyt/model/transformer/model.py:224-248 has no normalisation; GPhyT z-scores each field in
    # its data loader, GPhyT/gphyt/data/well_dataset.py:488-492), so the data is normalised exactly once,
    # here.
    std = np.sqrt(mean_sq - mean**2)
    assert (std > 0).all(), f"{pde}/{mix}: a channel is constant"

    def normalise(a):
        # mean[:, None, None] lines up with the channel axis of (..., C, H, W).
        return torch.from_numpy((a - mean[:, None, None]) / std[:, None, None]).float()

    return {
        "n_channels": n_channels,
        "mean": mean,
        "std": std,
        "x_train": normalise(x_train),  # (N, 4, C, 64, 64)
        "y_train": normalise(y_train),  # (N, C, 64, 64)
        "x_val": normalise(x_val),
        "y_val": normalise(y_val),
    }


# ============================================================================= model
def build_model(variant):
    model = PhysicsTransformer(**GPHYT_CONFIG)
    if variant == "pretrained":
        load_pretrained(model)
    # scratch: PhysicsTransformer's own random init. GPhyT applies no custom init: the positional
    # embedding is 0.02 * randn (GPhyT/gphyt/model/transformer/pos_encodings.py:40), every other layer
    # keeps PyTorch's default.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,} ({n_params / 1e6:.1f}M, GPhyT-S is ~8.0M)")
    return model


def strip_prefixes(key):
    # Remove the "module." (DDP) and "_orig_mod." (torch.compile, GPhyT/gphyt/run/train.py:171-173)
    # prefixes, as GPhyT's own loader does (GPhyT/gphyt/run/run_utils.py:29-40).
    while True:
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        else:
            return key


def load_pretrained(model):
    # gphyt-S.pth holds only the weights (32 MB, no optimizer state). GPhyT's training checkpoints keep
    # them under "model_state_dict" (GPhyT/gphyt/run/train.py:384-399), so both that layout and a bare
    # state dict are accepted. weights_only=False because GPhyT's checkpoints also pickle Python objects
    # (GPhyT/gphyt/run/run_utils.py:27-28).
    ckpt = torch.load(GPHYT_WEIGHTS, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        print(f"  gphyt-S.pth: checkpoint with keys {sorted(ckpt)}")
        ckpt = ckpt["model_state_dict"]
    else:
        print(f"  gphyt-S.pth: bare state dict with {len(ckpt)} tensors")
    state = {strip_prefixes(k): v for k, v in ckpt.items()}
    # Every tensor is loaded, as the model card says (hf:README.md:10).
    result = model.load_state_dict(state, strict=False)
    print(f"  missing keys: {result.missing_keys}")
    print(f"  unexpected keys: {result.unexpected_keys}")
    assert not result.missing_keys and not result.unexpected_keys

    # Nothing is replaced: every tensor must be exactly the one in gphyt-S.pth.
    model_state = model.state_dict()
    assert set(model_state) == set(state)
    differ = {k for k in state if not torch.equal(model_state[k], state[k])}
    assert not differ, f"tensors differing from the checkpoint: {sorted(differ)}"


def to_gphyt_input(x):
    """Normalised frames (B, 4, C, 64, 64) -> GPhyT input (B, 4, 256, 128, 5)."""
    x = fourier_resample(x, MODEL_HEIGHT, MODEL_WIDTH)  # (B, 4, C, 256, 128)
    b, t, c, h, w = x.shape
    pad = torch.full((b, t, MODEL_FIELDS - c, h, w), FIELD_PAD_VALUE, device=x.device)
    x = torch.cat([x, pad], dim=2)  # (B, 4, 5, 256, 128): our channels in the first C field slots
    return x.permute(0, 1, 3, 4, 2)  # GPhyT layout (B, T, H, W, C), GPhyT/gphyt/model/transformer/derivatives.py:85


def predict(model, x):
    # (B, 4, C, 64, 64) -> (B, C, 64, 64). Kept as GPhyT does it: forward applies one Euler step with
    # step size 1.0 to the network's output on every input frame and keeps the last frame, i.e.
    # x_last + f(x)_last, shape (B, 1, 256, 128, 5) (GPhyT/gphyt/model/transformer/model.py:243-248,
    # GPhyT/gphyt/model/transformer/num_integration.py:39-41). There is no dt input: the model infers the
    # time scale from the frames, through the fixed finite-difference channels it adds itself
    # (GPhyT/gphyt/model/transformer/model.py:227-229; GPhyT/gphyt/model/transformer/derivatives.py:36-37,
    # 89-99: [-1, 0, 1] stencils with replicate padding, kept unchanged although our fields are periodic).
    # Only the C real channels are kept, so the padded fields get no loss.
    n_channels = x.shape[2]
    pred = model(to_gphyt_input(x))
    pred = pred[:, 0, :, :, :n_channels].permute(0, 3, 1, 2)  # (B, C, 256, 128)
    return fourier_resample(pred, RESOLUTION, RESOLUTION)


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
        "n_channels": n_channels,  # real channels = the first n_channels of the model's 5 fields
        "gphyt_config": dict(GPHYT_CONFIG),
        "frames_used": FRAMES_USED,  # indices into x (input_frames, C, H, W): the last 4 input frames
        "field_slots": {i: GPHYT_FIELDS[i] for i in range(n_channels)},  # our channel i -> GPhyT field slot i
        "gphyt_fields": GPHYT_FIELDS,
        "field_pad_value": FIELD_PAD_VALUE,  # value of the 5 - n_channels padded input fields
        "resampling": RESAMPLING,
        "norm_mean": data["mean"].tolist(),  # per channel; model space = (field - mean) / std
        "norm_std": data["std"].tolist(),
        "pretrained_weights_sha256": weights_sha256,
        "gphyt_commit": GPHYT_COMMIT,
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
    model = build_model(variant).to(device)
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

        # One-step prediction with teacher forcing: the inputs are always ground-truth frames.
        # NOT IN PAPER: no data augmentation. GPhyT's pretraining flipped x and y with probability 0.5
        # (config.yaml:89-90; GPhyT/gphyt/data/phys_dataset.py:185-196).
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
    weights_sha256 = sha256_of(GPHYT_WEIGHTS)

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

"""Fine-tune DPOT-Ti on PhysBiasBench (training only: no inference, no evaluation).

Follows Chu et al., "Do Physics Foundation Models Learn Generalizable Physics?" (arXiv:2605.29283):
one-step prediction with teacher forcing on the dataset's own train/val windows
(5 input frames -> 1 target frame, taken from the first 15 frames of each 20-frame sequence).

48 independent runs = 2 variants x 3 mixtures x 8 PDEs:
  pretrained  DPOT-Ti weights, architecture unchanged (our data is padded to fit it), then fine-tuned
  scratch     the same architecture with random weights, trained the same way

Outputs per run, in checkpoint/DPOT-Ti/{variant}/{mix}/{pde}/:
  best.pt         best epoch (lowest val loss), enough to rebuild the model on its own:
                      model = DPOTNet(**ckpt["dpot_config"]); model.load_state_dict(ckpt["model_state_dict"])
                  (DPOTNet from third_party/DPOT/models/dpot.py)
  last.pt         latest epoch + optimizer, scheduler and RNG states, for resuming (deleted at the end)
  train_log.csv   epoch, train_loss, val_loss, lr, seconds

Code references in comments:
  DPOT/...          github.com/thu-ml/DPOT at commit dcd2f9a9359765e19ad63e2f3f879a2a8ce1aa17
                    (cloned into third_party/DPOT/ by third_party/download_dpot_code.py)
  hf:README.md      the model card at huggingface.co/hzk17/DPOT
  benchmark_api.py  data/benchmark_api.py (shipped with the PhysBiasBench dataset)
Choices that are not in the paper are marked "NOT IN PAPER".

Usage (from any directory): python DPOT-Ti/finetune.py
"""

import csv
import hashlib
import importlib.util
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

DPOT_DIR = REPO_ROOT / "third_party" / "DPOT"  # created by third_party/download_dpot_code.py
DPOT_COMMIT = "dcd2f9a9359765e19ad63e2f3f879a2a8ce1aa17"


def import_dpotnet():
    # DPOT is not pip-installable, and its package name "models" would clash with this repo's models/
    # folder, so DPOT/models/dpot.py is loaded by file path. It only imports numpy, torch and einops
    # (DPOT/models/dpot.py:3-16).
    head = DPOT_DIR / ".git" / "HEAD"
    if not head.exists():
        raise FileNotFoundError(f"{DPOT_DIR} not found: run python third_party/download_dpot_code.py")
    assert head.read_text().strip() == DPOT_COMMIT, f"{DPOT_DIR} is not at commit {DPOT_COMMIT}"
    spec = importlib.util.spec_from_file_location("dpot_model", DPOT_DIR / "models" / "dpot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DPOTNet


DPOTNet = import_dpotnet()


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
# NOT IN PAPER: the paper's scratch baseline is M-sized; we only have DPOT-Ti, so scratch is Ti-sized.
VARIANTS = ["pretrained", "scratch"]
# True: 1 PDE, 1 mix, 1 epoch, a few batches. Saved under checkpoint/DPOT-Ti_smoke/ so a smoke
# run is never mistaken for a finished real run.
SMOKE_TEST = False
SMOKE_BATCHES = 3

DATA_ROOT = REPO_ROOT / "data" / "Data"
DPOT_WEIGHTS = REPO_ROOT / "models" / "DPOT-Ti" / "model_Ti.pth"
OUT_ROOT = REPO_ROOT / "checkpoint" / ("DPOT-Ti_smoke" if SMOKE_TEST else "DPOT-Ti")

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

# The pretrained DPOT-Ti configuration, from the model card (hf:README.md:33, also DPOT/README.md:29).
# act and time_agg are DPOTNet's defaults (DPOT/models/dpot.py:247), which pretraining also used
# (DPOT/train_temporal_parallel.py:130 passes neither); they are written out so best.pt is complete.
# NOT IN PAPER: both variants keep this architecture unchanged (4 channels, 10 input frames, 128x128),
# and our data is padded and resampled to fit it.
DPOT_CONFIG = {
    "img_size": 128,
    "patch_size": 8,
    "mixing_type": "afno",
    "in_channels": 4,
    "out_channels": 4,
    "in_timesteps": 10,
    "out_timesteps": 1,
    "normalize": False,  # no normalisation inside the model (DPOT/models/dpot.py:366-370, 400-401)
    "embed_dim": 512,
    "modes": 32,
    "depth": 4,
    "n_blocks": 4,
    "mlp_ratio": 1,
    "out_layer_dim": 32,
    "n_cls": 12,
    "act": "gelu",
    "time_agg": "exp_mlp",
}
MODEL_CHANNELS = DPOT_CONFIG["in_channels"]
MODEL_FRAMES = DPOT_CONFIG["in_timesteps"]
MODEL_RESOLUTION = DPOT_CONFIG["img_size"]

# NOT IN PAPER: DPOT sees all 5 input frames, padded to the 10 it was pretrained with by repeating the
# first one: [x0, x0, x0, x0, x0, x0, x1, x2, x3, x4]. The time-aggregation weight has one slice per
# frame, shape (10, 512, 512) (DPOT/models/dpot.py:221-224), and the frame positions enter through
# linspace(0, 1, T) (DPOT/models/dpot.py:230, 356-357), so T must stay 10 for the pretrained weights.
# DPOT's own fine-tuning always uses 10 frames (DPOT/finetune.py:92).
FRAMES_USED = list(range(INPUT_FRAMES))
TIME_PADDING = "repeat the first input frame: [x0]*6 + [x1, x2, x3, x4]"

# NOT IN PAPER: the C real channels are followed by 4 - C channels filled with 1.0 (in normalised
# space), as DPOT pads every dataset to 4 channels with ones (DPOT/utils/griddataset.py:98-99,
# n_channels=4 from DPOT/finetune.py:101). The loss uses only the C real output channels.
CHANNEL_PAD_VALUE = 1.0

# NOT IN PAPER: DPOT only runs at its img_size (assert in DPOT/models/dpot.py:206; pos_embed is
# (1, 512, 16, 16), DPOT/models/dpot.py:280) and has no resampling of its own. We Fourier-upsample the
# inputs 64 -> 128 and Fourier-truncate the prediction 128 -> 64 (fourier_resample below), and compute
# the loss at 64x64. DPOT itself resizes data with bilinear interpolation (DPOT/utils/griddataset.py:96).
RESAMPLING = (
    "ours: input 64->128 by Fourier zero-padding, output 128->64 by Fourier truncation "
    "(FFT with norm='forward', Nyquist mode split / summed so that down(up(x)) == x)"
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


def fourier_resample(x, size):
    """(..., H, W) real periodic fields -> (..., size, size). Never bilinear or F.interpolate."""
    return fourier_resize_axis(fourier_resize_axis(x, size, -2), size, -1).real


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        # torch 2.0.1 may not support, on MPS, the FFT/complex ops used by our resampling and by DPOT's
        # AFNO layers (DPOT/models/dpot.py:59, 100, 102). Try them once and fall back to the CPU.
        try:
            x = torch.randn(1, 1, 8, 8, device="mps")
            fourier_resample(fourier_resample(x, 16), 8)
            spec = torch.view_as_complex(torch.view_as_real(torch.fft.rfft2(x, norm="ortho")))
            torch.fft.irfft2(spec, s=(8, 8), norm="ortho")
            return torch.device("mps")
        except Exception as err:
            print(f"MPS cannot run the FFT ops ({type(err).__name__}), using the CPU.")
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
    # (PDE, mix) only. The dataset API does not normalise, and neither does DPOT-Ti inside the model
    # (normalize=False, see DPOT_CONFIG; DPOT fine-tunes on raw data, DPOT/finetune.py:125-126), so the
    # data is normalised exactly once, here.
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


# ============================================================================= model
def build_model(variant):
    model = DPOTNet(**DPOT_CONFIG)
    if variant == "pretrained":
        load_pretrained(model)
    # scratch: DPOTNet's own random init (pos_embed trunc_normal, DPOT/models/dpot.py:325; AFNO and
    # time-aggregation weights, DPOT/models/dpot.py:45-48, 221-224; PyTorch defaults elsewhere).
    # DPOTNet._init_weights (DPOT/models/dpot.py:329-337) is never applied, in DPOT's scripts or here.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,} ({n_params / 1e6:.1f}M, DPOT-Ti is ~7.5M)")
    return model


def load_pretrained(model):
    # model_Ti.pth = {"args", "model", "optimizer"} as saved by DPOT/train_temporal_parallel.py:307.
    # "args" is a pickled argparse.Namespace, hence weights_only=False.
    ckpt = torch.load(DPOT_WEIGHTS, map_location="cpu", weights_only=False)
    print(f"  model_Ti.pth keys: {sorted(ckpt)}")
    # Strip a DDP "module." prefix if present, as DPOT/utils/utilities.py:119-125 does.
    state = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in ckpt["model"].items()}
    # Every tensor is loaded, as on the model card (hf:README.md:34) and in DPOT's fine-tuning with
    # load_components ['all'] (DPOT/configs/dpot_finetune.yaml:42, DPOT/utils/utilities.py:126-128).
    result = model.load_state_dict(state, strict=False)
    print(f"  missing keys: {result.missing_keys}")
    print(f"  unexpected keys: {result.unexpected_keys}")
    assert not result.missing_keys and not result.unexpected_keys

    # Nothing is replaced: every tensor must be exactly the one in model_Ti.pth.
    model_state = model.state_dict()
    assert set(model_state) == set(state)
    differ = {k for k in state if not torch.equal(model_state[k], state[k])}
    assert not differ, f"tensors differing from the checkpoint: {sorted(differ)}"


def to_dpot_input(x):
    """Normalised frames (B, 5, C, 64, 64) -> DPOT input (B, 128, 128, 10, 4)."""
    x = fourier_resample(x, MODEL_RESOLUTION)  # (B, 5, C, 128, 128)
    first = x[:, :1].expand(-1, MODEL_FRAMES - INPUT_FRAMES, -1, -1, -1)
    x = torch.cat([first, x], dim=1)  # (B, 10, C, 128, 128): [x0]*6 + [x1, x2, x3, x4]
    b, t, c, h, w = x.shape
    pad = torch.full((b, t, MODEL_CHANNELS - c, h, w), CHANNEL_PAD_VALUE, device=x.device)
    x = torch.cat([x, pad], dim=2)  # (B, 10, 4, 128, 128)
    return x.permute(0, 3, 4, 1, 2)  # DPOT layout (B, X, Y, T, C), DPOT/models/dpot.py:363-365


def predict(model, x):
    # (B, 5, C, 64, 64) -> (B, C, 64, 64). forward returns (prediction, cls_pred)
    # (DPOT/models/dpot.py:403). cls_pred is ignored: it never had a loss weight (DPOT/
    # train_temporal_parallel.py:243, DPOT/finetune.py:239). The prediction is (B, X, Y, 1, 4)
    # (DPOT/models/dpot.py:397-398); only the C real channels are kept, so the padded ones get no loss.
    n_channels = x.shape[2]
    pred, _ = model(to_dpot_input(x))
    pred = pred[:, :, :, 0, :n_channels].permute(0, 3, 1, 2)  # (B, C, 128, 128)
    return fourier_resample(pred, RESOLUTION)


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
    return {
        "variant": variant,
        "pde": pde,
        "mix": mix,
        "n_channels": data["n_channels"],  # real channels = the first n_channels of the model's 4
        "dpot_config": dict(DPOT_CONFIG),
        "frames_used": FRAMES_USED,  # indices into x (input_frames, C, H, W): all 5 input frames
        "time_padding": TIME_PADDING,
        "channel_pad_value": CHANNEL_PAD_VALUE,  # value of the 4 - n_channels padded input channels
        "resampling": RESAMPLING,
        "norm_mean": data["mean"].tolist(),  # per channel; model space = (field - mean) / std
        "norm_std": data["std"].tolist(),
        "pretrained_weights_sha256": weights_sha256,
        "dpot_commit": DPOT_COMMIT,
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
        # NOT IN PAPER: no input noise. DPOT pretrained with noise (DPOT/train_temporal_parallel.py:220,
        # noise_scale 0.0005 in DPOT/configs/pretrain_tiny.yaml:71), but its fine-tuning example uses
        # none (noise_scale [0.0] in DPOT/configs/dpot_finetune.yaml:77, which overrides line 16 via
        # DPOT/trainer.py:41-58).
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
    weights_sha256 = sha256_of(DPOT_WEIGHTS)

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

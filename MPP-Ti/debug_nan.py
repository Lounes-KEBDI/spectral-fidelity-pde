"""Diagnoses the non-finite gradient of MPP-Ti/finetune.py on pretrained / Mix-simple / gray_scott, and checks
the fix finetune.py now applies (flat train windows excluded + safety net, step 9).

Usage, from the repo root on Lab-IA (one GPU, at most 30 minutes, log in logs/debug-mpp-nan-<job id>.out):
    sbatch --qos=debug --partition=testing --time=00:30:00 --gres=gpu:1 --cpus-per-task=4 \
        --job-name=debug-mpp-nan --output=logs/%x-%j.out \
        --wrap='. /mnt/beegfs/projects/ftctinfer/conda_stagiaires/etc/profile.d/conda.sh && conda activate "$SLURM_SUBMIT_DIR/.venv-labia" && cd "$SLURM_SUBMIT_DIR" && export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 && nvidia-smi && python MPP-Ti/debug_nan.py'

It never writes anything: last.pt / best.pt are only read, and finetune.py is not modified. finetune.py is loaded
as a module so that its own functions build the data, the normalisation and the model (main() only runs under
__main__, so loading it starts no training).

Steps 1-8 replay the crash as finetune.py ran it before the fix: every train window, no safety net.
  0. scan the train and val windows for (near-)constant fields;
  1. rebuild exactly the state the crashed run started epoch 1 from (seed, pretrained model, AdamW, schedule);
  2. compare it with checkpoint/MPP-Ti/pretrained/Mix-simple/gray_scott/last.pt (is resuming involved?);
  3. probe the pretrained model on windows with a spatially uniform frame (output and gradient finite?);
  4. val loss of the pretrained model, batch by batch, before any training;
  5. epoch 1 as the crashed run ran it, batch by batch, stopping at the first non-finite loss, gradient or
     weight, then the val loss after epoch 1 (finetune.py asserts on train AND val loss);
  6. the same epoch from the same start with gradient clipping at 1.0 (MPP's setting, MPP/train_basic.py:266);
  7. replay steps 0-4 of epoch 1, then step 5 (the batch with window 161) under torch.autograd anomaly
     detection, with forward and backward hooks on every module: the first module whose backward produces a
     non-finite value, and max |grad| per module along the backward path;
  8. replay steps 0-4 again, then step 5 without window 161, and window 161 alone.
Steps 9 and 10 run epoch 1 with the fix, through finetune.py's own exclude_flat_windows, train_epoch and
evaluate: train windows with an input frame flat in every channel at once excluded + safety net. Both also
count the windows each of the three criteria tried would flag.
  9. gray_scott / Mix-simple: finite to the end? skipped steps, val loss, are windows 161 and 160, 640, 1260
     (start 0 of the trajectories that decay to a flat state) excluded? Every skipped batch is listed with its
     windows' flatness scores (min over the input frames of the max over channels of the spatial std).
 10. wave / Mix-balance: the same, and the windows starting at frame 0 (wave velocity identically 0 at t=0)
     must stay in training.
Steps 3 and 4 run in eval mode and use no random numbers, so steps 5 and 7 replay the crashed run exactly.
"""

import importlib.util
import math
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

HERE = Path(__file__).resolve().parent
VARIANT, MIX, PDE = "pretrained", "Mix-simple", "gray_scott"  # the run that crashed
CLIP_NORM = 1.0  # MPP's official gradient clipping (MPP/train_basic.py:266)
TINY_STD = 1e-3  # flag windows / frames whose std, in our normalised units, is below this
PRINT_EVERY = 20  # progress line every N training steps
TARGET_STEP = 5  # the step of epoch 1 that gave the first non-finite gradient
TARGET_WINDOW = 161  # the near-constant window in that batch (IC-simple, trajectory 16, start 1)
# Start 0 of the gray_scott trajectories that decay to a flat state: frame 0 still has structure, later frames
# are flat in both channels. The whole-window criterion kept them, and epoch 1 still had 2 non-finite steps.
DECAYING_START_WINDOWS = [160, 640, 1260]
WAVE_PDE, WAVE_MIX = "wave", "Mix-balance"  # step 10: its windows starting at frame 0 must stay in training


def load_finetune():
    spec = importlib.util.spec_from_file_location("mpp_finetune", HERE / "finetune.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # definitions and setup only: main() is behind __name__ == "__main__"
    return module


ft = load_finetune()
assert not ft.SMOKE_TEST, "set SMOKE_TEST = False in MPP-Ti/finetune.py: the crash only shows on the full data"


# ============================================================================= helpers
class NormWatcher:
    """Records, for every RMSInstanceNorm2d of MPP (MPP/models/spatial_modules.py:26-40), the smallest spatial
    std of its input and the largest |output| during the last forward pass."""

    def __init__(self, model):
        self.stats = {}
        for name, module in model.named_modules():
            if type(module).__name__ == "RMSInstanceNorm2d":
                module.register_forward_hook(self._hook(name))

    def _hook(self, name):
        def hook(module, inputs, output):
            with torch.no_grad():
                x = inputs[0]
                min_std = x.std(dim=(-2, -1)).min().item()
                self.stats[name] = (min_std, tuple(x.shape[-2:]), output.abs().max().item())

        return hook

    def reset(self):
        self.stats = {}

    def smallest(self):
        name, (std, grid, out) = min(self.stats.items(), key=lambda kv: kv[1][0])
        return f"{std:.3e} in {name} (grid {grid[0]}x{grid[1]}, max |output| {out:.3e})"

    def report(self, k=5):
        for name, (std, grid, out) in sorted(self.stats.items(), key=lambda kv: kv[1][0])[:k]:
            print(f"    {name:40s} grid {grid[0]}x{grid[1]}  min input std {std:.3e}  max |output| {out:.3e}")


def tensor_stats(tensors):
    """(max |finite value|, number of non-finite values) over a list of tensors, or None if there are none."""
    tensors = [t.detach() for t in tensors if isinstance(t, torch.Tensor) and t.is_floating_point()]
    if not tensors:
        return None
    max_abs, n_bad = 0.0, 0
    for t in tensors:
        finite = torch.isfinite(t)
        n_bad += int((~finite).sum())
        if finite.any():
            max_abs = max(max_abs, t[finite].abs().max().item())
    return max_abs, n_bad


def fmt(stats):
    if stats is None:
        return "-".rjust(22)
    max_abs, n_bad = stats
    return f"{max_abs:10.3e}" + (f" ({n_bad} NON-FINITE)" if n_bad else " " * 12)


class BackwardTracer:
    """Forward and full backward hooks on every module. Forward: max |output| of each call. Backward: one record
    per module call, in the order the backward pass reaches it, with max |grad| w.r.t. the module's output and
    w.r.t. its input, and the number of non-finite values in each."""

    def __init__(self, model):
        self.forward_max = {}
        self.records = []
        self.handles = []
        for name, module in model.named_modules():
            name = name or "(whole model)"
            self.handles.append(module.register_forward_hook(self._forward_hook(name)))
            self.handles.append(module.register_full_backward_hook(self._backward_hook(name)))

    def _forward_hook(self, name):
        def hook(module, inputs, output):
            stats = tensor_stats(output if isinstance(output, (tuple, list)) else [output])
            if stats is not None:
                previous = self.forward_max.get(name, (0.0, 0))
                self.forward_max[name] = (max(previous[0], stats[0]), previous[1] + stats[1])

        return hook

    def _backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            self.records.append((name, tensor_stats(grad_output), tensor_stats(grad_input)))

        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()

    def report(self):
        print(f"  backward path, {len(self.records)} module calls in the order backward reaches them:")
        print(f"    {'#':>4}  {'module':55s}  {'max|grad wrt output|':>22}  {'max|grad wrt input|':>22}  {'fwd max|output|':>22}")
        for k, (name, grad_out, grad_in) in enumerate(self.records):
            print(f"    {k:4d}  {name:55s}  {fmt(grad_out)}  {fmt(grad_in)}  {fmt(self.forward_max.get(name))}")
        first = next(
            (
                (k, name)
                for k, (name, grad_out, grad_in) in enumerate(self.records)
                if grad_in is not None and grad_in[1] > 0 and (grad_out is None or grad_out[1] == 0)
            ),
            None,
        )
        if first is not None:
            print(f"  FIRST module whose backward turns finite gradients into non-finite ones: #{first[0]} {first[1]}")
        elif any(r[1] is not None and r[1][1] > 0 for r in self.records):
            print("  non-finite gradients already reach the first module (they come from the loss itself)")
        else:
            print("  no module produced a non-finite gradient in this backward pass")
        largest = sorted(
            ((grad_in[0], k, name) for k, (name, _, grad_in) in enumerate(self.records) if grad_in is not None),
            reverse=True,
        )[:10]
        print("  largest max|grad wrt input|: " + ", ".join(f"#{k} {name} {v:.2e}" for v, k, name in largest))
        return first


def gradient_report(model, label):
    """Parameter gradients after a backward pass: non-finite entries, norms in float32 (as run_epoch computes
    them, which overflows above ~1.8e19) and in float64, and the largest |grad| per parameter."""
    rows, n_bad, bad_tensors = [], 0, []
    norms32, norms64 = [], []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        stats = tensor_stats([p.grad])
        rows.append((stats[0], name))
        if stats[1]:
            n_bad += stats[1]
            bad_tensors.append(name)
        norms32.append(p.grad.norm())
        norms64.append(p.grad.double().norm())
    if not rows:
        print(f"  {label}: no parameter received a gradient (the backward pass stopped before any)")
        return "no parameter gradient computed"
    norm32 = torch.norm(torch.stack(norms32)).item()
    norm64 = torch.norm(torch.stack(norms64)).item()
    print(f"  {label}: {n_bad} non-finite gradient entries in {len(bad_tensors)} parameters; "
          f"grad norm float32 {norm32:.3e}, float64 {norm64:.3e}")
    if bad_tensors:
        print(f"    parameters with non-finite gradients, e.g. {bad_tensors[:8]}")
    print("    largest finite |grad|: " + ", ".join(f"{name} {v:.2e}" for v, name in sorted(rows, reverse=True)[:8]))
    finite = n_bad == 0 and math.isfinite(norm64)
    return f"{'finite' if finite else 'NON-FINITE'} ({n_bad} non-finite entries, grad norm float32 {norm32:.3e}, float64 {norm64:.3e})"


def window_of(ds, i):
    """Trajectory entry and window start of window i of a benchmark_api dataset (benchmark_api.py:178-180)."""
    return ds.entries[i // len(ds.starts)], ds.starts[i % len(ds.starts)]


def finite_range(t):
    finite = t[torch.isfinite(t)]
    n_bad = t.numel() - finite.numel()
    if finite.numel() == 0:
        return f"no finite value ({n_bad} non-finite)"
    return f"min {finite.min().item():+.3e}, max {finite.max().item():+.3e}, {n_bad} non-finite"


def describe_batch(xb, indices, ds):
    """Per window and channel: min / max / std of our normalised input over (T, H, W), the std MPP divides
    by in its per-sample normalisation (+1e-7, MPP/models/avit.py:106-107), and the smallest spatial std of
    a single frame."""
    xt = xb.permute(1, 0, 2, 3, 4)  # (T, B, C, H, W), as predict() feeds MPP
    mpp_std = (torch.std_mean(xt, dim=(0, -2, -1), keepdim=True)[0] + 1e-7)[0, :, :, 0, 0]
    dims = (1, 3, 4)
    stats = [xb.amin(dims), xb.amax(dims), xb.std(dims), mpp_std, xb.std(dim=(3, 4)).amin(1)]
    stats = [s.cpu().tolist() for s in stats]
    for b, i in enumerate(indices.tolist()):
        entry, start = window_of(ds, i)
        cols = "  |  ".join(
            f"c{c}: min {stats[0][b][c]:+.3e} max {stats[1][b][c]:+.3e} std {stats[2][b][c]:.3e} "
            f"mpp_std {stats[3][b][c]:.3e} min_frame_std {stats[4][b][c]:.3e}"
            for c in range(xb.shape[2])
        )
        print(f"    window {i:4d} ({entry.ic_name}/{entry.dynamic_name} traj {entry.sample_index} start {start})  {cols}")


def scan(name, x, ds, k=8):
    """Smallest whole-window std (what MPP's per-sample normalisation divides by) and smallest single-frame
    spatial std, per window and channel."""
    with torch.no_grad():
        window_std = torch.cat([chunk.std(dim=(1, 3, 4)) for chunk in x.split(256)])  # (N, C)
        frame_std = torch.cat([chunk.std(dim=(3, 4)).amin(1) for chunk in x.split(256)])  # (N, C)
    n_channels = x.shape[2]
    for label, s in [("whole-window std", window_std), ("smallest single-frame spatial std", frame_std)]:
        flat = s.flatten().cpu()
        print(
            f"  {name}, {label}: min {flat.min().item():.3e}, median {flat.median().item():.3e}, "
            f"{int((flat < TINY_STD).sum())} (window, channel) pairs below {TINY_STD:g}"
        )
        for j in flat.argsort()[:k].tolist():
            i, c = divmod(j, n_channels)
            entry, start = window_of(ds, i)
            print(f"    window {i:4d} c{c}: {flat[j].item():.3e}  ({entry.ic_name} traj {entry.sample_index} start {start})")


def setup(device, data):
    """The model, optimizer, scheduler and shuffle generator exactly as finetune.py builds them before
    epoch 1 (MPP-Ti/finetune.py:526-534), for the train windows in data."""
    ft.seed_everything(ft.SEED)
    model = ft.build_model(VARIANT, data["n_channels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ft.LR, weight_decay=ft.WEIGHT_DECAY)
    total_steps = math.ceil(len(data["x_train"]) / ft.BATCH_SIZE) * ft.MAX_EPOCHS
    warmup_steps = round(ft.WARMUP_FRACTION * total_steps)
    scheduler = LambdaLR(optimizer, ft.cosine_with_warmup(warmup_steps, total_steps))
    shuffle_gen = torch.Generator().manual_seed(ft.SEED)
    print(f"  schedule: {total_steps} steps over {ft.MAX_EPOCHS} epochs, {warmup_steps} warmup steps")
    return model, optimizer, scheduler, shuffle_gen


def same_rng(a, b):
    cuda_same = (a["cuda"] is None and b["cuda"] is None) or (
        a["cuda"] is not None
        and b["cuda"] is not None
        and len(a["cuda"]) == len(b["cuda"])
        and all(torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"]))
    )
    return torch.equal(a["torch"], b["torch"]) and torch.equal(a["shuffle"], b["shuffle"]) and cuda_same


def check_resume(model, scheduler, fresh_rng, data):
    run_dir = ft.OUT_ROOT / VARIANT / MIX / PDE
    print(f"\n[2] Resume check: {run_dir}")
    for name in ["last.pt", "best.pt", "train_log.csv"]:
        print(f"  {name}: {'present' if (run_dir / name).exists() else 'absent'}")
    if (run_dir / "train_log.csv").exists():
        print("  train_log.csv:\n    " + (run_dir / "train_log.csv").read_text().strip().replace("\n", "\n    "))
    last_path = run_dir / "last.pt"
    if not last_path.exists():
        return "no last.pt: resuming is not involved"

    last = torch.load(last_path, map_location="cpu", weights_only=False)
    progress = last["progress"]
    print(
        f"  last.pt: epoch {last['epoch']}, val_loss {last['val_loss']}, next_epoch {progress['next_epoch']}, "
        f"best_epoch {progress['best_epoch']}, bad_epochs {progress['bad_epochs']}, "
        f"{len(progress['history'])} logged epochs"
    )
    fresh = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    saved = last["model_state_dict"]
    differ = [k for k in fresh if k not in saved or saved[k].shape != fresh[k].shape or not torch.equal(saved[k], fresh[k])]
    nonfinite = [k for k, v in saved.items() if v.is_floating_point() and not torch.isfinite(v).all()]
    optimizer_steps = len(last["optimizer_state_dict"]["state"])  # 0 if AdamW never took a step
    checks = {
        "weights identical to the fresh pretrained + expanded model": not differ and set(saved) == set(fresh),
        "all saved weights finite": not nonfinite,
        "optimizer state empty (no step taken)": optimizer_steps == 0,
        "scheduler at step 0": last["scheduler_state_dict"]["last_epoch"] == scheduler.state_dict()["last_epoch"],
        "RNG states identical (torch, CUDA, shuffle)": same_rng(last["rng_state"], fresh_rng),
        "same normalisation constants": np.allclose(last["norm_mean"], data["mean"])
        and np.allclose(last["norm_std"], data["std"]),
        "same state labels / bcs": last["state_labels"] == ft.state_labels(data["n_channels"]) and last["bcs"] == ft.BCS,
    }
    for name, ok in checks.items():
        print(f"  [{'yes' if ok else 'NO '}] {name}")
    if differ:
        print(f"    {len(differ)} differing tensors, e.g. {differ[:5]}")
    if nonfinite:
        print(f"    non-finite saved tensors, e.g. {nonfinite[:5]}")
    if all(checks.values()):
        return "last.pt is exactly the fresh epoch-0 state: resuming replays the same epoch 1, so it is not the cause"
    return "last.pt differs from the fresh epoch-0 state (see above): resuming may be involved"


def probe_uniform_frames(model, watcher, x_val):
    """Pretrained model in eval mode on 4 real val windows, then on the same windows with one frame of one
    channel made spatially uniform, then with every channel constant over the window. For each: is the output
    finite, and is the gradient of a loss on it finite? No random numbers; gradients are cleared afterwards."""
    model.eval()
    with torch.no_grad():
        xb = x_val[:4].clone()
        one_frame = xb.clone()
        one_frame[:, 0, 0] = one_frame[:, 0, 0].mean(dim=(-2, -1), keepdim=True)
        constant = xb.clone()
        constant[:] = xb.mean(dim=(1, 3, 4), keepdim=True)
    results = {}
    for name, x in [
        ("4 real val windows", xb),
        ("same, frame 0 of channel 0 made uniform", one_frame),
        ("same, every channel constant over the window", constant),
    ]:
        watcher.reset()
        model.zero_grad(set_to_none=True)
        pred = ft.predict(model, x)
        F.mse_loss(pred, x[:, -1]).backward()  # any scalar loss works; target = last input frame
        output_ok = bool(torch.isfinite(pred).all())
        grads_ok = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        results[name] = output_ok and grads_ok
        print(f"  {name}: output {'finite' if output_ok else 'NON-FINITE'} ({finite_range(pred.detach())}), "
              f"gradients {'finite' if grads_ok else 'NON-FINITE'}; smallest RMSInstanceNorm2d input std "
              f"{watcher.smallest()}")
    model.zero_grad(set_to_none=True)
    return results


@torch.no_grad()
def check_val(model, watcher, x_val, y_val, val_ds, label):
    """finetune.py's evaluate() (MPP-Ti/finetune.py:429-436), batch by batch, stopping at the first
    non-finite batch."""
    model.eval()
    squared_error = 0.0
    for i in range(0, len(x_val), ft.BATCH_SIZE):
        xb, yb = x_val[i : i + ft.BATCH_SIZE], y_val[i : i + ft.BATCH_SIZE]
        watcher.reset()
        pred = ft.predict(model, xb)
        batch_error = F.mse_loss(pred, yb, reduction="sum").item()
        if not math.isfinite(batch_error):
            print(f"  {label}: NON-FINITE at val batch {i // ft.BATCH_SIZE} (windows {i}..{i + len(xb) - 1})")
            print(f"    output: {finite_range(pred)}")
            print("    norm layers with the smallest input std:")
            watcher.report()
            print("    windows of that batch (our normalised input):")
            describe_batch(xb, torch.arange(i, i + len(xb)), val_ds)
            return f"non-finite at val batch {i // ft.BATCH_SIZE}"
        squared_error += batch_error
    val_loss = squared_error / y_val.numel()
    print(f"  {label}: val loss {val_loss:.4e}, every batch finite")
    return f"finite, val loss {val_loss:.4e}"


def run_epoch(model, optimizer, scheduler, shuffle_gen, watcher, x_train, y_train, train_ds, clip=None):
    """Epoch 1 as finetune.py ran it before the fix (every train window, no safety net), batch by batch.
    With clip, gradients are clipped
    to that norm before each step, as MPP does. Stops at the first non-finite loss, gradient or weight."""
    model.train()
    order = torch.randperm(len(x_train), generator=shuffle_gen).to(x_train.device)
    params = list(model.named_parameters())
    grad_norms, losses = [], []
    for step, i in enumerate(range(0, len(order), ft.BATCH_SIZE)):
        idx = order[i : i + ft.BATCH_SIZE]
        lr = optimizer.param_groups[0]["lr"]
        watcher.reset()
        pred = ft.predict(model, x_train[idx])
        loss = F.mse_loss(pred, y_train[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads = [p.grad for _, p in params if p.grad is not None]
        grad_norm = torch.norm(torch.stack([g.norm() for g in grads])).item()  # before clipping

        problem = None
        if not math.isfinite(loss.item()):
            problem = "loss"
        elif not math.isfinite(grad_norm):
            problem = "gradient"
        if problem is None:
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            scheduler.step()
            if not bool(torch.stack([torch.isfinite(p).all() for _, p in params]).all()):
                problem = "weights after the optimizer step"

        if problem is not None:
            print(f"  FIRST NON-FINITE {problem.upper()} at step {step} (batch {step} of epoch 1), lr {lr:.3e}, loss {loss.item()}")
            print(f"    gradient norm of this step (before clipping): {grad_norm}")
            bad = [n for n, p in params if p.grad is not None and not torch.isfinite(p.grad).all()]
            if bad:
                print(f"    {len(bad)} parameters with non-finite gradients, e.g. {bad[:6]}")
            print(f"    output: {finite_range(pred.detach())}")
            print("    norm layers with the smallest input std in this forward pass:")
            watcher.report()
            print("    windows of this batch (our normalised input):")
            describe_batch(x_train[idx], idx.cpu(), train_ds)
            if grad_norms:
                worst = int(np.argmax(grad_norms))
                print(f"    gradient norms of the previous {len(grad_norms)} steps: max {grad_norms[worst]:.3e} at step "
                      f"{worst}, median {np.median(grad_norms):.3e}, last ones "
                      + ", ".join(f"{g:.2e}" for g in grad_norms[-15:]))
                print("    losses of the last steps: " + ", ".join(f"{v:.3e}" for v in losses[-10:]))
            return f"first non-finite {problem} at step {step} (lr {lr:.2e})"

        grad_norms.append(grad_norm)
        losses.append(loss.item())
        if step % PRINT_EVERY == 0:
            print(f"  step {step:3d}  lr {lr:.2e}  loss {loss.item():.3e}  grad norm {grad_norm:.3e}  "
                  f"smallest norm input std {watcher.smallest()}")

    print(f"  epoch 1 finished: mean train loss {np.mean(losses):.4e}, max grad norm {max(grad_norms):.3e} "
          f"(step {int(np.argmax(grad_norms))}), median {np.median(grad_norms):.3e}")
    return f"finite over all {len(losses)} steps, mean train loss {np.mean(losses):.4e}"


def replay_to_step(device, data, x_train, y_train, step):
    """Fresh start (setup), then training steps 0 .. step-1 of epoch 1 exactly as the crashed run took them
    (every train window, no safety net). Returns the model, ready for `step`, and the epoch's shuffled order."""
    model, optimizer, scheduler, shuffle_gen = setup(device, data)
    model.train()
    order = torch.randperm(len(x_train), generator=shuffle_gen).to(x_train.device)
    for s in range(step):
        idx = order[s * ft.BATCH_SIZE : (s + 1) * ft.BATCH_SIZE]
        loss = F.mse_loss(ft.predict(model, x_train[idx]), y_train[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
    print(f"  replayed steps 0-{step - 1}; step {step} runs at lr {optimizer.param_groups[0]['lr']:.3e}")
    return model, order


def criteria_counts(x):
    """Train windows flagged by each of the three criteria tried, for comparison. Only the last one is used
    (finetune.py's flat_windows)."""
    with torch.no_grad():
        frame_std = torch.cat([chunk.std(dim=(3, 4)) for chunk in x.split(256)])  # (N, 5, C)
        window_std = torch.cat([chunk.std(dim=(1, 3, 4)) for chunk in x.split(256)])  # (N, C)
    threshold = ft.FLAT_STD_THRESHOLD
    counts = {
        "1st, a frame flat in ANY channel (ruled out)": int((frame_std.amin(dim=(1, 2)) < threshold).sum()),
        "2nd, ANY channel flat over the whole window (ruled out)": int((window_std.amin(dim=1) < threshold).sum()),
        "3rd, a frame flat in EVERY channel (used)": int(ft.flat_windows(x).sum()),
    }
    for name, n in counts.items():
        print(f"    {name}: {n} of {len(x)} train windows")
    return counts


def epoch_with_fix(device, data):
    """Epoch 1 as finetune.py now runs it for the (PDE, mix) in data: exclude_flat_windows, the same setup
    (MPP-Ti/finetune.py:526-534), then train_epoch (safety net included) and evaluate on every val window.
    data itself is not modified. Returns (result line, indices of the excluded train windows, skipped batches),
    each skipped batch being (step, [(window, flatness score), ...]) as train_epoch records it."""
    fixed = dict(data)  # exclude_flat_windows replaces the train tensors in this copy only
    excluded = ft.exclude_flat_windows(fixed)
    print(f"  flat train windows excluded: {len(excluded)} of {len(data['x_train'])}: {excluded}")
    model, optimizer, scheduler, shuffle_gen = setup(device, fixed)
    x_train, y_train = fixed["x_train"].to(device), fixed["y_train"].to(device)
    x_val, y_val = fixed["x_val"].to(device), fixed["y_val"].to(device)
    skipped = []
    try:
        train_loss, skipped, n_steps = ft.train_epoch(
            model, optimizer, scheduler, shuffle_gen, x_train, y_train, epoch=1, window_ids=fixed["train_window_ids"]
        )
        val_loss = ft.evaluate(model, x_val, y_val)
        finite = math.isfinite(train_loss) and math.isfinite(val_loss)
        result = (f"{'finite' if finite else 'NON-FINITE'} to the end of epoch 1: train loss {train_loss:.4e}, "
                  f"val loss {val_loss:.4e} (all {len(x_val)} val windows), skipped steps {len(skipped)}/{n_steps}, "
                  f"{len(excluded)} train windows excluded")
    except RuntimeError as err:  # raised by train_epoch when more than MAX_SKIPPED_FRACTION of the steps are skipped
        result = f"ABORTED by the safety net (its skipped batches are printed above): {err}"
    print(f"  {result}")
    if skipped:
        print(f"  skipped batches, windows flattest first (flatness score; excluded below {ft.FLAT_STD_THRESHOLD:g}):")
        for step, windows in skipped:
            print(f"    step {step}: " + ", ".join(f"{w} ({s:.2e})" for w, s in windows))
    del model, optimizer, scheduler, x_train, y_train, x_val, y_val
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, excluded, skipped


def describe_skipped(skipped):
    """One summary line: each skipped step with its flattest window and that window's flatness score."""
    if not skipped:
        return "none"
    return "; ".join(f"step {step}: flattest window {windows[0][0]} ({windows[0][1]:.2e})" for step, windows in skipped)


def get_torch_rng():
    return torch.get_rng_state(), (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def set_torch_rng(state):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


# ============================================================================= main
def main():
    torch.backends.cuda.matmul.allow_tf32 = False  # as finetune.py's main()
    torch.backends.cudnn.allow_tf32 = False
    device = ft.pick_device()
    print(f"Device: {device}. Run: {VARIANT} / {MIX} / {PDE}")
    summary = {}

    data = ft.load_windows(PDE, MIX)
    train_ds, val_ds = ft.make_train_val_datasets(
        ft.DATA_ROOT, PDE, MIX, input_frames=ft.INPUT_FRAMES, target_frames=ft.TARGET_FRAMES,
        train_context_frames=ft.TRAIN_CONTEXT_FRAMES, window_stride=ft.WINDOW_STRIDE,
    )  # for window metadata only
    print(f"  {len(data['x_train'])} train / {len(data['x_val'])} val windows, channels {data['n_channels']}, "
          f"mean {np.round(data['mean'], 4).tolist()}, std {np.round(data['std'], 4).tolist()}")
    x_train, y_train = data["x_train"].to(device), data["y_train"].to(device)
    x_val, y_val = data["x_val"].to(device), data["y_val"].to(device)

    print("\n[0] Near-constant fields, in our normalised units")
    scan("train", x_train, train_ds)
    scan("val", x_val, val_ds)

    print("\n[1] Rebuilding the state finetune.py starts epoch 1 from")
    model, optimizer, scheduler, shuffle_gen = setup(device, data)
    fresh_rng = ft.get_rng_state(shuffle_gen)  # what finetune.py saves in last.pt before epoch 1
    summary["resume"] = check_resume(model, scheduler, fresh_rng, data)
    watcher = NormWatcher(model)

    print("\n[3] Probe: pretrained model on windows with a spatially uniform frame (eval mode)")
    probe = probe_uniform_frames(model, watcher, x_val)
    summary["probe"] = ", ".join(f"{k}: {'finite' if v else 'NON-FINITE'}" for k, v in probe.items())

    print("\n[4] Val loss of the pretrained model, before any training")
    summary["val before training"] = check_val(model, watcher, x_val, y_val, val_ds, "pretrained model")

    print("\n[5] Epoch 1 as the crashed run ran it (every train window, no safety net, no gradient clipping)")
    summary["epoch 1, no clipping"] = run_epoch(model, optimizer, scheduler, shuffle_gen, watcher, x_train, y_train, train_ds)
    if summary["epoch 1, no clipping"].startswith("finite"):
        summary["val after epoch 1, no clipping"] = check_val(model, watcher, x_val, y_val, val_ds, "after epoch 1")

    print(f"\n[6] Same epoch 1 from the same start, with gradient clipping at {CLIP_NORM}")
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model, optimizer, scheduler, shuffle_gen = setup(device, data)
    watcher = NormWatcher(model)
    summary["epoch 1, clip 1.0"] = run_epoch(
        model, optimizer, scheduler, shuffle_gen, watcher, x_train, y_train, train_ds, clip=CLIP_NORM
    )
    if summary["epoch 1, clip 1.0"].startswith("finite"):
        summary["val after epoch 1, clip 1.0"] = check_val(model, watcher, x_val, y_val, val_ds, "after epoch 1, clipped")
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\n[7] Step {TARGET_STEP} under anomaly detection, with forward and backward hooks on every module")
    model, order = replay_to_step(device, data, x_train, y_train, TARGET_STEP)
    idx = order[TARGET_STEP * ft.BATCH_SIZE : (TARGET_STEP + 1) * ft.BATCH_SIZE]
    print(f"  batch of step {TARGET_STEP}: windows {idx.tolist()}"
          f" ({'contains' if TARGET_WINDOW in idx.tolist() else 'does NOT contain'} window {TARGET_WINDOW})")
    tracer = BackwardTracer(model)
    try:
        with torch.autograd.set_detect_anomaly(True):  # stops backward at the first operation returning NaN
            pred = ft.predict(model, x_train[idx])
            loss = F.mse_loss(pred, y_train[idx])
            print(f"  loss {loss.item():.4e}, output {finite_range(pred.detach())}")
            model.zero_grad(set_to_none=True)
            try:
                loss.backward()
                anomaly = "anomaly detection reported no operation returning NaN"
            except RuntimeError as err:
                anomaly = f"anomaly detection stopped the backward pass: {err}"
        print(f"  {anomaly}")
        print("  (anomaly detection only reacts to NaN; the hooks below also catch inf and show the growth of |grad|)")
        first = tracer.report()
        summary["step 5, anomaly detection"] = anomaly
        summary["step 5, first module with non-finite backward"] = (
            f"#{first[0]} {first[1]}" if first is not None else "none found by the hooks (see the table)"
        )
        summary["step 5, parameter gradients"] = gradient_report(model, f"step {TARGET_STEP}, full batch")
    except Exception:  # keep going to step 8 if the instrumentation itself fails
        traceback.print_exc()
        summary["step 5, instrumented replay"] = "FAILED, see the traceback above"
    tracer.remove()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\n[8] Step {TARGET_STEP} again from the same state, without window {TARGET_WINDOW}, and with it alone")
    model, order = replay_to_step(device, data, x_train, y_train, TARGET_STEP)
    idx = order[TARGET_STEP * ft.BATCH_SIZE : (TARGET_STEP + 1) * ft.BATCH_SIZE]
    rng = get_torch_rng()  # the same random state (drop path masks) for both variants
    for label, sub in [
        (f"batch without window {TARGET_WINDOW}", idx[idx != TARGET_WINDOW]),
        (f"window {TARGET_WINDOW} alone", idx[idx == TARGET_WINDOW]),
    ]:
        if len(sub) == 0:
            print(f"  {label}: empty, skipped")
            continue
        set_torch_rng(rng)
        model.zero_grad(set_to_none=True)
        pred = ft.predict(model, x_train[sub])
        loss = F.mse_loss(pred, y_train[sub])
        loss.backward()
        print(f"  {label} ({len(sub)} windows): loss {loss.item():.4e}, output {finite_range(pred.detach())}")
        summary[f"step 5, {label}"] = gradient_report(model, label)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    del x_train, y_train, x_val, y_val
    print(f"\n[9] {PDE} / {MIX}: epoch 1 with the fix, as finetune.py now runs it")
    print("  windows flagged by each criterion tried:")
    criteria_counts(data["x_train"])
    result, excluded, skipped = epoch_with_fix(device, data)
    for window in [TARGET_WINDOW] + DECAYING_START_WINDOWS:
        print(f"  window {window} {'is' if window in excluded else 'is NOT'} among the excluded windows")
    summary[f"{PDE} / {MIX}, epoch 1 with the fix"] = result
    summary[f"{PDE} / {MIX}, skipped batches"] = describe_skipped(skipped)
    summary[f"{PDE} / {MIX}, windows {[TARGET_WINDOW] + DECAYING_START_WINDOWS} excluded"] = ", ".join(
        f"{w}: {'yes' if w in excluded else 'NO'}" for w in [TARGET_WINDOW] + DECAYING_START_WINDOWS
    )

    print(f"\n[10] {WAVE_PDE} / {WAVE_MIX}: epoch 1 with the fix, as finetune.py now runs it")
    wave = ft.load_windows(WAVE_PDE, WAVE_MIX)
    wave_train_ds, _ = ft.make_train_val_datasets(
        ft.DATA_ROOT, WAVE_PDE, WAVE_MIX, input_frames=ft.INPUT_FRAMES, target_frames=ft.TARGET_FRAMES,
        train_context_frames=ft.TRAIN_CONTEXT_FRAMES, window_stride=ft.WINDOW_STRIDE,
    )  # for window metadata only
    start0 = {i for i in range(len(wave_train_ds)) if window_of(wave_train_ds, i)[1] == 0}
    print(f"  {len(start0)} train windows start at frame 0 (wave velocity identically 0 at t=0)")
    print("  windows flagged by each criterion tried:")
    criteria_counts(wave["x_train"])
    result, excluded, skipped = epoch_with_fix(device, wave)
    start0_excluded = sorted(start0 & set(excluded))
    kept = f"{len(start0) - len(start0_excluded)} of {len(start0)} kept"
    print(f"  [{'PASS' if not start0_excluded else 'FAIL'}] windows starting at frame 0 stay in training: {kept}"
          + (f", excluded: {start0_excluded}" if start0_excluded else ""))
    summary[f"{WAVE_PDE} / {WAVE_MIX}, epoch 1 with the fix"] = result
    summary[f"{WAVE_PDE} / {WAVE_MIX}, skipped batches"] = describe_skipped(skipped)
    summary[f"{WAVE_PDE} / {WAVE_MIX}, windows starting at frame 0"] = kept

    print("\n================ SUMMARY")
    for name, result in summary.items():
        print(f"  {name}: {result}")


if __name__ == "__main__":
    main()

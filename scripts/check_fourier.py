"""Checks the Fourier resampling used by DPOT-Ti/finetune.py and GPhyT-S/finetune.py.

Usage (from anywhere, CPU is enough): python scripts/check_fourier.py

The two training scripts are not imported. Importing them would not start training (main() only runs
under __main__), but it would still load the dataset API and the DPOT / GPhyT model code from
third_party/. Instead, fourier_resize_axis and fourier_resample are read from each script's source with
ast and executed on their own, in a namespace that only holds torch, so nothing else runs.

Checks, on random float32 fields of shape (4, 2, 64, 64):
  1. both scripts define a byte-identical fourier_resize_axis, and their fourier_resample wrappers agree
  2. round trip 64 -> 128 -> 64 (DPOT) and 64 -> 256x128 -> 64 (GPhyT): max absolute and relative error
  3. the upsampled field passes through the original points: x_up[..., ::2, ::2] == x (DPOT),
     x_up[..., ::4, ::2] == x (GPhyT), up to float32 round-off
  4. the radially averaged energy spectrum after the round trip equals the original
Prints PASS or FAIL for each check, and exits with status 1 if any check fails.
"""

import ast
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = {"DPOT-Ti": REPO_ROOT / "DPOT-Ti" / "finetune.py", "GPhyT-S": REPO_ROOT / "GPhyT-S" / "finetune.py"}
FUNCTIONS = ["fourier_resize_axis", "fourier_resample"]  # fourier_resample calls fourier_resize_axis

SHAPE = (4, 2, 64, 64)
SEED = 0
FIELD_TOL = 1e-5  # float32 round-off allowed on field values, relative to max |x|
SPECTRUM_TOL = 1e-4  # allowed relative difference per wavenumber shell of the energy spectrum

results = []


def report(name, passed, detail=""):
    results.append(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def load_functions(path):
    """Reads FUNCTIONS from a script's source and defines them in a namespace that only holds torch.
    Returns ({name: function}, {name: exact source text})."""
    source = path.read_text()
    sources = {
        node.name: ast.get_source_segment(source, node)
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
    }
    missing = set(FUNCTIONS) - set(sources)
    if missing:
        sys.exit(f"{path}: functions not found: {sorted(missing)}")
    namespace = {"torch": torch}
    for name in FUNCTIONS:
        exec(compile(sources[name], str(path), "exec"), namespace)
    return {name: namespace[name] for name in FUNCTIONS}, sources


def radial_spectrum(x):
    """Radially averaged energy spectrum of fields (..., N, N): mean of |FFT|^2 over each integer
    wavenumber shell round(|k|). Returns (..., n_shells), computed in float64."""
    n = x.shape[-1]
    energy = torch.fft.fft2(x.double(), norm="forward").abs() ** 2
    k = torch.fft.fftfreq(n, d=1.0 / n, dtype=torch.float64)  # integer wavenumbers
    kx, ky = torch.meshgrid(k, k, indexing="ij")
    shell = torch.sqrt(kx**2 + ky**2).round().long().flatten()
    counts = torch.bincount(shell).double()
    total = torch.zeros(*x.shape[:-2], len(counts), dtype=torch.float64)
    total.index_add_(-1, shell, energy.flatten(-2))
    return total / counts


def field_errors(a, b):
    """Max absolute error, and that error relative to max |b|."""
    max_abs = (a - b).abs().max().item()
    return max_abs, max_abs / b.abs().max().item()


# ----------------------------------------------------------------------------- 1. same functions
functions, sources = {}, {}
for model, path in SCRIPTS.items():
    functions[model], sources[model] = load_functions(path)

report(
    "fourier_resize_axis is byte-identical in DPOT-Ti and GPhyT-S",
    sources["DPOT-Ti"]["fourier_resize_axis"] == sources["GPhyT-S"]["fourier_resize_axis"],
)
# The fourier_resample wrappers differ in their signature only: DPOT-Ti takes one size (square output),
# GPhyT-S takes a height and a width. Both call fourier_resize_axis on H, then W, and keep the real part.
torch.manual_seed(SEED)
x = torch.randn(*SHAPE, dtype=torch.float32)
dpot_resample = functions["DPOT-Ti"]["fourier_resample"]
gphyt_resample = functions["GPhyT-S"]["fourier_resample"]
report(
    "fourier_resample wrappers give identical results (DPOT size=128 vs GPhyT 128 x 128)",
    torch.equal(dpot_resample(x, 128), gphyt_resample(x, 128, 128)),
    "signatures differ by design: (x, size) vs (x, height, width)",
)

# ----------------------------------------------------------------------------- 2-4. per model
cases = {
    "DPOT-Ti 64 -> 128 -> 64": (lambda f: dpot_resample(f, 128), lambda f: dpot_resample(f, 64), (2, 2)),
    "GPhyT-S 64 -> 256x128 -> 64": (
        lambda f: gphyt_resample(f, 256, 128),
        lambda f: gphyt_resample(f, 64, 64),
        (4, 2),
    ),
}
spectrum_x = radial_spectrum(x)

for name, (up, down, (step_h, step_w)) in cases.items():
    print(f"\n{name}")
    x_up = up(x)
    x_rt = down(x_up)
    assert x_up.dtype == torch.float32 and x_rt.shape == x.shape

    max_abs, max_rel = field_errors(x_rt, x)
    l2_rel = ((x_rt - x).norm() / x.norm()).item()
    print(f"  round trip: max abs error {max_abs:.3e}, relative to max |x| {max_rel:.3e}, relative L2 {l2_rel:.3e}")
    report("round trip returns the original field", max_rel <= FIELD_TOL and l2_rel <= FIELD_TOL)

    points = x_up[..., ::step_h, ::step_w]
    max_abs, max_rel = field_errors(points, x)
    report(
        f"upsampled field passes through the original points (x_up[..., ::{step_h}, ::{step_w}] == x)",
        points.shape == x.shape and max_rel <= FIELD_TOL,
        f"max abs error {max_abs:.3e}, relative to max |x| {max_rel:.3e}",
    )

    spectrum_rt = radial_spectrum(x_rt)
    shell_rel = ((spectrum_rt - spectrum_x).abs() / spectrum_x).max().item()
    report(
        "radially averaged energy spectrum unchanged by the round trip",
        shell_rel <= SPECTRUM_TOL,
        f"max relative difference over {spectrum_x.shape[-1]} shells {shell_rel:.3e}",
    )

print(f"\n{'ALL PASS' if all(results) else 'SOME CHECKS FAILED'}: {sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)

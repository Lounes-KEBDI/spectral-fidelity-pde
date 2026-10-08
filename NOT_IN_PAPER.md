# Choices not in the paper

The paper ("Do Physics Foundation Models Learn Generalizable Physics?", Chu et al., arXiv:2605.29283) fixes the data protocol, but leaves many training details open. This file lists every one of those details that we had to decide ourselves. The code marks each of them with `# NOT IN PAPER`.

Section 1 applies to every model. Each model then gets its own section: Poseidon-T (2), DPOT-Ti (3), GPhyT-S (4) and MPP-Ti (5). Section 6 covers inference.

**How to read the sources**

| Prefix | Refers to |
|---|---|
| `Poseidon-T/...`, `DPOT-Ti/...`, `GPhyT-S/...`, `MPP-Ti/...`, `models/...`, `third_party/download_*.py`, `requirements.txt` (at the repo root, shared by the 4 models) | this repo |
| `data/benchmark_api.py` | the loader shipped with the PhysBiasBench dataset |
| `scOT/...`, `configs/...`, `pyproject.toml`, `README.md` (section 2) | the Poseidon repo, [camlab-ethz/poseidon](https://github.com/camlab-ethz/poseidon) at commit `b8fa28f59bd7f7673323f28d11a12c6f3a215c61` |
| `transformers/...` | huggingface/transformers v4.29.2, the version Poseidon pins |
| `DPOT/...` | the DPOT repo, [thu-ml/DPOT](https://github.com/thu-ml/DPOT) at commit `dcd2f9a9359765e19ad63e2f3f879a2a8ce1aa17`, cloned into `third_party/DPOT/` |
| `GPhyT/...` | the GPhyT repo, [FloWsnr/General-Physics-Transformer](https://github.com/FloWsnr/General-Physics-Transformer) at commit `4374116e8c96db08f5b0691b04485959749d6374`, cloned into `third_party/GPhyT/`. In section 4, `config.yaml` alone means `GPhyT/gphyt/run/scripts/config.yaml`. |
| `MPP/...` | the MPP repo, [PolymathicAI/multiple_physics_pretraining](https://github.com/PolymathicAI/multiple_physics_pretraining) at commit `e751fc25ae5274c3ddc869bf73afd6fcd4163cb5`, cloned into `third_party/MPP/`. In section 5, `config.yaml` alone means `MPP/config/mpp_avit_ti_config.yaml`. |
| `hf:README.md` | the model card of the section's model: [huggingface.co/hzk17/DPOT](https://huggingface.co/hzk17/DPOT) in section 3, [huggingface.co/flwi/Physics-Foundation-Model](https://huggingface.co/flwi/Physics-Foundation-Model) in section 4 |
| "our choice" | no external source |

**About the "Poseidon repo" comparisons:** the Poseidon repo has no PhysBiasBench recipe. Its numbers below come from two places:
- the example config `configs/run.yaml` (Poseidon-B on its `wave.Layer` task);
- the defaults in `scOT/train.py` and `scOT/trainer.py`.

They show how the official code fine-tunes. They are not a recipe from the PhysBiasBench paper.

---

## 1. General choices (all models)

### 1.1 One independent run per (PDE, mix, variant)

- **Chosen:** each model gets 8 PDEs × 3 mixes × 2 variants = 48 separate runs.
  - Every run starts from the same weights: the pretrained checkpoint, or a seed-0 random init for scratch.
  - Every run trains on exactly one (PDE, mix) training set.
- **Why:** results for one PDE or one mix cannot leak into another, and each run can be checked or redone on its own.
- **Source:** our choice. `Poseidon-T/finetune.py:464-465`, `DPOT-Ti/finetune.py:548-549`, `GPhyT-S/finetune.py:574-575`, `MPP-Ti/finetune.py:610-611`.

### 1.2 Training recipe

- **Chosen:** the same recipe for every run of every model (table below).
- **Why:** if every model and both variants are trained the same way, differences in the results come from the model and the data, not from per-model tuning. The values are common fine-tuning settings; we did not tune them on PhysBiasBench.
- **Source:** our choice. `Poseidon-T/finetune.py:84-91`, `DPOT-Ti/finetune.py:103-110`, `GPhyT-S/finetune.py:105-112`, `MPP-Ti/finetune.py:111-118`.

The "Our code" column cites `Poseidon-T/finetune.py`. The other scripts use the same code:
- `DPOT-Ti/finetune.py`: AdamW at line 457, the training loss at 500, the best-epoch check at 518-520;
- `GPhyT-S/finetune.py`: AdamW at line 485, the training loss at 526, the best-epoch check at 544-546;
- `MPP-Ti/finetune.py`: AdamW at line 531, the training loss at 481, the best-epoch check at 580-582.

DPOT's own recipe is compared in 3.8, GPhyT's in 4.10, MPP's in 5.9.

**One exception, for MPP-Ti only:** its flat train windows are left out of training (5.11), and a batch whose gradient is non-finite is skipped (5.12). Validation, early stopping and the choice of the best epoch still use every window, as for the other models.

| Item | Chosen | Our code | Poseidon repo, where it differs |
|---|---|---|---|
| Optimizer | AdamW on all parameters | `finetune.py:377` | Also AdamW (`scOT/train.py:286`), but norm and bias parameters get no weight decay (`scOT/trainer.py:281-285`) |
| Learning rate | 1e-4, one value for every parameter (see 2.5) | `finetune.py:85, 377` | 5e-5, and 10× on the new layers (`configs/run.yaml:7-12`) |
| Weight decay | 1e-4, on every parameter | `finetune.py:86, 377` | 1e-6 (`configs/run.yaml:13-14`), not applied to norms and biases |
| Schedule | Linear warmup over the first 5% of steps, then cosine decay to 0 at the last step of epoch 50. Updated every step. Runs that stop early never reach the end of the cosine. | `finetune.py:87, 189-198, 378-379` | Cosine with no warmup (`configs/run.yaml:15-18`) |
| Loss | MSE in normalised space, averaged over all values. The val loss is computed the same way. | `finetune.py:416, 324-330` | L1 (`p=1`, `scOT/train.py:267`), made relative per channel group (`scOT/train.py:268`, `scOT/model.py:1426-1484`) |
| Batch size | 16. Reshuffled every epoch with a seeded generator. The last partial batch is kept: 3000 = 187 × 16 + 8. | `finetune.py:88, 380, 412-414` | 40 (`configs/run.yaml:23-24`) |
| Epochs | At most 50 | `finetune.py:89` | 200 (`configs/run.yaml:21-22`) |
| Early stopping | Stop after 10 epochs without a strict improvement in val loss | `finetune.py:90, 406, 434-438` | Patience 200 (`configs/run.yaml:19-20`), threshold 0 (`scOT/train.py:325-328`) |
| Checkpoint kept | The epoch with the lowest val loss, saved as `best.pt` | `finetune.py:434-436` | Same idea: best model on eval loss (`scOT/train.py:313-315`) |
| Seed | 0, reset at the start of every run | `finetune.py:91, 372, 380` | Same seed, 0 (`scOT/train.py:29-32, 310`) |
| Gradient clipping | None | none in `finetune.py` | Max grad norm 5.0 (`configs/run.yaml:25-26`, `scOT/train.py:284`) |

### 1.3 fp32 precision

- **Chosen:** everything runs in float32.
  - No autocast and no mixed precision.
  - TF32 is switched off for matmuls and for cuDNN.
- **Why:** numerical precision should not differ between models. TF32 would quietly lower matmul precision on Ampere and newer GPUs. It does not exist on V100, so switching it off changes nothing there.
- **Source:** our choice. `Poseidon-T/finetune.py:458-460`, `DPOT-Ti/finetune.py:542-544`, `GPhyT-S/finetune.py:568-570`, `MPP-Ti/finetune.py:604-606`; none of the scripts uses autocast.
- **Poseidon repo:** also no fp16 (`scOT/train.py:311`), but TF32 is left at PyTorch's defaults.
- **DPOT repo:** also plain fp32 (`DPOT/finetune.py` uses no autocast), with TF32 at PyTorch's defaults.
- **GPhyT repo:** pretraining used bf16 autocast and TF32 matmuls (`GPhyT/gphyt/run/scripts/config.yaml:45-46`; `GPhyT/gphyt/run/train.py:162, 486-488`).
- **MPP repo:** also fp32. Mixed precision is off in its config (`MPP/config/mpp_avit_ti_config.yaml:10`, used at `MPP/train_basic.py:238`), and TF32 is at PyTorch's defaults.

### 1.4 Normalisation

- **Chosen:** per-channel mean and std.
  - They are computed only on the **train** windows of the (PDE, mix) being trained.
  - Every frame of every window counts: the 5 inputs and the target. Frames shared by overlapping windows are counted once per window.
  - Inputs and targets use the same constants.
  - The model works entirely in normalised space. The constants are saved in `best.pt` (`norm_mean`, `norm_std`) so predictions can be converted back.
- **Why:**
  - The dataset API returns raw fields (`data/benchmark_api.py:184-187`), and the PDEs have very different scales.
  - Using only train statistics keeps validation data out of training.
  - scOT, DPOT-Ti (`normalize=False`, see 3.5) and GPhyT (see 4.8) don't normalise inside the model, so the data is normalised exactly once.
  - MPP does normalise inside the model, per sample. That normalisation cancels ours exactly, so ours is neutral there (see 5.5).
- **Source:** our choice.
  - `Poseidon-T/finetune.py:226-266` (statistics at 235-240 and 248-253), saved at `Poseidon-T/finetune.py:345-346`.
  - `DPOT-Ti/finetune.py:302-342` (statistics at 311-313 and 324-329), saved at `DPOT-Ti/finetune.py:425-426`.
  - `GPhyT-S/finetune.py:309-351` (statistics at 318-320 and 331-337), saved at `GPhyT-S/finetune.py:453-454`.
  - `MPP-Ti/finetune.py:294-338` (statistics at 303-305 and 316-324), saved at `MPP-Ti/finetune.py:457-458`.
- **Poseidon repo:** the constants are fixed per dataset and hard-coded in each dataset class, which applies them (`scOT/problems/reaction_diffusion/allen_cahn.py:20-24, 46-47`; `scOT/problems/fluids/normalization_constants.py:3-9`).
- **DPOT repo:** no normalisation at all. It fine-tunes on raw data (`DPOT/finetune.py:125-126`), with `normalize=False` in the model (`DPOT/configs/pretrain_tiny.yaml:68`).
- **GPhyT repo:** the same kind as ours: its data loader z-scores each field with dataset-wide constants (`GPhyT/gphyt/run/scripts/config.yaml:83`; `GPhyT/gphyt/data/well_dataset.py:224-239, 488-492`).
- **MPP repo:** its data loader doesn't normalise. The model z-scores each sample per channel and de-normalises its output (`MPP/models/avit.py:104-108, 129-130`).

### 1.5 Scratch at the small size

- **Chosen:** the scratch variant has exactly the architecture of the small pretrained checkpoint we fine-tune, with random weights and the same recipe. It is Poseidon-T sized for Poseidon, DPOT-Ti sized for DPOT, GPhyT-S sized for GPhyT and MPP-Ti sized for MPP.
- **Why:** we only fine-tune the smallest checkpoint (see 1.6). This size keeps pretrained vs scratch a matched-size comparison, as in the paper, only at a smaller size.
- **Source:** our choice. `Poseidon-T/finetune.py:63-64, 293`, `DPOT-Ti/finetune.py:83-84, 347-348`, `GPhyT-S/finetune.py:85-86, 355-356`, `MPP-Ti/finetune.py:91-92, 368`.
- **Paper:** scratch is M-sized and is compared with pretrained-M (Section 4.5; Table 2 caption).

### 1.6 Only the smallest public checkpoint

- **Chosen:** one pretrained size per architecture, the smallest public checkpoint: Poseidon-T for Poseidon, DPOT-Ti for DPOT, GPhyT-S for GPhyT, MPP-AViT-Ti for MPP.
- **Why:** the paper uses "three pretrained sizes from small to large" (S/M/L) but never names the checkpoints behind them. Its exact S/M/L can't be reproduced, so we use one small model per architecture.
- **Source:** our choice.
  - `Poseidon-T/finetune.py:71`, `models/download_poseidon_t.py`.
  - `DPOT-Ti/finetune.py:91`, `models/download_dpot_ti.py`.
  - `GPhyT-S/finetune.py:93`, `models/download_gphyt_s.py`.
  - `MPP-Ti/finetune.py:99`, downloaded by hand (`models/MPP_Ti.md`).
- **Poseidon repo:** publishes three sizes, T, B and L (`README.md:33`).
- **DPOT repo:** publishes five sizes, Ti, S, M, L and H (`hf:README.md:18-26`).
- **GPhyT repo:** publishes four sizes, S, M, L and XL (`hf:README.md:7`; `GPhyT/gphyt/model/model_specs.py:4-37`).
- **MPP repo:** publishes four sizes, Ti, S, B and L (`MPP/README.md:58`; one config per size in `MPP/config/`).

### 1.7 Fourier-only resampling

- **Chosen:** whenever a model needs a resolution other than 64×64, the change is made in Fourier space.
  - Going up: zero-pad the spectrum.
  - Going down: truncate the spectrum.
  - Amplitudes are preserved both ways.
  - Never bilinear interpolation or `F.interpolate`.
  - The loss is always computed at the dataset's 64×64.
- **Why:**
  - Every PhysBiasBench field is periodic (paper, Appendix A). Fourier interpolation therefore keeps every resolved mode exactly and adds no smoothing.
  - Bilinear interpolation would damp the high frequencies and bias any spectral analysis of the predictions.
- **Source:** our choice.
  - For Poseidon, scOT does this itself (see 2.3): `Poseidon-T/finetune.py:103-111`.
  - For DPOT, our own code does it (see 3.3): `DPOT-Ti/finetune.py:153-160, 167-200`.
  - For GPhyT, the same routine resamples anisotropically to 256 × 128 (see 4.4): `GPhyT-S/finetune.py:159-167, 174-207`.
  - For MPP, no resampling is needed: it runs natively at 64 × 64 (see 5.4).
- **Paper:** does not say how models with a different native resolution were fed 64×64 data.

---

## 2. Poseidon-T

Code: `Poseidon-T/finetune.py`. Weights: `models/Poseidon-T/`. The checkpoint's `config.json` has:
- `image_size` 128;
- 4 input and 4 output channels;
- `use_conditioning` true.

### 2.1 Input frame

- **Chosen:** the model gets only the last of the 5 input frames (`x[4]`).
  - Frames 0-3 are not fed to the model.
  - They still count in the normalisation statistics (1.4).
- **Why:**
  - scOT maps one state `(B, C, H, W)` plus a time to one output state, and has no slot for a history of frames (`scOT/model.py:1318-1321`; channel check at `scOT/model.py:299-302`).
  - The last frame is the one just before the target, so the lead time is exactly one frame step.
- **Source:** `Poseidon-T/finetune.py:93-95, 238`.
- **Poseidon repo:** same idea. Each sample is one input snapshot at t1 and one target at t2 (`scOT/problems/reaction_diffusion/allen_cahn.py:35-44`).

### 2.2 Lead time 1/19

- **Chosen:** a constant `time = 1/19 ≈ 0.0526` for every window, in both train and val.
- **Convention:** scOT's time is "frames elapsed / index of the trajectory's last frame", so a whole trajectory spans [0, 1].
  - 21-snapshot fluid datasets use `t/20` (`scOT/problems/fluids/incompressible.py:87`, `scOT/problems/fluids/normalization_constants.py:6`).
  - The 20-snapshot Allen-Cahn dataset uses `t/19` (`scOT/problems/reaction_diffusion/allen_cahn.py:23, 33`).
  - PhysBiasBench sequences have 20 frames (`data/benchmark_api.py:87-94`), so one step is 1/19.
- **Why:**
  - Poseidon-T has `use_conditioning=true`, so `Linear(1, dim)(time)` scales and shifts every LayerNorm (`scOT/model.py:143-160`). Time is therefore a required input, and its scale should match pretraining.
  - 1/19 is inside the range pretraining saw: 0 to 0.7 with the default settings (`scOT/problems/base.py:114, 344-353`).
  - The value is constant, so the time-conditioning layers can adapt to it during fine-tuning.
- **Note:** each mix contains sequences subsampled with raw strides 2, 3 or 4 (`data/benchmark_api.py:41-53`), so one frame step is not the same physical time in every window. As the protocol asks, every window still gets the same constant lead time; the model is not told the stride.
- **Source:** `Poseidon-T/finetune.py:96-101, 317-320`.
- **Poseidon repo:** fine-tuning uses every pair t1 ≤ t2 of snapshots (`scOT/problems/base.py:340-354`). The lead time therefore varies from sample to sample and includes 0. We use one fixed lead time because the dataset API gives only one-step windows.

### 2.3 64↔128 resampling done inside scOT

- **Chosen:**
  - Feed 64×64 frames directly, and keep the checkpoint's `image_size=128`.
  - scOT's `forward` Fourier-upsamples the input to 128 (`scOT/model.py:1360-1366`, zero-padding in `scOT/model.py:1302-1316`), runs the network at 128, and truncates the prediction back to 64 (`scOT/model.py:1416-1420`, truncation in `scOT/model.py:1293-1300`).
  - The loss is computed on that 64×64 output. Our code does no resampling of its own.
- **Why:**
  - The weights load unchanged, and the network sees a 32×32 patch grid, as in pretraining.
  - The internal resampling already follows rule 1.7. It uses `fft2` with `norm="forward"`, which keeps amplitudes, and takes `.real`. That is the same as `rfft2` for real fields.
  - Resampling in our code as well would resample twice.
- **Source:** `Poseidon-T/finetune.py:103-111`; `image_size` comes from `models/Poseidon-T/config.json`.
- **Poseidon repo:**
  - `scOT/train.py:249` sets `image_size` to the dataset's resolution.
  - Every Poseidon dataset is 128×128 (e.g. `scOT/problems/reaction_diffusion/allen_cahn.py:14`), so the repo never fine-tunes at another resolution.
  - Following `train.py` literally on 64×64 data would give `image_size=64`. The weight shapes would be the same, but the patch grid would be 16×16 instead of 32×32, and the attention windows would shrink at every stage (`scOT/model.py:412-440`), unlike pretraining.

### 2.4 The 4 replaced layers and their init

- **Chosen:**
  - The config is the checkpoint's own, with only `num_channels = num_out_channels = C` changed (`Poseidon-T/finetune.py:278-283`).
  - The weights load through `ScOT.from_pretrained(..., ignore_mismatched_sizes=True)` (`Poseidon-T/finetune.py:288-290`).
  - Only these 4 tensors depend on C (`scOT/model.py:282-284, 616-630`), so only they are not loaded:

    | Tensor | Checkpoint shape | New shape |
    |---|---|---|
    | `embeddings.patch_embeddings.projection.weight` | (48, 4, 4, 4) | (48, C, 4, 4) |
    | `patch_recovery.projection.weight` | (48, 4, 4, 4) | (48, C, 4, 4) |
    | `patch_recovery.projection.bias` | (4,) | (C,) |
    | `patch_recovery.mixup.weight` | (4, 4, 5, 5) | (C, C, 5, 5) |

  - `embeddings.patch_embeddings.projection.bias` has shape (48,) for any C, so it stays loaded.
- **Init of the 4 tensors:** PyTorch's default layer init (Kaiming-uniform). Their keys exist in the checkpoint, so transformers marks their modules as already initialised and skips its own `_init_weights` (`transformers/modeling_utils.py:122-135, 469-477, 2984-2986`).
- **Check:** `Poseidon-T/finetune.py:299-314` asserts three things:
  - there are no missing or unexpected keys;
  - exactly these 4 tensors are replaced;
  - every other tensor is identical to `model.safetensors`.
- **Scratch variant:** `ScOT(cfg)` with the same config (`Poseidon-T/finetune.py:293`), so every weight comes from transformers' Swinv2 init: normal(0, 0.02) for Linear and Conv2d layers (`transformers/models/swinv2/modeling_swinv2.py:969-981`), PyTorch's default for the rest. This matches `scOT/train.py:334-335`.
- **Why:** this is the official way to adapt Poseidon to new channels. Every tensor that doesn't depend on C is reused unchanged.
- **Source:** `Poseidon-T/finetune.py:113-122, 270-314`.
- **Poseidon repo:** same loading mechanism, via `--replace_embedding_recovery` (`scOT/train.py:330-333`; `README.md:35-43`). It builds a fresh `ScOTConfig` from its size table (`scOT/train.py:36-44, 247-275`) instead of copying the checkpoint's config, but the architecture is the same.

### 2.5 One learning rate instead of 10× on the new layers

- **Chosen:** a single lr of 1e-4 for every parameter, including the 4 new tensors and the time-conditioning layers.
- **Why:** the same recipe for every model and both variants (1.2). Per-group learning rates would be Poseidon-specific tuning, and the scratch variant has no "new" layers.
- **Source:** our choice. `Poseidon-T/finetune.py:85, 377`.
- **Poseidon repo:** when fine-tuning (`--finetune_from`), the trainer splits the parameters into two extra groups (`scOT/train.py:288-297`; `scOT/trainer.py:287-293, 300-365`):
  - parameters named `embeddings*` or `patch_recovery*` get their own learning rate;
  - ConditionalLayerNorm parameters (the time conditioning) get another.

  The example config uses 5e-5 for the rest of the model and 5e-4 for both groups, i.e. 10× (`configs/run.yaml:7-12`).

### 2.6 Pinned versions

- **Chosen** (the root `requirements.txt`, shared by the 4 models):

  | Requirement | Line | Reason |
  |---|---|---|
  | Poseidon at commit `b8fa28f`, the latest on `main` (2025-04-10) | 11 | reproducible code |
  | `torch==2.0.1`, `transformers==4.29.2` | 15, 19 | same as Poseidon's own pins |
  | `safetensors` | 21 | transformers 4.29.2 reads `model.safetensors` only when this is installed, and doesn't install it |
  | `numpy<2` | 27 | torch 2.0.1 was built against NumPy 1.x |
  | `huggingface_hub<1.0` | 29 | required by transformers 4.29.2; used by `models/download_poseidon_t.py` |
  | Python 3.9 | 2 | the shared environment; torch 2.0.1 has wheels for Python 3.8 to 3.11 only |

- **Why:** scOT imports internal Swinv2 classes from transformers (`scOT/model.py:35-47`), so a newer transformers could break it. Pinning the commit makes the code reproducible.
- **Poseidon repo:**
  - `pyproject.toml:5-16` pins `torch==2.0.1`, `torchvision==0.15.2`, `transformers==4.29.2`, `accelerate==0.31.0` and `wandb==0.14.2`.
  - numpy is unpinned, and safetensors is not listed.
  - It is installed with `pip install -e .` from a clone (`README.md:13-17`).

### 2.7 Smaller implementation details

- **Cleared loss setting.** `channel_slice_list_normalized_loss` is set to `None` (`Poseidon-T/finetune.py:282`).
  - Only scOT's built-in loss uses it (`scOT/model.py:1432-1482`), and we don't call that loss.
  - The checkpoint's value `[0, 1, 3, 4]` describes 4 channels.
- **Apple GPUs.** On MPS, the script first tries scOT's FFT resampling once and falls back to the CPU if it fails (`Poseidon-T/finetune.py:37, 129-141`). This only matters for local tests on a Mac; on CUDA nothing changes.

---

## 3. DPOT-Ti

Code: `DPOT-Ti/finetune.py`. Weights: `models/DPOT-Ti/model_Ti.pth`. DPOT code: `third_party/DPOT/`, created by `third_party/download_dpot_code.py`.

The pretrained configuration (`hf:README.md:33`, copied at `DPOT-Ti/finetune.py:112-135`) has:
- `img_size` 128 and `patch_size` 8;
- 4 input and 4 output channels;
- 10 input frames and 1 output frame;
- `normalize` False;
- about 7.5M parameters.

**Overall approach:** both variants keep this architecture exactly. Instead of changing layers, our data is padded and resampled to fit it (3.1 to 3.3).

### 3.1 Input frames: all 5, padded to 10

- **Chosen:**
  - DPOT sees all 5 input frames.
  - The time axis is padded to the 10 frames DPOT-Ti was pretrained with, by repeating the first frame: `[x0, x0, x0, x0, x0, x0, x1, x2, x3, x4]`.
  - Scratch uses the same padding.
  - `best.pt` stores `frames_used` and `time_padding`.
- **Why:**
  - The time-aggregation layer has one 512×512 weight slice per input frame: `w` has shape (10, 512, 512) (`DPOT/models/dpot.py:221-224`). That is 2.62M of the 7.5M parameters (35%).
  - Frame positions also enter through `linspace(0, 1, T)`, in the default `exp_mlp` aggregation (`DPOT/models/dpot.py:230-232`) and in the time coordinate channel (`DPOT/models/dpot.py:356-357`).
  - Keeping T = 10 keeps every pretrained weight. Our real frames sit in the last 5 time slots, where pretraining had its most recent frames.
  - Building the model with `in_timesteps=5` would mean re-initialising the whole time-aggregation layer.
- **Source:** `DPOT-Ti/finetune.py:140-146, 383-384`.
- **DPOT repo:**
  - Always uses 10 input frames (`DPOT/finetune.py:92`, `DPOT/train_temporal_parallel.py:86`).
  - Loads the time-aggregation layer strictly (`DPOT/utils/utilities.py:157-160`).
  - Has no mechanism for fewer frames.

### 3.2 Channels: padded with 1.0 to 4, loss on the real channels only

- **Chosen:**
  - The model keeps 4 input and 4 output channels.
  - Our C real normalised channels (1 or 2) come first, followed by 4 − C channels filled with 1.0.
  - The loss uses only the C real output channels; the padded output channels are ignored.
  - Scratch uses the same 4-channel model.
  - `best.pt` stores `n_channels` (the number of real channels) and `channel_pad_value`.
- **Why:**
  - The layers that depend on the channel count are:
    - the whole `patch_embed`: its input is C + 3, the 3 being coordinate channels, and its hidden width is `out_channels × patch_size + 3` = 35 (`DPOT/models/dpot.py:198-202, 278`);
    - the last conv of `out_layer` (`DPOT/models/dpot.py:320`).

    Replacing them would throw away the whole input embedding.
  - Padding with ones is how DPOT itself handles datasets with fewer than 4 channels, in pretraining too. The pretrained weights have therefore already seen such padding.
  - A loss on the padded channels would add a trivial target that has nothing to do with the PDE.
- **Source:** `DPOT-Ti/finetune.py:148-151, 386-387, 396-398, 419, 423`.
- **DPOT repo:**
  - Pads the data with ones up to `n_channels=4` (`DPOT/utils/griddataset.py:98-99`; `DPOT/finetune.py:101`; `DPOT/configs/dpot_finetune.yaml:24`). The padding is applied to raw, unnormalised data.
  - Its training loss **includes** the padded channels, because the mask is all ones (`DPOT/utils/griddataset.py:157`).
  - Its evaluation masks them out (`DPOT/utils/griddataset.py:109-117, 162`).

### 3.3 Resolution: our own Fourier resampling 64↔128

- **Chosen:**
  - Keep `img_size=128` and the pretrained `pos_embed`.
  - Our code Fourier-upsamples the inputs from 64 to 128 and Fourier-truncates the prediction from 128 to 64.
  - The loss is computed at 64×64 (rule 1.7).
  - The resampling uses an FFT with `norm="forward"`, which keeps amplitudes. The Nyquist mode of the 64 grid is split equally between +32 and −32 when upsampling and summed back when downsampling, so downsampling undoes upsampling exactly.
  - DPOT's coordinate channels are computed at 128, as in pretraining.
- **Why:**
  - DPOT cannot run at 64×64 with its pretrained weights:
    - `PatchEmbed` asserts that the input size equals `img_size` (`DPOT/models/dpot.py:206`);
    - `pos_embed` has shape (1, 512, 16, 16) for 128/8 (`DPOT/models/dpot.py:280`).
  - `img_size=64` would need a resized `pos_embed`. DPOT's resize helper is written for ViT-style tokens (1, N, D), uses bilinear interpolation (`DPOT/models/dpot.py:424-459`), and is not used by its fine-tuning.
  - The AFNO weights do not depend on resolution (`DPOT/models/dpot.py:45-48, 70-94`); only `pos_embed` blocks 64×64.
  - At 128 the model sees the 16×16 patch grid it was pretrained on.
- **Source:** `DPOT-Ti/finetune.py:153-160, 167-200, 382, 399`.
- **DPOT repo:** resizes all data, inputs and targets, to `res=128` with **bilinear** `F.interpolate`, and computes the loss at 128 (`DPOT/utils/griddataset.py:96`; `DPOT/configs/dpot_finetune.yaml:7`).

### 3.4 No input noise

- **Chosen:** no noise is added to the inputs.
- **Why:** our common recipe has none, and DPOT's own fine-tuning example has none either.
- **Source:** `DPOT-Ti/finetune.py:491-494`.
- **DPOT repo:**
  - **Pretraining** adds `x += noise_scale · ‖x‖₂ · randn`, where the norm is taken over (X, Y, T) per sample and channel (`DPOT/train_temporal_parallel.py:220`). DPOT-Ti used `noise_scale = 0.0005` (`DPOT/configs/pretrain_tiny.yaml:71`).
  - **Fine-tuning** uses no noise:
    - `DPOT/finetune.py:57` defaults to 0.0;
    - `DPOT/configs/dpot_finetune.yaml` has 0.0005 at the top level (line 16), but its task list sets `[0.0]` (line 77);
    - task arguments are appended after the base ones (`DPOT/trainer.py:41-58`), so 0.0 wins.

### 3.5 `normalize` flag off

- **Chosen:** `normalize=False`, as in the pretrained model.
- **Why:**
  - `normalize=True` would add per-sample normalisation inside the model, plus two extra layers (`scale_feats_mu`, `scale_feats_sigma`) that `model_Ti.pth` does not contain (`DPOT/models/dpot.py:298-300, 366-370, 386-387, 400-401`).
  - With `False`, our per-channel normalisation (1.4) is the only one.
- **Source:** `DPOT-Ti/finetune.py:125`.
- **DPOT repo:** same flag (`hf:README.md:33`; `DPOT/configs/pretrain_tiny.yaml:68`), but DPOT trains on raw data (see 1.4).

### 3.6 Classification output ignored

- **Chosen:**
  - `forward` returns `(prediction, cls_pred)` (`DPOT/models/dpot.py:403`), and only the prediction is used.
  - `cls_head` stays in the model so the checkpoint loads in full.
  - `cls_head` gets no gradient, so AdamW never updates it and applies no weight decay to it.
- **Why:** `cls_pred` guesses which pretraining dataset a sample came from, and it never had a loss weight: 0.0 in pretraining (`DPOT/train_temporal_parallel.py:243`), and commented out in fine-tuning (`DPOT/finetune.py:239`).
- **Source:** `DPOT-Ti/finetune.py:391-399`.
- **DPOT repo:** computes the classification loss and accuracy for logging only (`DPOT/finetune.py:223-226, 239`).

### 3.7 Loading `model_Ti.pth`, and the scratch init

- **Chosen:**
  - **Contents:** `model_Ti.pth` holds `{'args', 'model', 'optimizer'}`, as saved by `DPOT/train_temporal_parallel.py:307`. The file size matches: 7.53M parameters × 4 bytes × 3 (the weights plus Adam's `exp_avg` and `exp_avg_sq`, `DPOT/utils/optimizer.py:134-136`) ≈ 90.4 MB.
  - **Loading:**
    - Only `'model'` is used.
    - `weights_only=False` is needed because `'args'` is a pickled argparse Namespace.
    - A DDP `module.` prefix is stripped if present, as `DPOT/utils/utilities.py:119-125` does.
    - Every tensor is then loaded.
  - **Check:** the script prints the file's keys and the missing and unexpected keys. It then asserts there are none, and that every tensor is identical to the checkpoint's.
  - **Scratch:** `DPOTNet(**DPOT_CONFIG)` with DPOTNet's own random init:
    - `pos_embed`: truncated normal, std 0.02 (`DPOT/models/dpot.py:325`);
    - AFNO weights: scaled uniform (`DPOT/models/dpot.py:45-48`);
    - time-aggregation weights: scaled normal (`DPOT/models/dpot.py:221-224`);
    - everything else: PyTorch defaults.

    `DPOTNet._init_weights` (`DPOT/models/dpot.py:329-337`) is defined but never applied, in DPOT's scripts or in ours.
- **Why:** it is the same mechanism as the model card, and no tensor is replaced.
- **Source:** `DPOT-Ti/finetune.py:347-377`.
- **DPOT repo:** the model card loads it this way (`hf:README.md:33-34`). Fine-tuning loads everything with `load_components: ['all']` (`DPOT/configs/dpot_finetune.yaml:42` → `DPOT/utils/utilities.py:126-128`).

### 3.8 Training recipe compared with DPOT's own fine-tuning

We use the common recipe of 1.2. DPOT's numbers below come from its example fine-tuning config `DPOT/configs/dpot_finetune.yaml`, which targets DPOT-S on its `ns2d_cond_pda` task (lines 41-46, 89; the Ti block at lines 30-35 is commented out), and from the defaults in `DPOT/finetune.py`. They are not a recipe from the PhysBiasBench paper.

| Item | Ours (1.2) | DPOT repo |
|---|---|---|
| Optimizer | AdamW on all parameters, weight decay 1e-4 | DPOT's own Adam, betas (0.9, 0.9), weight decay 1e-6 (`DPOT/finetune.py:149`; `dpot_finetune.yaml:22-23`) |
| Learning rate | 1e-4 | 1e-3 (`dpot_finetune.yaml:10`) |
| Schedule | 5% linear warmup, then cosine | One-cycle: 40 warmup epochs out of 200 (`DPOT/finetune.py:152-154`; `dpot_finetune.yaml:11, 74-75`) |
| Loss | MSE in normalised space, real channels only | Relative L2 per channel, summed over the batch (`DPOT/finetune.py:189, 220`; `DPOT/utils/criterion.py:52-59`), padded channels included |
| Batch size | 16 | 32 (`dpot_finetune.yaml:76`) |
| Epochs / stopping | At most 50, early stopping with patience 10, best val epoch kept | 200 epochs (`dpot_finetune.yaml:74`), no early stopping; saves the latest model every epoch (`DPOT/finetune.py:299-300`) |
| Gradient clipping | None | Clip at 10000 (`DPOT/finetune.py:241`; `dpot_finetune.yaml:21`), so effectively none |
| Seed | 0 | Seeding commented out (`DPOT/finetune.py:33-34`) |

### 3.9 Code location and pinned versions

- **Chosen:**

  | What | Where | Reason |
  |---|---|---|
  | DPOT code at commit `dcd2f9a` (latest on `main`, 2024-06-10), cloned into `third_party/DPOT/` by `third_party/download_dpot_code.py` | `DPOT-Ti/finetune.py:50-68` | DPOT has no `setup.py`, so pip can't install it. GitHub reports no licence for the code, so we don't copy it into this repo; the weights on Hugging Face are Apache-2.0 (`hf:README.md:2`). |
  | The script checks that `third_party/DPOT/.git/HEAD` is that commit, and loads `models/dpot.py` by file path | `DPOT-Ti/finetune.py:54-65` | DPOT's package name `models` would clash with this repo's `models/` folder |
  | `torch==2.0.1` | `requirements.txt:15` | DPOT uses pytorch 2.0.0 (`DPOT/requirements.txt:4`, `DPOT/README.md:95`). 2.0.1 is the bug-fix release of the same version and matches Poseidon-T, so one environment runs both. |
  | `einops==0.7.0` | `requirements.txt:25` | imported by `DPOT/models/dpot.py`; same version as `DPOT/requirements.txt:9` |
  | `numpy<2` | `requirements.txt:27` | torch 2.0.1 was built against NumPy 1.x |
  | `huggingface_hub<1.0` | `requirements.txt:29` | used by `models/download_dpot_ti.py`; keeps the environment compatible with Poseidon-T |
  | Python 3.9 | `requirements.txt:2` | DPOT's environment is Python 3.9; torch 2.0.1 has no wheels for 3.12+ |

- **DPOT repo:** `DPOT/requirements.txt` is a conda export (lines 4-16): pytorch 2.0.0 for Python 3.9 and CUDA 11.7, plus accelerate, timm, tensorboard, h5py, scipy, torch-optimizer and others. We only need what `DPOT/models/dpot.py` imports.

### 3.10 Smaller implementation details

- **Apple GPUs.** On MPS, the script first tries our FFT resampling and DPOT's AFNO FFT ops once, and falls back to the CPU if they fail (`DPOT-Ti/finetune.py:37, 203-217`). On CUDA nothing changes.

---

## 4. GPhyT-S

Code: `GPhyT-S/finetune.py`. Weights: `models/GPhyT-S/gphyt-S.pth`. GPhyT code: `third_party/GPhyT/`, created by `third_party/download_gphyt_code.py`.

The pretrained configuration (`GPhyT-S/finetune.py:114-141`) has:
- 5 fields: pressure, density, temperature, vel_x, vel_y;
- 4 input frames at 256 × 128 (H × W), cut into 16 × 16 patches;
- a learned absolute positional encoding, finite-difference derivative channels and an Euler integrator;
- dropout and stochastic depth at 0;
- about 8.0M parameters.

**Overall approach:** as for DPOT-Ti, both variants keep this architecture exactly, and our data is padded and resampled to fit it (4.2 to 4.4). The integrator, derivatives, dropout and stochastic depth stay as they are (4.5 to 4.7).

### 4.1 Where the configuration comes from

- **Chosen:** the arguments GPhyT's own `get_model` passes to `PhysicsTransformer` (`GPhyT/gphyt/model/transformer/model.py:17-53`, the same function as `hf:README.md:12-48`):
  - the sizes of `GPT_S` (`GPhyT/gphyt/model/model_specs.py:5-10`);
  - every other value from the training config (`config.yaml:21-34`);
  - `img_size` = (`n_steps_input`, `out_shape`) = (4, 256, 128) (`config.yaml:86, 88`; `GPhyT/gphyt/run/train.py:100-102`).
- **Why:** the model card only points to `model_specs` (`hf:README.md:8`) and gives no other values.
- **Caveat:** the shipped `config.yaml` is the XL run (`model_size: GPT_XL`, line 22), and nothing in the repo lists the S settings separately. The file size is the evidence:
  - this configuration gives 7,992,773 parameters, or 31.97 MB in fp32, against 32,039,435 bytes for `gphyt-S.pth`;
  - switching derivatives off, using a rotary encoding, or a 256 × 256 positional grid would not fit.

  The strict load (4.9) confirms it.
- **Source:** `GPhyT-S/finetune.py:114-141`.

### 4.2 Input frames: the last 4 of 5

- **Chosen:**
  - GPhyT sees the last 4 input frames, `x1` to `x4`.
  - `x0` is not fed to the model, though it still counts in the normalisation statistics (1.4).
  - Scratch does the same.
  - `best.pt` stores `frames_used`.
- **Why:**
  - GPhyT-S was pretrained with 4 input frames (`n_steps_input`, `config.yaml:86`).
  - Its learned positional encoding has one slice per frame, of shape (1, 4, 16, 8, 192), and is added as is (`GPhyT/gphyt/model/transformer/pos_encodings.py:40, 56`; `GPhyT/gphyt/model/transformer/model.py:146-148, 182-188`). So exactly 4 frames are required.
  - Taking the last 4 needs no padding and keeps every pretrained weight. They are also the frames closest to the target.
- **Source:** `GPhyT-S/finetune.py:145-149, 321`.
- **GPhyT repo:** has no mechanism for another number of frames.

### 4.3 Fields: padded with 0.0 to 5, loss on the real channels only

- **Chosen:**
  - The model keeps its 5 fields, in the order `[pressure, density, temperature, vel_x, vel_y]`.
  - For every PDE, our C real normalised channels go into the first C slots (pressure; or pressure and density), and the other 5 − C fields are filled with 0.0.
  - The same rule applies to Burgers, even though its two channels are velocity components (paper, Appendix A).
  - The loss uses only the C real output channels.
  - Scratch uses the same 5-field model.
  - `best.pt` stores `field_slots` (our channel i goes to field slot i), `gphyt_fields` and `field_pad_value`.
- **Why:**
  - The layers that depend on the number of fields are:
    - the tokenizer `Conv3d`, with 4 × 5 = 20 inputs: each field plus its dt, dh and dw (`GPhyT/gphyt/model/transformer/model.py:152-158, 229`; `GPhyT/gphyt/model/tokenizer/tokenizer.py:169-176`). It holds about 983k parameters, 12% of the model;
    - the detokenizer `ConvTranspose3d`, with 5 outputs (`GPhyT/gphyt/model/transformer/model.py:214-222`; `GPhyT/gphyt/model/tokenizer/tokenizer.py:234-246`);
    - the derivative filter buffers (`GPhyT/gphyt/model/transformer/derivatives.py:47-59`).

    Replacing them would throw away the input embedding.
  - 0.0 in normalised space is the mean of a z-scored field, the neutral value.
  - One rule for every PDE avoids per-PDE guesses about which of our channels matches which GPhyT field.
- **Source:** `GPhyT-S/finetune.py:151-157, 404-410, 424, 449-451`.
- **GPhyT repo:**
  - Its pretraining fields, in this order, are set in `config.yaml:21` and `GPhyT/gphyt/data/dataset_utils.py:119-122`.
  - No fill value for missing fields is visible: the loader reads all 5 named fields from every file (`GPhyT/gphyt/data/well_dataset.py:458-473`), so missing fields were filled in preprocessing that isn't published.
  - A helper `zero_field_to_value` exists but is never called (`GPhyT/gphyt/data/phys_dataset.py:15-27`).

### 4.4 Resolution: anisotropic Fourier resampling 64 × 64 ↔ 256 × 128

- **Chosen:**
  - `img_size` stays (4, 256, 128), with the pretrained positional encoding.
  - Our code Fourier-resamples the inputs from 64 × 64 to 256 × 128 (×4 along H, ×2 along W), and the prediction back to 64 × 64.
  - It uses the same routine as DPOT-Ti: `fourier_resize_axis` is identical, so downsampling undoes upsampling exactly.
  - The loss is computed at 64 × 64 (rule 1.7).
  - Pixels end up with a 2:1 aspect ratio.
- **Why:**
  - 16 × 16 patches of 256 × 128 give a fixed 16 × 8 grid for the positional encoding (`GPhyT/gphyt/model/transformer/model.py:146-148, 182-188`; `GPhyT/gphyt/model/transformer/pos_encodings.py:40`). A 64 × 64 input would give 4 × 4, which can't be loaded or added.
  - The repo has no resizing code.
  - The isotropic alternative (upsample to 128 × 128 and tile the periodic field twice along H) was rejected: the model would see duplicated content.
- **Source:** `GPhyT-S/finetune.py:159-167, 174-207, 406, 425`.
- **GPhyT repo:** pretraining data were all 256 × 128 (`out_shape`, `config.yaml:88`). That resizing happened in preprocessing, which is not in the repo.

### 4.5 Integrator and time step: kept

- **Chosen:** GPhyT's own `forward`, unchanged.
  - It applies one Euler step with step size 1.0 to the network output on every input frame, and keeps the last frame: the prediction is `x_last + f(x)_last`.
  - There is no dt input.
- **Why:**
  - That is how GPhyT-S was pretrained.
  - The model infers the time scale from the frames, and pretraining drew the frame stride at random between 1 and 8 (`config.yaml:85`).
- **Source:** `GPhyT-S/finetune.py:132, 413-425`.
- **GPhyT repo:** `GPhyT/gphyt/model/transformer/model.py:243-248`; `forward` never passes its own `step_size` to the integrator (line 247). Euler is `GPhyT/gphyt/model/transformer/num_integration.py:39-41`.

### 4.6 Derivative channels: kept

- **Chosen:** GPhyT adds dt, dh and dw channels itself, and they are kept unchanged:
  - fixed `[-1, 0, 1]` stencils, with no ½ factor and no grid spacing;
  - **replicate** padding;
  - computed on the 256 × 128 resampled grid.
- **Why:**
  - The pretrained tokenizer expects these exact channels. Circular padding would change the inputs compared with pretraining.
  - **Consequence:** our fields are periodic, so replicate padding gives one-sided differences on the outermost row and column of each frame, and for dt on the first and last frame. Pretraining used the same code on periodic datasets too (for example `euler_multi_quadrants_periodicBC`, `config.yaml:108`).
- **Source:** `GPhyT-S/finetune.py:129, 413-421`.
- **GPhyT repo:** `GPhyT/gphyt/model/transformer/model.py:152-158, 227-229`; `GPhyT/gphyt/model/transformer/derivatives.py:36-37, 66, 89-99`.

### 4.7 Dropout and stochastic depth: kept at 0

- **Chosen:** dropout 0.0 and stochastic depth 0.0, as pretrained (`config.yaml:24, 27`).
- **Why:** our recipe keeps the model's own architecture settings. The model has no train-time randomness.
- **Source:** `GPhyT-S/finetune.py:139-140`.

### 4.8 No normalisation inside the model

- **Chosen:** our per-channel normalisation (1.4) is the only one.
- **Why:** `PhysicsTransformer` has no normalisation step (`GPhyT/gphyt/model/transformer/model.py:224-248`). The `RevIN` and `RevLN` layers in `GPhyT/gphyt/model/transformer/norms.py` are not used by the model.
- **Source:** `GPhyT-S/finetune.py:331-335`.
- **GPhyT repo:**
  - Its data loader z-scores each field with dataset-wide constants (`config.yaml:83`; `GPhyT/gphyt/data/well_dataset.py:224-239, 488-492`).
  - It normalises each sample instead only when that option is off (`GPhyT/gphyt/data/phys_dataset.py:131-133, 168-174, 197-198`).

### 4.9 Loading `gphyt-S.pth`, and the scratch init

- **Chosen:**
  - **Contents:** `gphyt-S.pth` holds only the weights: 32 MB, with no optimizer state. GPhyT's training checkpoints keep the weights under `model_state_dict` (`GPhyT/gphyt/run/train.py:384-399`). The script accepts that layout or a bare state dict, and prints which one it found.
  - **Loading:**
    - `weights_only=False`, as GPhyT's own loader uses (`GPhyT/gphyt/run/run_utils.py:27-28`).
    - The `module.` (DDP) and `_orig_mod.` (`torch.compile`, `GPhyT/gphyt/run/train.py:171-173`) prefixes are stripped, as GPhyT's loader does (`GPhyT/gphyt/run/run_utils.py:29-40`).
    - Every tensor is then loaded.
  - **Check:** the script prints the missing and unexpected keys. It then asserts there are none, and that every tensor is identical to the file's.
  - **Scratch:** `PhysicsTransformer(**GPHYT_CONFIG)` with GPhyT's own init. GPhyT applies no custom init: the positional encoding is `0.02 · randn` (`GPhyT/gphyt/model/transformer/pos_encodings.py:40`), and every other layer keeps PyTorch's default.
- **Why:** the model card says to "initialize the model and then load the state dicts" (`hf:README.md:10`). No tensor is replaced.
- **Source:** `GPhyT-S/finetune.py:355-401`.

### 4.10 Training recipe compared with GPhyT's pretraining

We use the common recipe of 1.2. GPhyT publishes no fine-tuning recipe, so its numbers below come from its pretraining config `config.yaml` (the XL run) and `GPhyT/gphyt/run/train.py`. They are not a recipe from the PhysBiasBench paper.

| Item | Ours (1.2) | GPhyT repo |
|---|---|---|
| Optimizer | AdamW on all parameters, weight decay 1e-4 | AdamW, weight decay 0.01, betas (0.9, 0.999) (`config.yaml:65-69`) |
| Learning rate | 1e-4 | Same, 1e-4 (`config.yaml:67`) |
| Schedule | 5% linear warmup, then cosine to 0 | Linear warmup over 5000 updates from 0.001×, then cosine to 1% (`config.yaml:72-81`) |
| Loss | MSE in normalised space, real channels only | MSE in normalised space on all 5 fields (`config.yaml:70`; `GPhyT/gphyt/run/train.py:291-299, 489-490`) |
| Batch size | 16 | 64 per GPU (`config.yaml:49`) |
| Length / stopping | At most 50 epochs, early stopping, best val epoch kept | 1,000,000 batches (`config.yaml:50`), a checkpoint every 1000 batches (`config.yaml:52`) |
| Precision | fp32 (1.3) | bf16 autocast and TF32 (`config.yaml:45-46`; `GPhyT/gphyt/run/train.py:162, 486-488`) |
| Gradient clipping | None | 1.0 (`config.yaml:64`) |
| Data augmentation | None (`GPhyT-S/finetune.py:519-520`) | Random flips of x and y, each with probability 0.5 (`config.yaml:89-90`; `GPhyT/gphyt/data/phys_dataset.py:185-196`) |
| Seed | 0 | 42 (`config.yaml:48`) |

### 4.11 Code location and pinned versions

- **Chosen:**

  | What | Where | Reason |
  |---|---|---|
  | GPhyT code at commit `4374116` (2025-10-08, the commit that published the weights), cloned into `third_party/GPhyT/` by `third_party/download_gphyt_code.py` | `GPhyT-S/finetune.py:51-70` | Since then, `main` (`df1527c`, 2026-02-07) has renamed `gphyt.model` to `gphyt.models`, rewritten `forward`, and moved to Python ≥ 3.13 |
  | The script checks that `third_party/GPhyT/.git/HEAD` is that commit, and puts the clone on `sys.path` | `GPhyT-S/finetune.py:55-67` | GPhyT's `pyproject.toml` asks for Python ≥ 3.12 and lists no dependencies (`GPhyT/pyproject.toml:6-7`), so pip won't install it on Python 3.9–3.11. The licence is MIT. |
  | `torch==2.0.1` | `requirements.txt:15` | Shares the environment with Poseidon-T and DPOT-Ti. GPhyT's README uses Python 3.12 and a recent torch (`GPhyT/README.md:52-56`), but the model code only uses layers torch 2.0.1 has, and no Python 3.10+ syntax. |
  | `torchvision==0.15.2` | `requirements.txt:17` | `StochasticDepth` comes from torchvision (`GPhyT/gphyt/model/transformer/attention.py:11`); 0.15.2 is the release paired with torch 2.0.1 |
  | `einops==0.7.0` | `requirements.txt:25` | imported by the model code; same version as DPOT-Ti |
  | `numpy<2` | `requirements.txt:27` | torch 2.0.1 was built against NumPy 1.x |
  | `huggingface_hub<1.0` | `requirements.txt:29` | used by `models/download_gphyt_s.py`; keeps the environment compatible with Poseidon-T |
  | Python 3.9 | `requirements.txt:2` | the shared environment; torch 2.0.1 has no wheels for 3.12+ |

### 4.12 Smaller implementation details

- **Apple GPUs.** On MPS, the script first tries our FFT resampling and a 3D convolution once, and falls back to the CPU if they fail (`GPhyT-S/finetune.py:39-40, 210-224`). On CUDA nothing changes.

---

## 5. MPP-Ti

Code: `MPP-Ti/finetune.py`. Weights: `models/MPP-Ti/MPP_AViT_Ti`, downloaded by hand (`models/MPP_Ti.md`). MPP code: `third_party/MPP/`, created by `third_party/download_mpp_code.py`.

The pretrained configuration (`MPP-Ti/finetune.py:129-147`) has:
- an AViT with embedding size 192, 3 heads and 12 space-time blocks: time attention, then axial attention over H and W, with an MLP of 768;
- 16 × 16 patches made by convolution stacks, and relative position biases instead of a positional embedding;
- 12 state slots (`n_states`), of which pretraining used 10;
- stochastic depth up to 0.2;
- about 7.3M parameters.

**Overall approach:** unlike DPOT-Ti and GPhyT-S, MPP needs no padding and no resampling:
- it accepts any number of frames;
- it accepts any resolution that is a multiple of 16;
- it takes new fields through its own fine-tuning procedure.

Its training set leaves out flat windows (5.11), and a safety net skips any batch with non-finite gradients (5.12).

### 5.1 Configuration and checkpoint

- **Chosen:**
  - **Model:** MPP's own `build_avit`, given the values it reads from `config.yaml` (lines 12, 32, 40-48): `MPP/models/avit.py:14-27`; `MPP/models/mixed_modules.py:16-24`; `MPP/models/spatial_modules.py:18-20`; `MPP/models/time_modules.py:15-20`.
  - **Checkpoint:** a bare state dict, with no `model_state` key and no prefix. It holds 439 tensors, exactly the number this architecture has (checked by listing the file).
- **Why:** the README says to use the configuration that matches the weights file (`MPP/README.md:58`), and that all provided weights use `n_states = 12` (`MPP/README.md:133-135`).
- **Parameter count:** 7,285,884, or 29.1 MB in fp32, against a 29.3 MB file.
- **Source:** `MPP-Ti/finetune.py:129-143, 366-368, 384-400`.

### 5.2 Input frames: all 5

- **Chosen:**
  - MPP sees all 5 input frames, with no padding.
  - Scratch does the same.
  - `best.pt` stores `frames_used`.
- **Why:**
  - MPP was pretrained with 16 frames (`n_steps`, `config.yaml:35`), but its time attention has no fixed length. Its relative position bias is computed for any number of frames, and there is no absolute time embedding (`MPP/models/time_modules.py:45-61`; `MPP/models/shared_modules.py:99-128`).
  - So 5 frames work, and every weight loads unchanged.
- **Source:** `MPP-Ti/finetune.py:149-152, 306`.
- **MPP repo:** fine-tuning inherits the pretraining history length of 16 (`config.yaml:35, 75-76`). Shorter histories are allowed (`enforce_max_steps: False`, `config.yaml:36`).

### 5.3 Fields: MPP's own fine-tuning procedure

- **Chosen:**
  - For every PDE, our C channels get the state labels 10 to 10 + C − 1.
  - Right after the weights are loaded, MPP's `expand_projections` enlarges the projections by C, as MPP's trainer does.
  - Scratch uses the same expanded model with random weights.
  - `best.pt` stores `state_labels` and `n_channels`, which is also the size of the expansion.
- **How MPP handles fields:**
  - Inputs go through `SubsampledLinear`. It uses only the weight columns of the given labels, and scales by √(n_states / C) (`MPP/models/spatial_modules.py:43-70`).
  - Outputs use the matching channels of the output kernel (`MPP/models/spatial_modules.py:117-120`).
  - Pretraining numbered the fields in the order of `DSET_NAME_TO_OBJECT` (`MPP/data_utils/datasets.py:21-26, 108-114`; field names at `MPP/data_utils/hdf5_datasets.py:220, 242, 267, 331`):

    | Index | Field | Dataset |
    |---|---|---|
    | 0 | h | shallow water |
    | 1–3 | Vx, Vy, particles | incompressible NS |
    | 4–5 | activator, inhibitor | diffusion-reaction |
    | 6–9 | Vx, Vy, density, pressure | compressible NS |
    | 10–11 | — | never used |

- **The official procedure** for fine-tuning the released weights (`MPP/README.md:133-144`):
  1. Add the new dataset at the bottom of `DSET_NAME_TO_OBJECT`.
  2. Keep `n_states = 12`.
  3. List the dataset in `append_datasets`.
  4. Run with the `finetune` namespace.

  With `use_all_fields` (`config.yaml:52`), the new fields take the indices after the 10 pretraining fields, i.e. 10 and 11 (`MPP/data_utils/datasets.py:108-114`). The trainer then calls `expand_projections` with the number of new fields (`MPP/train_basic.py:202-207`; `MPP/models/avit.py:57-72`).
- **Consequences:**
  - Slots 10 and 11 exist in the checkpoint but were never used in pretraining. So the input and output projections of our fields start from untrained weights, while the rest of the network is pretrained.
  - The C slots the expansion adds (12 and up) stay unused. Their only effect is to change the input scale factor to √((12 + C) / C).
- **Why:** this is the procedure MPP documents for its released weights. Mapping our fields to related pretrained slots (for example Gray–Scott to the activator/inhibitor slots 4–5) would reuse trained weights, but needs a guess for each PDE.
- **Source:** `MPP-Ti/finetune.py:154-161, 177, 184-186, 370-373, 424, 451`.

### 5.4 Resolution: native 64 × 64

- **Chosen:**
  - No resampling: 64 × 64 in, 64 × 64 out, and the loss at 64 × 64.
  - That gives 4 × 4 tokens per frame.
- **Why:**
  - MPP has no positional embedding.
  - Its patch stem and output are convolution stacks with strides 4, 2 and 2 (`MPP/models/spatial_modules.py:72-120`), so any multiple of 16 works.
  - Its spatial attention uses a relative position bias grouped into distance buckets (`MPP/models/spatial_modules.py:160-178`; `MPP/models/shared_modules.py:37-128`).
  - So 64 × 64 runs with the pretrained weights unchanged, and rule 1.7 only resamples when a model needs it.
  - Pretraining already mixed resolutions: the config has 128 and 512 subsets (`config.yaml:59-61`), read as they are (`MPP/data_utils/hdf5_datasets.py:182-209`).
- **Consequence:** pretraining saw 8 × 8 and 32 × 32 tokens per frame; we use 4 × 4.
- **Source:** `MPP-Ti/finetune.py:169-174`.
- **MPP repo:** has no resizing code at all.

### 5.5 Normalisation: our pipeline kept, and why it is neutral

- **Chosen:** the same per-channel train-set normalisation as for every other model (1.4). The loss is computed in that space.
- **MPP's own normalisation:** inside the model, MPP z-scores the input frames per sample and per channel, over (T, H, W), with no gradient through the statistics. It then multiplies its output by the same std and adds the same mean back (`MPP/models/avit.py:104-108, 129-130`).
- **Why ours is neutral:**
  - Ours is a fixed affine map per channel: x′ = (x − m) / s, with s > 0.
  - For one sample, the mean and std of x′ are μ′ = (μ − m) / s and σ′ = σ / s. MPP's normalised input is then (x′ − μ′) / σ′ = (x − μ) / σ, exactly what it would compute from raw data.
  - So the network sees the same numbers either way, and produces the same output z.
  - MPP's de-normalised output is z·σ′ + μ′ = ((z·σ + μ) − m) / s. That is our normalisation applied to the prediction MPP would make from raw data.
  - The only difference is the 1e-7 MPP adds to σ (`MPP/models/avit.py:107`). It matters even less on our unit-scale data than on raw fields.

  So ours doesn't normalise twice in any way that matters. It only sets the space the prediction comes out in, which is the space our loss uses, as for every model.
- **Source:** `MPP-Ti/finetune.py:316-324`.
- **MPP repo:** its data loader doesn't normalise (`MPP/data_utils/hdf5_datasets.py:182-209`). The model's per-sample normalisation is the only one.

### 5.6 Boundary conditions: periodic

- **Chosen:** `bcs = [1, 1]` for every sample.
- **Why:**
  - Every PhysBiasBench domain is periodic (paper, Appendix A).
  - MPP gives `[1, 1]` to its periodic datasets (`MPP/data_utils/hdf5_datasets.py:361-362`) and `[0, 0]` to the others (e.g. `MPP/data_utils/hdf5_datasets.py:230-231`).
  - Only `bcs[0, 0]` (W axis) and `bcs[0, 1]` (H axis) are read (`MPP/models/spatial_modules.py:163, 172`). With 1, the spatial relative position bias wraps around the domain (`MPP/models/shared_modules.py:113-116`).
- **Source:** `MPP-Ti/finetune.py:163-167, 425`.

### 5.7 Drop path: kept at 0.2

- **Chosen:** stochastic depth rising from 0 to 0.2 across the 12 blocks, active only in training.
- **Why:**
  - That is what `build_avit` actually builds. It doesn't pass the config's `drop_path: 0.1` (`config.yaml:21`), so `AViT`'s default of 0.2 applies (`MPP/models/avit.py:22-26, 40-44, 53-54`), with timm's `DropPath` (`MPP/models/spatial_modules.py:143`; `MPP/models/time_modules.py:43`).
  - Our recipe keeps the model's own settings.
  - It uses the seeded torch random generator, whose state is restored on resume.
- **Source:** `MPP-Ti/finetune.py:144-147, 474`.
- **MPP repo:** the same, since its pretraining and fine-tuning also go through `build_avit`. There is no dropout.

### 5.8 Loading `MPP_AViT_Ti`, and the scratch init

- **Chosen:**
  - **Contents:** a bare state dict (an `OrderedDict`) with no `model_state` key and no prefix, holding 439 tensors. At 28 MB it can only be the weights.
  - **Loading:**
    - `weights_only=False`, as in the other scripts.
    - The script also accepts `{'model_state': …}` and strips a `module.` prefix, as MPP's loader does (`MPP/train_basic.py:171-186`).
    - Every tensor is loaded before the expansion.
  - **Check:**
    - The script prints the missing and unexpected keys and asserts there are none.
    - After the expansion, it asserts every tensor is identical to the file's, except the 3 expanded ones (`space_bag.weight`, `debed.out_kernel`, `debed.out_bias`), whose first 12 slots must be identical.
  - **torch version:** the file's zip also holds `byteorder` and `.data/serialization_id` records written by a newer torch. torch 2.0.1 only reads `data.pkl` and the tensor records; the strict load confirms this on the first run.
  - **Scratch:** `build_avit` followed by `expand_projections`, with random weights. MPP applies no custom init: the layer-scale gammas start at 1e-6 (`MPP/models/spatial_modules.py:128-131`; `MPP/models/time_modules.py:31-32`), and every other layer keeps PyTorch's default.
- **Why:** it is the same mechanism as MPP's fine-tuning (`MPP/train_basic.py:168-207`).
- **Source:** `MPP-Ti/finetune.py:366-415`.

### 5.9 Training recipe compared with MPP's fine-tuning

We use the common recipe of 1.2. MPP's numbers come from the `finetune` namespace of `config.yaml`, which inherits the pretraining settings (`config.yaml:75-92`), and from `MPP/train_basic.py`. They are not a recipe from the PhysBiasBench paper.

| Item | Ours (1.2) | MPP repo |
|---|---|---|
| Optimizer | AdamW on all parameters, weight decay 1e-4 | Adan with D-Adaptation, because `learning_rate` is −1 (`config.yaml:27, 30`; `MPP/train_basic.py:128-130`). Weight decay 1e-3 (`config.yaml:31`), not applied to biases or scales (`MPP/train_basic.py:35-53, 122`) |
| Learning rate | 1e-4 | Set by D-Adaptation |
| Schedule | 5% linear warmup, then cosine | Cosine over all steps, with no warmup in the D-Adaptation branch (`MPP/train_basic.py:144-149`) |
| Loss | MSE in normalised space | Relative MSE per sample and per channel: squared error over space divided by the target's mean square (`MPP/train_basic.py:241-248`) |
| Batch size | 16 | 1 per GPU (`config.yaml:22`), with no accumulation for fine-tuning (`config.yaml:79`) |
| Length / stopping | At most 50 epochs, early stopping, best val epoch kept | 500 epochs (`config.yaml:77`) of 2000 samples (`config.yaml:25`); a checkpoint every 10 epochs, plus the best on val loss (`config.yaml:6`) |
| Precision | fp32 (1.3) | Also fp32, mixed precision off (`config.yaml:10`) |
| Gradient clipping | None | 1.0 (`MPP/train_basic.py:266`) |
| Stochastic depth | Up to 0.2 (5.7) | Same |
| Seed | 0 | No global seed; only the validation subset's generator is seeded (`MPP/train_basic.py:342`) |

### 5.10 Code location and pinned versions

- **Chosen:**

  | What | Where | Reason |
  |---|---|---|
  | MPP code at commit `e751fc2` (latest on `main`, 2024-12-06), cloned into `third_party/MPP/` by `third_party/download_mpp_code.py` | `MPP-Ti/finetune.py:56-76` | The only change to `models/` since the paper's submission commit is `462d299` (2024-05-04, "Fixed bias copy during finetuning expansion"), and our expansion uses it |
  | The script checks that `third_party/MPP/.git/HEAD` is that commit, and puts `third_party/MPP/models/` on `sys.path` | `MPP-Ti/finetune.py:60-73` | MPP has no `setup.py`, and its package name `models` would clash with this repo's `models/` folder. Its model modules fall back to plain imports (`MPP/models/avit.py:7-12`, …). The licence is MIT. |
  | `torch==2.0.1` | `requirements.txt:15` | MPP pins torch 2.1 (`MPP/requirements.txt:9`), and its README warns about nightly torch (`MPP/README.md:42`). The model code only uses functions torch 2.0.1 has (`F.scaled_dot_product_attention`, `tensor_split`, `std_mean`), so the environment is shared with the other models. |
  | `torchvision==0.15.2` | `requirements.txt:17` | needed by timm; the release paired with torch 2.0.1 |
  | `timm==0.9.5` | `requirements.txt:23` | provides `DropPath`; same version as `MPP/requirements.txt:8` |
  | `einops==0.7.0` | `requirements.txt:25` | MPP pins 0.7 (`MPP/requirements.txt:5`) |
  | `numpy<2` | `requirements.txt:27` | torch 2.0.1 was built against NumPy 1.x |
  | `huggingface_hub<1.0` | `requirements.txt:29` | needed by timm; keeps the environment compatible with Poseidon-T |
  | Python 3.9 | `requirements.txt:2` | the shared environment; MPP's model files have no Python 3.10+ syntax |

### 5.11 Flat windows left out of MPP's training set

- **Chosen:**
  - A train window is **flat** if at least one of its 5 input frames has a spatial standard deviation below `FLAT_STD_THRESHOLD = 1e-3` **in every channel at once**, in our normalised units (1.4). 1e-3 is a thousandth of the channel's std over the train windows of the same (PDE, mix).
  - MPP trains without its flat windows, for both variants. They are flagged once per run, before training.
  - Validation keeps every window: it only runs forward passes.
  - Each run prints how many windows it leaves out. `best.pt` stores `n_flat_excluded`, `flat_train_windows` (their indices among the dataset's train windows) and `flat_std_threshold`.
  - `MPP-Ti/count_flat_windows.py` prints the counts for every PDE × mix, train and val.
- **What the debug logs showed** (`MPP-Ti/debug_nan.py`, pretrained / Mix-simple / gray_scott, before this change):
  - Epoch 1 produced a non-finite gradient at step 5, with or without gradient clipping at 1.0. The forward output stayed finite, the pretrained model's val loss was finite, and resuming from `last.pt` was not involved.
  - The batch of step 5 contains window 161 (IC-simple, trajectory 16, start 1). It is nearly constant in both channels (whole-window std 4.6e-4 and 4.6e-5), and one frame of channel 1 is exactly constant.
  - **Window 161 alone has a loss of 2.1e-11 but NaN or 1e37 gradients. The same batch without it has normal, finite gradients.**
  - **A single flat channel in a frame is harmless:** the debug probe with one uniform frame in one channel gave finite outputs and gradients, and Wave trains finitely although its velocity channel is identically zero at t = 0.
  - **What breaks MPP is a frame that is flat in every channel at once.** With the second criterion below, gray_scott / Mix-simple still aborted, with 2 of its 186 steps non-finite. The criterion had kept windows 160, 640 and 1260: start 0 of trajectories that decay to a flat state. Their frame 0 still has structure, but their later frames are flat in both channels.
- **Mechanism:**
  - MPP embeds each frame separately: a per-pixel linear map over the channels (`space_bag`), then the stem, whose convolutions are each followed by an `RMSInstanceNorm2d` (`MPP/models/avit.py:110-117`; `MPP/models/spatial_modules.py:72-94`).
  - That layer divides each feature map by its spatial std + 1e-8, without subtracting the mean (`MPP/models/spatial_modules.py:26-40`).
  - If only one channel of a frame is flat, the other channel still varies, and so do the feature maps.
  - If every channel of a frame is flat, every feature map of that frame is spatially constant, and the layer divides by about 1e-8.
- **Why this criterion:**
  - It matches that mechanism: it flags exactly the frames that give constant feature maps, and nothing else.
  - The 1e-3 threshold is applied in normalised units, so it means the same thing for every PDE and channel.
  - Wave's windows starting at frame 0 stay in training: their velocity is flat at t = 0, but their height is not. `MPP-Ti/debug_nan.py` step 10 checks it, and step 9 checks that windows 161, 160, 640 and 1260 are excluded.
- **Why the training objective is essentially unchanged:**
  - **Most flagged windows contribute almost nothing to the loss.** MPP de-normalises its output with each window's own std and mean (`MPP/models/avit.py:129-130`). On a near-constant window, prediction and target both sit at the window's mean, whatever the network computes: window 161's loss is 2.1e-11. What these windows add is their NaN or 1e37 gradients.
  - **They are a small share** of the train windows; `count_flat_windows.py` and each run print the exact share.
  - **Validation, early stopping and the choice of the best epoch still use every window,** so models are compared on the same windows as the other models.
  - **Caveat:** one flat frame is enough to flag a window, so some flagged windows still carry signal in their other frames. Windows 160, 640 and 1260 are examples: frame 0 still has structure. `count_flat_windows.py` shows how many windows are flagged.
- **What was ruled out:**
  - **1st criterion, a frame flat in ANY channel:** it excluded every Wave window starting at frame 0, i.e. exactly 300 of 3000 train windows in every mix. The Wave velocity is identically zero at t = 0 by construction (paper, Appendix A.2: the initial velocity amplitude is zero). Those windows carry real signal, and every Wave test input contains that frame, so they must stay in training.
  - **2nd criterion, ANY channel near-constant over the whole window** (its 5 input frames and space together, the std MPP's per-sample normalisation divides by, `MPP/models/avit.py:106-107`): it kept windows 160, 640 and 1260, whose frame 0 lifts the whole-window std above the threshold, and gray_scott / Mix-simple still had 2 non-finite steps out of 186.
  - **Gradient clipping:** it did not help (step 6 of the debug run), because it cannot repair a non-finite gradient.
  - **Patching MPP's `RMSInstanceNorm2d`** to divide by √(var + ε): epoch 1 still failed, and the pretrained model's outputs on normal val windows changed by 1.5e-3 relative. The patch was reverted.
  - **Leaving these windows out of validation:** validation only runs forward passes, which stay finite, and it should stay identical across models.
- **Source:** `MPP-Ti/finetune.py:120-124, 341-362, 454-456, 516-521`.

### 5.12 Safety net: steps with non-finite gradients are skipped

- **Chosen:**
  - After each backward pass, the script computes the total gradient norm in float64. If it is non-finite, the batch is skipped: no optimizer step, gradients cleared, and the batch is left out of the train loss. The scheduler still steps.
  - The float64 norm is non-finite exactly when some gradient entry is non-finite. Finite float32 gradients cannot overflow it, unlike a float32 norm, which overflows above about 1.8e19.
  - Each epoch prints its number of skipped steps. Each skipped batch is printed as it happens, with its windows' indices and flatness scores: the min over the input frames of the max over channels of the frame's spatial std, so a window is flat (5.11) when its score is below `FLAT_STD_THRESHOLD`. This is logging only.
  - If more than 5% of an epoch's steps are skipped (`MAX_SKIPPED_FRACTION = 0.05`, about 9 of the 186 steps of gray_scott / Mix-simple), the run aborts with an error. A run that skips that often has a systematic problem, which resubmitting would not fix.
- **Why:**
  - It guards against flat windows the threshold misses, in this run or in other PDEs and mixes. A single AdamW step with a NaN or inf gradient corrupts the weights for good, which is how the original run crashed.
  - **Why 5% and not 1%.** With the flat windows excluded, epoch 1 of gray_scott / Mix-simple skipped 1 of its 186 steps in one run and 0 in another, with the same 30 windows excluded. So that skip was a rare event that depends on the hardware, not a borderline window:
    - `count_flat_windows.py` gives the same counts at every flatness-score threshold from 1e-3 to 3e-2: 60 windows, all far below 1e-3, while every other window scores above 3e-2.
    - So there is no window near the threshold, and `FLAT_STD_THRESHOLD` stays at 1e-3.
    - At 1%, two such rare skips in one epoch (1% of 186 steps is 1.86) would abort the run. The abort also stops the whole job, including the runs after it.
    - 5% tolerates rare skips and still stops a run that produces non-finite gradients systematically.
  - The limit also keeps the safety net from quietly changing the training objective: a run that would need it often stops instead, and its windows can be checked.
- **Limit:** only non-finite gradients are caught. A huge but finite gradient, like the 1e37 one, would still pass and could overflow AdamW's second-moment estimate. The exclusion of flat windows (5.11) is what keeps those out.
- **Source:** `MPP-Ti/finetune.py:125-127, 269-273, 470-504, 575-578`.
- **MPP repo:** MPP's trainer clips gradients at 1.0 (`MPP/train_basic.py:266`); we keep the common recipe of no clipping (1.2), since clipping did not help here (5.11).

### 5.13 Smaller implementation details

- **Apple GPUs.** On MPS, the script first tries MPP's masked attention and its statistics function once, and falls back to the CPU if they fail (`MPP-Ti/finetune.py:44-45, 189-202`). On CUDA nothing changes.

---

## 6. Inference

Code: `Poseidon-T/infer.py`, `DPOT-Ti/infer.py`, `GPhyT-S/infer.py` and `MPP-Ti/infer.py`. They save predictions only and compute no metric.
- The four files are identical except for the model name in the docstring and the model-specific section (`Poseidon-T/infer.py:71-110`, `DPOT-Ti/infer.py:71-125`, `GPhyT-S/infer.py:71-130`, `MPP-Ti/infer.py:71-130`).
- Line numbers in 6.1-6.6 refer to `Poseidon-T/infer.py`. Every line after the model-specific section is 15 lines further down in `DPOT-Ti/infer.py`, and 20 lines further down in `GPhyT-S/infer.py` and `MPP-Ti/infer.py`.

### 6.1 Autoregressive rollout

- **Chosen:**
  - The model gets ground-truth frames 0-4 and predicts frames 5-19 one at a time.
  - Each prediction is appended to the 5-frame input window and the oldest frame is dropped. After frame 4 the model only sees its own predictions: ground truth is never used again.
  - All 15 frames come from a single rollout. The 10 in-horizon frames and the 5 OOD rollout frames are only told apart at evaluation.
- **Why:**
  - The paper gives the models 5 input frames, then evaluates 10 in-horizon frames and 5 extra "OOD rollout frames" (Figure 1 caption).
  - Its RolloutAmplification metric, E_roll / E_1-step, measures "long-horizon growth from short-horizon error" (Table 2). Errors can only grow from step to step if each step starts from the previous prediction.
  - Every model is trained for one step only, with teacher forcing (`Poseidon-T/finetune.py:410-416`). Feeding predictions back is the only way to reach frames 6-19.
- **Source:** `Poseidon-T/infer.py:257-275`.
- **Paper:** says "rollout" but never "autoregressive", and does not say how the input window is updated.

### 6.2 Runs and checkpoint used

- **Chosen:**
  - Every finished run: `best.pt` exists and `last.pt` does not, the same test as in training (`Poseidon-T/finetune.py:184-186`). Unfinished runs are listed in the log and skipped.
  - The weights come from `best.pt`, the epoch with the lowest one-step val loss (1.2).
  - Each model runs only on its own PDE's grid: 25 cells × 50 trajectories (`count=None`).
  - The trajectories must come in file order (sample_index 0..49), so row k of a saved array is test trajectory k. The script stops otherwise.
- **Why:** `best.pt` is the model that training selected. An unfinished run's `best.pt` can still change.
- **Source:** `Poseidon-T/infer.py:190-213, 233-254, 329-332`.

### 6.3 Normalisation

- **Chosen:**
  - The inputs are normalised with the `norm_mean` and `norm_std` saved in the run's `best.pt` (1.4).
  - The whole rollout stays in normalised space: a step's output is the next step's input as it is.
  - The predictions are de-normalised once, before saving, and saved in raw physical units.
  - Both conversions use float64 arithmetic, then float32, as in training.
- **Why:** this is exactly the mapping the model was trained with. In raw units, the evaluation needs no normalisation constants.
- **Source:** `Poseidon-T/infer.py:216-230, 354-355`.

### 6.4 fp32 inference

- **Chosen:** the same precision as training (1.3):
  - strict float32, no autocast;
  - `torch.set_float32_matmul_precision("highest")`, and TF32 off for matmuls and cuDNN;
  - `model.eval()` and `torch.no_grad()`;
  - the device, chosen in the model-specific section: the same logic as training for Poseidon-T; cuda, else cpu, never MPS for DPOT-Ti, GPhyT-S and MPP-Ti (6.8-6.10). On Lab-IA all give cuda, as in training;
  - batches of 16 trajectories, the training batch size.
- **Why:** predictions should not depend on numerical precision (1.3).
- **Source:** `Poseidon-T/infer.py:90, 183-187, 257, 266, 379-380`.

### 6.5 Non-finite values kept as they are

- **Chosen:**
  - The outputs are never clipped, replaced or post-processed.
  - A NaN or inf is fed back into the next step as it is, and saved as it is.
  - The manifest counts the non-finite values of each cell (`n_nonfinite`), and the log prints the totals.
- **Why:** a diverging rollout is a result. Replacing the values would hide it and change the errors. The evaluation decides how to count them.
- **Source:** `Poseidon-T/infer.py:262-264, 300, 359-368`.

### 6.6 Output files

- **Chosen:**
  - One `.npy` per (run, cell): `predictions/Poseidon-T/{variant}/{mix}/{pde}/{dynamic}__{ic}.npy`, float32, shape [50, 15, C, 64, 64]. Frame j is sequence frame 5 + j: j = 0-9 are in-horizon, 10-14 are OOD rollout.
  - The ground truth is not saved; it stays in the dataset.
  - `manifest.csv` has one row per `.npy`. Its `shift_group` column uses the paper's five groups (Section 3.3):
    - train-seen: the 3 (dynamic, IC) pairs the train mixes are built from (`data/benchmark_api.py:49-53`);
    - joint-OOD: both names contain "OOD";
    - dynamic-OOD or IC-OOD: only that axis's name contains "OOD";
    - compositional-ID: every other cell.

    That gives 3 train-seen, 6 compositional-ID, 6 dynamic-OOD, 6 IC-OOD and 4 joint-OOD cells, which the script checks.
  - Every file is first written to a temporary file, then renamed. A cell whose `.npy` already exists is skipped, so a stopped job resumes when it is submitted again.
  - With `SMOKE_TEST`, everything goes to `predictions/Poseidon-T_smoke/` instead.
- **Why:** our choice of format for the evaluation's inputs.
- **Source:** `Poseidon-T/infer.py:113-140, 155-180, 278-323, 338-347`.

### 6.7 Poseidon-T step

- **Chosen:**
  - The model is rebuilt from the run's own `scot_config` and weights, both read from `best.pt` (`Poseidon-T/infer.py:82-90`).
  - Each step is training's `predict` (`Poseidon-T/finetune.py:317-320`), called from `finetune.py` itself. scOT gets the last frame of the current window and the lead time 1/19. At the first step that frame is ground-truth frame 4; after that it is the previous prediction.
  - The lead time is 1/19 at every step. We never ask for frame 4 + k directly with lead time k/19.
  - `frame_used` and `lead_time` are read from `best.pt` and checked against `finetune.py`.
  - scOT still resamples 64→128→64 internally (2.3).
- **Why:**
  - This is exactly the one-step map the model was fine-tuned on (2.1, 2.2).
  - scOT could take a larger lead time in a single call, but fine-tuning only ever used 1/19, and a direct jump would not be a rollout (6.1).
- **Source:** `Poseidon-T/infer.py:93-110`.

### 6.8 DPOT-Ti step

- **Chosen:** each step is training's `predict` (`DPOT-Ti/finetune.py:380-399`), called from `finetune.py` itself, on all 5 frames of the current window.
  - **Frame padding:** the window is padded to the 10 frames DPOT-Ti takes by repeating its first (oldest) frame: [w0]×6 + [w1, w2, w3, w4], as in training (3.1). At the first step w0 is ground-truth frame 0; later it is whichever frame is oldest in the window.
  - **Channel padding:** the C real channels are padded to 4 with 1.0, built anew at every step (3.2). The window only keeps the C real channels of each prediction, so the 4 − C padded output channels are never fed back.
  - **Resampling:** the window is Fourier-upsampled 64→128 before the model, and the prediction is Fourier-truncated 128→64 after it, with training's routine (3.3). So the window stays at 64×64 between steps.
  - **No noise:** noise_scale 0, as in training (3.4). DPOTNet adds no noise itself (`DPOT/models/dpot.py:364-403`); DPOT's scripts add it to the inputs before calling the model (`DPOT/finetune.py:218`).
  - **Output:** only `pred[..., 0, :C]` is kept, i.e. the single output step and the C real channels. `cls_pred` is ignored (3.6).
  - **Checks:** the model is rebuilt from the `dpot_config` and weights in `best.pt` (`DPOT-Ti/infer.py:84-94`). The config, the DPOT commit, `frames_used`, `time_padding`, `channel_pad_value` and `resampling` saved in `best.pt` must equal those of `DPOT-Ti/finetune.py`.
  - **Device:** cuda, else cpu, never MPS (`DPOT-Ti/infer.py:77-81`). On Apple GPUs, DPOT's complex FFT aborts the process, so the MPS test in `DPOT-Ti/finetune.py:206-216` cannot fall back to the CPU. On CUDA this is training's choice.
- **Why:**
  - This is exactly the one-step map the model was fine-tuned on.
  - Fresh channel padding: in training, the padded input channels were always 1.0 and the padded output channels had no loss, so their predicted values mean nothing. Feeding them back would give the model inputs it never saw.
  - Resampling at every step: in training, the inputs were always upsampled 64×64 frames, with no modes above the 64×64 grid's Nyquist frequency. Keeping the window at 128×128 between steps would feed back high frequencies the model never saw as input.
- **Source:** `DPOT-Ti/infer.py:71-125`.
- **DPOT repo:** its own test loop also rolls out autoregressively: it appends the prediction and drops the oldest frame (`DPOT/finetune.py:277-287`). But it feeds back all 4 predicted channels, padded ones included, and it does not resample between steps.

### 6.9 GPhyT-S step

- **Chosen:** each step is training's `predict` (`GPhyT-S/finetune.py:404-425`), called from `finetune.py` itself, on the last 4 frames of the current window (4.2).
  - **Fields:** the C real channels go into the first C of GPhyT's 5 field slots, and the other 5 − C are filled with 0.0, built anew at every step (4.3). The window only keeps the C real channels of each prediction, so the padded output fields are never fed back.
  - **Resampling:** the 4 frames are Fourier-resampled 64 × 64 → 256 × 128 (×4 along H, ×2 along W) before the model, and the prediction 256 × 128 → 64 × 64 after it, with training's routine (4.4). So the window stays at 64 × 64 between steps.
  - **Integrator:** the model's forward applies the Euler integrator, as in training (4.5): the last input frame plus the network's output (`GPhyT/gphyt/model/transformer/model.py:243-248`, `GPhyT/gphyt/model/transformer/num_integration.py:39-41`).
  - **Output:** only the first C fields of the single output frame are kept.
  - **Checks:** the model is rebuilt from the `gphyt_config` and weights in `best.pt` (`GPhyT-S/infer.py:84-96`). The config (Euler integrator included), the GPhyT commit, `frames_used`, `gphyt_fields`, `field_slots`, `field_pad_value` and `resampling` saved in `best.pt` must equal those of `GPhyT-S/finetune.py`.
  - **Device:** cuda, else cpu, never MPS, as for DPOT-Ti (6.8): the Fourier resampling uses the same complex FFT ops (`GPhyT-S/infer.py:77-81`). On CUDA this is training's choice.
- **Why:**
  - This is exactly the one-step map the model was fine-tuned on.
  - Fresh field padding: in training, the padded input fields were always 0.0 and the padded output fields had no loss, so their predicted values mean nothing. Feeding them back would give the model inputs it never saw.
  - Resampling at every step: in training, the inputs were always upsampled 64 × 64 frames. Keeping the window at 256 × 128 between steps would feed back high frequencies the model never saw as input.
- **Source:** `GPhyT-S/infer.py:71-130`.
- **GPhyT repo:** its own rollout also appends the prediction and drops the oldest frame (`GPhyT/gphyt/run/model_eval.py:402-416`, with `rollout=True`). It differs from ours between steps in four ways:
  - it runs under bf16 autocast (`GPhyT/gphyt/run/model_eval.py:398-401`); we stay in fp32 (6.4);
  - it feeds back all 5 predicted fields, including those a dataset does not use; those are only dropped for the loss (`GPhyT/gphyt/run/model_eval.py:412, 443-445`);
  - it does not resample: its data is already at 256 × 128;
  - it stops at the first NaN or inf and fills the remaining frames with NaN (`GPhyT/gphyt/run/model_eval.py:405-407, 423-434`). We keep rolling out and save the values as they are (6.5).

### 6.10 MPP-Ti step

- **Chosen:** each step is training's `predict` (`MPP-Ti/finetune.py:418-426`), called from `finetune.py` itself, on all 5 frames of the current window (5.2).
  - **Resolution:** native 64 × 64, no resampling (5.4). The window is fed as it is at every step.
  - **State labels and bcs:** the run's state labels 10 .. 10 + C − 1 (5.3) and the periodic bcs [1, 1] (5.6), at every step. `predict` builds them from `MPP-Ti/finetune.py`, and the values saved in `best.pt` must be the same.
  - **Output:** MPP returns the next frame only, de-normalised by its own per-sample normalisation (5.5), so it comes back in our normalised space and is appended to the window as it is.
  - **No flat-window exclusion:** leaving flat windows out (5.11) and skipping steps with non-finite gradients (5.12) are training-only measures. At inference every test trajectory is predicted. If an input makes MPP's output non-finite, for example a frame constant in every channel, the prediction is saved as it is and counted in `n_nonfinite`, as for every model (6.5).
  - **Checks:** the model is rebuilt as in training: `build_avit` with the `mpp_config` in `best.pt`, then `expand_projections(C)`, then the weights (`MPP-Ti/infer.py:83-102`). The config, `drop_path`, the MPP commit, the width of the 3 expanded tensors (12 + C), `frames_used`, `state_labels`, `bcs` and `resampling` saved in `best.pt` must equal those of `MPP-Ti/finetune.py`.
  - **Device:** cuda, else cpu, never MPS, as for the other models (`MPP-Ti/infer.py:77-80`). On CUDA this is training's choice.
- **Why:** this is exactly the one-step map the model was fine-tuned on. Excluding flat test trajectories would change the test set, which the paper fixes (50 trajectories per cell).
- **Source:** `MPP-Ti/infer.py:71-130`.
- **MPP repo:** has no rollout code. Its validation predicts one step from ground-truth inputs only (`MPP/train_basic.py:352-357`), so there is nothing between steps to compare with.

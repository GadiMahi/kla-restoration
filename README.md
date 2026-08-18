# KLA Image Restoration — Data Pipeline

Joint denoising + 2× super-resolution for semiconductor inspection images.
SEMICON India Hackathon 2026. **Phase 1 deadline: 16 August 2026.**

This repo currently contains the **EDA / preprocessing / I-O half** of the
solution (owner: Mahi). The model, losses and training loop plug into the
interfaces described below without changing anything here.

---

## Quick start (Kaggle)

```python
!git clone https://github.com/GadiMahi/oomsurvivors.git
%cd oomsurvivors
!pip install -q -r requirements.txt

DATA = "/kaggle/input/<dataset-slug>"

# 1. FORMAT CONTRACT + INVENTORY  <- run this first, always
#    Interactive version with plots: notebooks/01_inventory.ipynb
!python scripts/run_inventory.py --set data.root=$DATA

# 2. Cache + dataloader throughput
!python scripts/make_cache.py --set data.root=$DATA cache.dir=/kaggle/working/cache

# 3. Splits, including the OOD proxy
!python scripts/make_splits.py --set data.root=$DATA

# 4. Mandatory baseline
!python scripts/baseline_bicubic.py --set data.root=$DATA

# 5. End-to-end inference harness (runs today with the bicubic placeholder)
!python inference.py --input_dir $DATA/NoisyLR --output_dir /kaggle/working/out
```

Every path is config-driven. Nothing requires a source edit — a hard
requirement of the KLA spec (section 4C).

---

## Run this first, and read the output

`scripts/run_inventory.py` answers the four questions that everything else
depends on, in order of consequence:

1. **What format is the ground truth, and can we reproduce it exactly?**
   KLA scores images *exactly as saved* and performs no clipping or
   renormalisation. If our output format doesn't match the GT format, the
   score is capped by an I/O bug rather than by the model. The script fails
   loudly if the round-trip check doesn't pass.
2. Are pairs complete, grayscale, and exactly 2×?
3. How far outside `[0,1]` does NoisyLR actually go? (The spec says this
   overshoot is intentional and must be handled, not clipped away.)
4. Is the intensity range stable enough for fixed global normalisation?

---

## Interfaces for the model team

**Normalisation** — `src/transforms.py` is the single source of truth. Every
constant lives in `artifacts/stats.json`. Do not write a scale factor anywhere
else.

```python
from src.transforms import normalize, denormalize
```

* The **input is never clipped** — out-of-range NoisyLR values are real signal.
* The **output is always clipped to [0,1]** inside `denormalize`.
* `denormalize(normalize(x)) == x` is unit-tested: `python -m pytest tests/ -q`

**Model registry** — register the architecture in `src/model.py` and
`inference.py` picks it up unchanged:

```python
@register("nafnet")
def _nafnet(scale=2, **kw):
    return NAFNetSR(scale=scale, **kw)
```

Then set `inference.pad_multiple` in `configs/default.yaml` to the network's
stride (e.g. 16 for a 4-level NAFNet). Padding and unpadding are already
handled.

**Loss and metric helpers** — `src/eval_utils.py`

* `edge_weight(hr)` — per-pixel weight map for the Charbonnier term. Wafer
  images are mostly flat; this puts gradient where the structure is.
* `stratified_ssim(pred, gt)` — returns `ssim`, `ssim_edge`, `ssim_flat`.
  Global SSIM hides line smearing. `ssim_edge` is the diagnostic that matters,
  given the spec's "do not blur the image to remove noise".

**Synthetic augmentation** — the measured degradation is reproducible, so extra
training pairs can be generated on the fly from the clean GT images:

```python
from src.dataset import RestorationDataset, degrade_cfg_from_stats

ds = RestorationDataset(cache_dir, stems=sp["train"], lr_patch=64, grad_thresh=thr,
                        synth_p=0.5,                       # half the batch synthesised
                        degrade_cfg=degrade_cfg_from_stats(width=0.3),
                        jitter_range=(0.7, 1.4))
...
ds.set_width(w)      # curriculum: ramp 0.3 -> 1.0 over training
```

`jitter_range` rescales the GT before degrading it, which varies apparent feature
size. That is the main defence against the resolution gap: every training pair is
256->128, but evaluation may include 512x512 content.

**Validation** — `artifacts/splits.json` has `train`, `val_id` and `val_ood`.
`val_ood` is an entire held-out structure cluster and is the **primary
metric**; `val_id` is a sanity check. KLA's hidden test set contains
unfamiliar image *content*, so in-distribution validation will look good
regardless and tell us very little.

---

## Design decisions, and the spec lines behind them

| Decision | Why |
|---|---|
| Output format mirrors GT exactly | "KLA will score the images exactly as saved by the submitted pipeline"; "KLA does not clip or renormalize outputs" |
| Input never clipped; output clipped to [0,1] | "NoisyLR values may extend slightly outside [0,1]; this is intentional" |
| Degradation **order sampled per image** | "The three degradations may have been applied in any order" |
| Noise jitter kept modest (±30%) | "Noise mechanisms remain the same; sampled levels may vary within a similar range" — over-widening makes the model hedge and blur |
| Augmentation prioritises **content** (scale jitter) over noise | Test set OOD is unfamiliar *image content*, not unfamiliar degradations |
| Gradient-based crop rejection sampling | Wafer images are mostly flat die area; uniform crops waste training on blank regions |
| Memmap cache, never float16 | Per-item decode starves the GPU; float16 precision sits too close to the 8-bit floor |
| Per-stage timing in `inference.py` | Runtime "includes disk reading, preprocessing, CPU-to-GPU transfer, model execution, GPU-to-CPU transfer, post-processing and saving" |
| No hardcoded paths; seeds fixed | "Training & compute hygiene" is a scored evaluation axis |

---

## Notebooks vs scripts

Both exist on purpose.

* **`notebooks/`** — the exploratory blocks, where you need to *see* things:
  histograms, sample images, the variance-vs-signal scatter, kernel rankings.
  Written for Kaggle; they `sys.path` into `src/` and use `%autoreload`.
* **`scripts/` + `src/`** — the reproducible pipeline. The submission requires a
  repo that evaluators can run without editing source, and "training & compute
  hygiene" is a scored evaluation axis, so the pipeline cannot live in notebook
  cells.

The notebooks call into `src/`, so there is one implementation, not two.

| Notebook | Covers | Writes |
|---|---|---|
| `01_inventory.ipynb` | format contract, shapes, ranges, overshoot, normalisation decision, visual check | `artifacts/stats.json` |
| `02_degradation.ipynb` | kernel recovery, sub-pixel alignment, noise model fit, degradation order, log-transform decision, recipe verification | updates `artifacts/stats.json` |

`scripts/run_inventory.py` is the headless equivalent of notebook 01, for CI and
for re-running after the data changes.

## Layout

```
configs/default.yaml     all paths + hyperparameters; override with --set
src/config.py            config loader
src/io_utils.py          format detection, lossless load, GT-matching save   <- highest risk
src/transforms.py        normalize / denormalize contract
src/cache.py             decode once into memmaps
src/dataset.py           patches, gradient rejection sampling, mixed resolutions
src/degrade.py           synthetic pairs; randomised degradation order
src/augment.py           scale jitter, D4, CutBlur
src/splits.py            structure clustering + OOD proxy split
src/eval_utils.py        PSNR / SSIM / LPIPS + edge-stratified SSIM
src/model.py             model registry (bicubic placeholder today)
notebooks/               interactive EDA, imports from src/
inference.py             standalone --input_dir/--output_dir harness, timed
scripts/                 runnable entry points
tests/                   normalisation round-trip
```

## Status

Verified without the real dataset, using `scripts/make_dummy_data.py`:

* format round-trip passes for `npy`, `tiff32`, `png16`, `png8`
* inventory, shape/range/overshoot reporting, normalisation decision
* structure clustering and OOD split
* kernel recovery correctly identified the planted kernel
* noise-model fit recovered sigma_mult within ~20% of the planted value

One method was corrected as a result: the lag-1 autocorrelation test for
degradation order **fails at 2x** (reported 0% when the truth was 40%). Notebook
02 now uses forward-simulation hypothesis testing as the primary method and keeps
autocorrelation only as a secondary signal.

Not yet exercised (no GPU/torch in the authoring environment) — **verify these
on Kaggle in the first session:** `make_cache.py`, `dataset.py`,
`baseline_bicubic.py`, `inference.py`.

## Not yet built (model team)

`train.py`, the NAFNet + PixelShuffle architecture, the composite
Charbonnier + SSIM + LPIPS loss, and the checkpoint format.

# KLA Image Restoration – SEMICON India Hackathon 2026

**Team: OOM Survivors**

## Contributors
<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/RK-NerdyBirdy">
        <img src="https://github.com/RK-NerdyBirdy.png" width="115px;" alt="Maneet Gupta"/>
        <br /><sub><b>Maneet Gupta</b></sub>
      </a><br />
      <a href="https://www.kaggle.com/rknerdybirdy">Kaggle</a> &nbsp;|&nbsp; <a href="https://www.linkedin.com/in/maneet-gupta/">LinkedIn</a>
    </td>
    <td align="center">
      <a href="https://github.com/GadiMahi">
        <img src="https://github.com/GadiMahi.png" width="115px;" alt="Mahi Gadi"/>
        <br /><sub><b>Mahi Gadi</b></sub>
      </a><br />
      <a href="https://www.kaggle.com/mahigadi">Kaggle</a> &nbsp;|&nbsp; <a href="https://www.linkedin.com/in/mahigadi/">LinkedIn</a>
    </td>
  </tr>
</table>
---

## Project Overview

**Phase 1 Submission for the SEMICON India Hackathon 2026.**

This project is an AI-based restoration pipeline performing **joint denoising and 2× spatial resolution recovery** on signal-degraded grayscale semiconductor inspection images. It utilizes a custom NAFNet-style U-Net architecture featuring a 2x pixel-shuffle upsampler with a bilinear-upsample residual shortcut.

Every path is config-driven. Nothing requires a source edit — a hard requirement of the KLA spec (section 4C).

---

## Quick Start & Setup

**No internet access, API keys, or additional downloads are required at inference time.** The `models/best_nafnet.pt` checkpoint contains everything the model needs.

```bash
# Clone the repository
git clone https://github.com/GadiMahi/oomsurvivors.git
cd oomsurvivors

# Set up a virtual environment (optional but recommended)
python -m venv .venv && source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt

```

---

## Running Inference

The `run.py` script is the mandatory evaluation entry point. It is fully self-contained and fixes previous registry bugs so the "nafnet" architecture resolves perfectly.

```bash
python run.py <input_dir> <output_dir>

```

* **`<input_dir>`**: Directory containing degraded `.npy` grayscale arrays, shape `(H, W)` or `(H, W, 1)`.
* **`<output_dir>`**: Created automatically if it doesn't exist. One `restored.npy` file is written per input file, using the exact same filename.

### Output Contract

* **Format:** One `.npy` per input, same base filename.
* **Shape:** Grayscale, shape `(H, W)` or `(H, W, 1)` (matches the input's ndim).
* **Data Type:** `float32`, values clamped strictly to `[0, 1]`, with no `NaN/Inf` (defensively sanitized with `np.nan_to_num` on both the read and write sides).
* **Resolution:** Spatial resolution is restored at a fixed **2x super-resolution factor** (e.g., 128→256 or 256→512), matching the training config (`dataset.scale = 2`).
* **Dynamic Sizing:** Inputs are batched and grouped by shape for efficient, fully batched GPU execution. Height/width are reflect-padded to a multiple of 2 internally (required by the stride-2 U-Net level) and seamlessly cropped back after upsampling, supporting arbitrary input sizes.

---

## Model Architecture & Training

**Architecture:** `NAFNet_UNet (in_channels=1, out_channels=1, dim=64, scale=2)`

* **Flow:** Intro conv → 2 NAFBlocks → stride-2 down → 2 NAFBlocks (bottleneck) → 2 NAFBlocks (middle) → PixelShuffle(2) up → concat skip → reduce → 2 NAFBlocks → PixelShuffle(2) SR tail.
* **Shortcut:** A bilinear-upsampled input is added back at the end as a residual shortcut to preserve global structure.

**Training Details:**
Trained with a carefully balanced, multi-objective weighted loss to prevent edge blurring:
`1.0 * Charbonnier + 0.05 * LPIPS(vgg) + 0.5 * Sobel-edge L1`

*(See the `train_nafnet.py` script for the full training loop, loss functions, AdamW optimizer, Cosine Annealing scheduler, and OOD validation via `stratified_ssim`.)*

---

## Data Pipeline & Reproducing Training

To reproduce the training pipeline from scratch (e.g., on Kaggle), follow these exact steps. The data pipeline is designed to be highly memory-efficient, utilizing memory mapping (`mmap`) to prevent RAM exhaustion.

```bash
export DATA="/kaggle/input/<dataset-slug>"

# 1. FORMAT CONTRACT + INVENTORY (Run this first, always)
python scripts/run_inventory.py --set data.root=$DATA

# 2. Build fast I/O Memmap Cache
python scripts/make_cache.py --set data.root=$DATA cache.dir=/kaggle/working/cache

# 3. Generate Splits (including OOD proxy)
python scripts/make_splits.py --set data.root=$DATA

# 4. Train the Model
python train_nafnet.py --set data.root=$DATA cache.dir=/kaggle/working/cache output.dir=/kaggle/working/artifacts

```

---

## Design Decisions & Spec Justifications

| Decision | Why (Based on KLA Spec) |
| --- | --- |
| **Output format mirrors GT exactly** | "KLA will score the images exactly as saved by the submitted pipeline"; "KLA does not clip or renormalize outputs." |
| **Input never clipped; output clipped to [0,1]** | "NoisyLR values may extend slightly outside [0,1]; this is intentional." |
| **Memmap cache, never float16** | Per-item decode starves the GPU; `float16` precision sits too close to the 8-bit floor. |
| **Gradient-based crop rejection sampling** | Wafer images are mostly flat die area; uniform random crops waste training on blank regions. |
| **Augmentation prioritises content over noise** | Test set OOD contains unfamiliar *image content*, not unfamiliar degradations. Jitter range rescales GT before degrading. |
| **Noise jitter kept modest (±30%)** | "Noise mechanisms remain the same; sampled levels may vary within a similar range" — over-widening makes the model hedge and blur. |
| **Per-stage timing in `inference.py**` | Runtime "includes disk reading, preprocessing, CPU-to-GPU transfer, model execution, GPU-to-CPU transfer, post-processing and saving." |
| **No hardcoded paths; seeds fixed** | "Training & compute hygiene" is a directly scored evaluation axis. |
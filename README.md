# GeoEdit

Manifold Geometry-Guided Model Editing for Vision Transformers.

## Method

GeoEdit estimates a diagonal empirical Fisher from anchor images and combines it with the editing gradient to select a sparse parameter subspace. A Gaussian coordinate multiplier randomizes the Fisher-conditioned scores. This multiplier is not a curvature estimate or a finite difference of a re-estimated Fisher.

In real arithmetic, the localization score has the form

$$I_i=0.1\,|g_i|\,|z_i|\frac{\widehat F_i}{\widehat F_i^2+10^{-5}},\qquad z_i\sim\mathcal N(0,1).$$

The implementation retains the original tensor arithmetic used to construct the multiplier. A global Top-k mask enforces the parameter budget. Two-stage masked optimization combines an editing objective, a preservation objective, and parameter regularization. Neighborhood samples are used for evaluation, not for training.

## Installation

Use Python with a compatible PyTorch and torchvision installation. Install the appropriate PyTorch build for your hardware, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The pretrained torchvision ViT-B/16 weights are downloaded on first use unless already cached. Dataset images and model weights are not included.

## Data

Set dataset paths in `settings.py`. Natural groups are directories named with the true and incorrect ImageNet class indices, separated by an underscore. Images are stored directly inside each group directory. Sensitive images are stored directly inside the sensitive-data directory.

```text
datasets/
  Natural-Image-Subset/
    407_654/
      ... image files ...
  Sensitive-Images/
    ... image files ...
```

The data settings reserve 20 anchors per group, split the remaining natural images into editing and neighborhood subsets at an 80:20 ratio, and use 50 preservation-training images and 100 preservation-evaluation images. The dataset manager uses its fixed split seed of 42. The runtime seed controls the other random operations. Data and ablation settings are Python dictionaries; no YAML file is required.

## Run

Optimization settings are explicit command-line arguments. Budget is a fraction: `0.0002` means **0.02%**, not 0.02.

```bash
python run.py --group 0 --seed 42 --budget 0.0002 --lr 0.003 --steps 100 --warmup-steps 20 --ramp-steps 30 --lambda-edit 1.0 --lambda-edit-base 0.1 --lambda-locality 5.0 --lambda-reg 0.001
```

`--group` is the zero-based index in the sorted group directory list. Other runtime options include `--exp-name`, `--eval-interval`, and `--fisher-batch-size`. The Fisher implementation accumulates per-image squared gradients; the batch-size argument does not enable batched Fisher computation.

This release contains no early stopping. Runs execute the requested number of optimization steps. Periodic R/L/G evaluation only records metrics. Removing stopping rules does not establish equivalence to historical runs that used different stopping conditions.

## Ablation and mask visualization

`ablation.py` and `visualize_masks.py` use the same explicit optimization arguments. Run their `--help` commands to see their additional options. Ablation settings, including random repeats, are in `settings.py`.

The retained ablation aggregation reports population standard deviations and pools random-repeat observations. For manuscript statistics defined as one observation per editing group, average repeats within each group first and compute the stated group-level standard deviation separately.

## Outputs

The run logger writes metrics and text logs under `results/`. This repository contains the editing implementation and ablations from the source project; it does not include separate baseline implementations.

# SNN Calorimeter Reconstruction Pipeline

Spiking Neural Network (SNN) pipeline to reconstruct deposited energy and the 3D
energy centroid per event from temporal photon-scintillation data of a PbWO₄
calorimeter segmented into cubelets (10×10×10 grid, 1000 cubelets, 100 sensors).

The pipeline has three stages, run in order:

```
generate_dataset.py   →   train_model.py   →   print_predictions.py
   (raw → .pt)              (.pt → .pth)         (.pt + .pth → plots/metrics)
```

Energy is **always** handled in `log10(E/MeV)`. The linear-energy view is only
recovered afterwards (in `print_predictions.py`) by undoing the log10; the model
is never trained or evaluated on linear energy.

---

## Requirements

```bash
pip install torch snntorch numpy matplotlib scipy pandas scikit-learn tqdm
```

The split seed is fixed to `42` in both `train_model.py` and
`print_predictions.py`, so the train/val/test partition (70 % / 15 % / 15 %) is
identical across training and evaluation. Do not change one without the other.

---

## File overview

| File | Role |
|------|------|
| `dataset.py` | Raw binary reader + `CustomDataset` + `build_dataset` |
| `SNN_func.py` | Network, encoder, predictor, trainer, loss, **basic** diagnostics |
| `generate_dataset.py` | Build & cache the dataset (`.pt`) | 
| `train_model.py` | Train the SNN, save weights (`.pth`) | 
| `print_predictions.py` | Evaluate, **deep** diagnostics, plots |

Basic diagnostics (loss curve, prediction-vs-target, residuals) live in
`SNN_func.py`. Deep diagnostics (profiling, bias/variance, CDF comparison,
population variance, per-neuron activity) live in `print_predictions.py` and are
all fed by a **single** forward pass over the test set.

---

## Stage 1 — `generate_dataset.py`

Reads the raw binary files, flattens the chosen target into a numeric vector,
applies optional energy/spatial filtering, applies `log10` to energy, and caches
everything into one `.pt` file.

```bash
python generate_dataset.py --data-dir RAW_DIR --out cached_dataset.pt --target Epos
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | *(required)* | Folder containing the raw sub-directories |
| `--out` | `cached_dataset.pt` | Output `.pt` filename |
| `--target` | `energy` | Target to regress (see below) |
| `--max-files` | `100` | Max files read per sub-directory |
| `--primary-only` | off | Read only the primary cubelets |
| `--e-max` | `None` | Keep only events with `E ≤ e-max` (MeV, applied before log10) |
| `--x-center` | `4.5` | X center of the spatial region (0–9) |
| `--y-center` | `4.5` | Y center of the spatial region (0–9) |
| `--r-min` | `0.0` | Inner radius of the region |
| `--r-max` | `15.0` | Outer radius of the region |
| `--z-min` | `0` | First Z layer (0–9) |
| `--z-max` | `9` | Last Z layer (0–9) |
| `--invert` | off | Keep what is **outside** the region instead |

### Accepted `--target` values

| Value | Columns produced |
|-------|------------------|
| `energy` | `log10(E)` |
| `centroid` | `x_c, y_c, z_c` |
| `dispersion` | `sigma_x, sigma_y, sigma_z` |
| `Epos` | `log10(E), x_c, y_c, z_c` |
| `Edsp` | `log10(E), sigma_x, sigma_y, sigma_z` |
| `"a,b,c"` | comma-separated list of base keys |

The radial/Z arguments implement the **radial sectioning** study: you can carve
the detector into regions (e.g. an inner cylinder vs. its complement with
`--invert`) and build a separate cache per region.

### Output

A single `.pt` file containing `samples`, `cubelets`, `targets`, and
`target_name`.

---

## Stage 2 — `train_model.py`

Trains the SNN on a cached dataset and saves the learned weights.

```bash
python train_model.py --cache cached_dataset.pt --epochs 50 --model-out snn_model.pth
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--cache` | *(required)* | Cached dataset `.pt` from stage 1 |
| `--epochs` | `5` | Number of training epochs |
| `--lr` | `1e-2` | Learning rate (Adam) |
| `--batch` | `32` | Batch size |
| `--model-out` | `snn_model.pth` | Output weights filename |

### Outputs

| File | Content |
|------|---------|
| `snn_model.pth` (or `--model-out`) | Trained network weights |
| `loss.png` | Training & validation loss vs. epoch (log scale) |
| `learned_threshold_exponents.txt` | Learned encoder threshold exponents (log10) |

The console also prints the encoder threshold exponents and the linear-scale
thresholds (min / mean / max). "Linear scale" here refers to the threshold
values `10^exponent`, not to linear energy.

---

## Stage 3 — `print_predictions.py`

Loads the cached dataset and the trained weights, runs **one** forward pass over
the test set, and produces all metrics and plots from that single pass.

```bash
# Basic evaluation (test metrics + basic plots + linear-energy view)
python print_predictions.py --cache cached_dataset.pt --model snn_model.pth

# Full evaluation including deep diagnostics
python print_predictions.py --cache cached_dataset.pt --model snn_model.pth --analyze
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--cache` | *(required)* | Cached dataset `.pt` from stage 1 |
| `--model` | *(required)* | Trained weights `.pth` from stage 2 |
| `--batch` | `32` | Batch size for evaluation |
| `--analyze` | off | Run the deep diagnostics block |

> **Note:** the variable names in `main()` are hard-coded to the four `Epos`
> targets (`log(E/MeV)`, `x_c`, `y_c`, `z_c`). Running evaluation on a cache
> with a different number of targets requires editing those lists accordingly.

### Console output

- Test loss and test relative error (%).
- A figures-of-merit table (Pearson r, R², RMSE, MAE, bias, residual variance)
  per target, printed both as plain text and as a ready-to-paste LaTeX table
  (only when `--analyze` is set).
- A one-line shape-check confirmation, e.g. `[shape check] OK  -  N=..., n_tasks=4, pop=20`.

### Plots generated

**Always produced:**

| File | What it shows |
|------|---------------|
| `pred_vs_target.png` | Basic prediction vs. true value, per target |
| `residuals.png` | Basic residual distributions, per target |
| `scatter_target_vs_predict_energy.png` | Linear-energy scatter (log10 undone) |
| `scatter_target_vs_predict_energy_v2.png` | Linear-energy 2D histogram + profile |

**Only with `--analyze`:**

| File | What it shows |
|------|---------------|
| `diagnostic_profile.png` | 2×2: 2D histogram + binned mean±std profile, per target |
| `diagnostic_bias_var.png` | 2×2: binned bias (median residual) and residual std, per target |
| `diagnostic_cdf_comparison.png` | 2×2: empirical (prediction) vs. expected (true) CDF, per target |
| `diagnostic_population_variance.png` | Intra-population spike variance vs. true energy |
| `diagnostic_neuron_activity.png` | Boxplot of total spikes per neuron in the ensemble |

All plots are saved at 300 dpi in the current working directory.

---

## Typical full run

```bash
# 1. Build the cache for the full detector, energy + centroid
python generate_dataset.py --data-dir RAW_DIR --out epos.pt --target Epos --primary-only

# 2. Train
python train_model.py --cache epos.pt --epochs 50 --model-out epos.pth

# 3. Evaluate with deep diagnostics
python print_predictions.py --cache epos.pt --model epos.pth --analyze
```

For the radial sectioning study, repeat stages 1–3 per region, e.g. an inner
region (`--r-max 3`) and its complement (`--r-min 3 --invert`), and compare the
figures-of-merit tables against the single global model.

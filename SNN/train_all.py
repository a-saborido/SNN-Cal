"""train_all.py

Trains 18 distinct SNNs (9 regression targets × 2 output modes: spike / membrane).

Targets
-------
  Single-output (population of 20 neurons → 1 value):
    1.  energy
    2.  centroid_x
    3.  centroid_y
    4.  centroid_z
    5.  dispersion_x
    6.  dispersion_y
    7.  dispersion_z
  Multi-output (4 populations of 20 neurons → 4 values):
    8.  energy + centroid  x,y,z
    9.  energy + dispersion x,y,z

Checkpoints are saved to ./checkpoints/<task_name>_<output_mode>.pt
with an intermediate save every SAVE_EVERY epochs.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import snntorch as snn
from snntorch import surrogate

import dataset as ds
import SNN_func as snnfn

# ── Hyperparameters ────────────────────────────────────────────────────────────
DATA_PATH  = "/lustre/ific.uv.es/ml/uovi123/snncalo/PhotonData/PrimaryOnly"
MAX_FILES  = 50          # .dat files per particle subdirectory
BATCH_SIZE = 64
NUM_EPOCHS = 10
SAVE_EVERY = 2           # save an intermediate checkpoint every N epochs
CKPT_DIR   = "./checkpoints"

POPULATION   = 20       # neurons per regression output
INPUT_SIZE   = ds.nSensors * 4   # 100 sensors × 4 multiplicity thresholds = 400


# ── Spike generation (static multi-threshold encoder) ─────────────────────────
def spikegen_multi(data, multiplicity=4):
    """Encode photon counts into spikes using 4 fixed thresholds (10², 10³, 10⁴, 10⁵)."""
    og_shape   = data.shape
    spike_data = torch.zeros(og_shape[1], og_shape[0], multiplicity * og_shape[2])
    for i in range(multiplicity):
        condition = data > np.power(10, i + 2)
        batch_idx, time_idx, sensor_idx = torch.nonzero(condition, as_tuple=True)
        spike_data[time_idx, batch_idx, multiplicity * sensor_idx + i] = 1
    return spike_data


# ── Prediction functions ───────────────────────────────────────────────────────
def predict_spikefreq(output):
    """Sum spikes over time, average over the population dimension.

    Works for both single-task (T, B, pop) → (B,)
    and multi-task after internal reshape (T, B, pop, tasks) → (B, tasks).
    """
    return output.sum(0).mean(1)


def predict_membrane(output):
    """Read the last-timestep membrane potential, averaged over population.

    (T, B, pop) → (B,)  or  (T, B, pop, tasks) → (B, tasks)
    """
    prediction = output[-1]       # (B, pop) or (B, pop, tasks)
    return prediction.mean(1)     # average over population


# ── Accuracy functions ─────────────────────────────────────────────────────────
def relative_error(prediction, targets):
    """Relative absolute error: |t - p| / |t|.  Only sensible when t ≠ 0."""
    return torch.abs(targets - prediction) / torch.abs(targets).clamp(min=1e-8)


def absolute_error(prediction, targets):
    """Absolute error: |t - p|."""
    return torch.abs(targets - prediction)


# ── Network builder ────────────────────────────────────────────────────────────
def make_net_desc(output_size: int, output_mode: str) -> dict:
    """Return a 3-layer SNN descriptor with learnable Leaky neurons.

    Args:
        output_size: total number of output neurons (POPULATION × n_tasks)
        output_mode: "spike" or "membrane"
    """
    leaky_params = dict(
        beta=1.0, learn_beta=True,
        threshold=1.0, learn_threshold=True,
        spike_grad=surrogate.atan(),
    )
    last_layer_params = leaky_params.copy()
    if output_mode == "membrane":
        last_layer_params.update(threshold=1e20, learn_threshold=False)
    neuron_params = {i: [snn.Leaky, leaky_params] for i in range(1, 4)}
    neuron_params[3] = [snn.Leaky, last_layer_params]
    return {
        "layers":       [INPUT_SIZE, 120, 120, output_size],
        "timesteps":    ds.timesteps,
        "output":       output_mode,
        "neuron_params": neuron_params,
    }


# ── Dataset helpers ────────────────────────────────────────────────────────────
class ComponentDataset(Dataset):
    """Wraps a multi-valued target dataset and returns a single scalar component."""

    def __init__(self, base: Dataset, idx: int):
        self.base = base
        self.idx  = idx

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        sample, target = self.base[i]
        return sample, target[self.idx]


class CombinedTargetDataset(Dataset):
    """Concatenates targets from two datasets; samples are taken from ds1.

    Intended for combining a scalar target (e.g. energy) from ds1 with a
    3-vector target (e.g. centroid) from ds2 into a 4-element target vector.
    Both datasets must have been built from the same files in the same order.
    """

    def __init__(self, ds1: Dataset, ds2: Dataset):
        if len(ds1) != len(ds2):
            raise ValueError("Both datasets must have the same number of samples.")
        self.ds1 = ds1
        self.ds2 = ds2

    def __len__(self):
        return len(self.ds1)

    def __getitem__(self, i):
        sample, t1 = self.ds1[i]
        _,     t2 = self.ds2[i]
        t1 = t1.unsqueeze(0) if t1.dim() == 0 else t1
        t2 = t2.unsqueeze(0) if t2.dim() == 0 else t2
        return sample, torch.cat([t1, t2])


# ── Load base datasets ─────────────────────────────────────────────────────────
print("Indexing / loading datasets …")

convert_to_log = lambda x:  (x[0], torch.log10(x[1]))
energy_data     = ds.build_dataset(DATA_PATH, MAX_FILES, lazy=True, primary_only=True, target="energy",
                                   transform=convert_to_log)
centroid_data   = ds.build_dataset(DATA_PATH, MAX_FILES, lazy=True, primary_only=True, target="centroid")
dispersion_data = ds.build_dataset(DATA_PATH, MAX_FILES, lazy=True, primary_only=True, target="dispersion")

energy_centroid_data   = CombinedTargetDataset(energy_data, centroid_data)    # (E, cx, cy, cz)
energy_dispersion_data = CombinedTargetDataset(energy_data, dispersion_data)  # (E, sX, sY, sZ)

print(f"  Total samples: {len(energy_data)}\n")


# ── Task table ─────────────────────────────────────────────────────────────────
# (label, base_dataset, component_idx_or_None, n_tasks, accuracy_fn, set_mse)
#
#   component_idx: int  → wrap base_dataset in ComponentDataset to pick one axis
#                  None → use base_dataset directly
#   n_tasks:    1 for scalar regression, 4 for (E, x, y, z) regression
#   accuracy_fn: relative_error for energy (always > 0), absolute_error otherwise
#   set_mse:    per-task loss flag — 1 = MSELoss, 0 = L1Loss
#               Rule: MSE for centroid x and y; L1 for everything else.
#               Multi-task order always follows (E, x, y, z).

TASKS = [
    # ── single-output tasks ──────────────────────────────── set_mse ──────────
    ("energy",          energy_data,            None, 1, relative_error, [0]),  # L1
    ("centroid_x",      centroid_data,          0,    1, absolute_error, [1]),  # MSE
    ("centroid_y",      centroid_data,          1,    1, absolute_error, [1]),  # MSE
    ("centroid_z",      centroid_data,          2,    1, absolute_error, [0]),  # L1
    ("dispersion_x",    dispersion_data,        0,    1, absolute_error, [0]),  # L1
    ("dispersion_y",    dispersion_data,        1,    1, absolute_error, [0]),  # L1
    ("dispersion_z",    dispersion_data,        2,    1, absolute_error, [0]),  # L1
    # ── multi-output tasks ─────────────────── (E,  cx,  cy,  cz/sZ) ─────────
    ("energy_centroid",   energy_centroid_data,   None, 4, absolute_error, [0, 1, 1, 0]),
    ("energy_dispersion", energy_dispersion_data, None, 4, absolute_error, [0, 0, 0, 0]),
]


# ── Training ───────────────────────────────────────────────────────────────────
os.makedirs(CKPT_DIR, exist_ok=True)

for task_name, base_dataset, comp_idx, n_tasks, acc_fn, set_mse in TASKS:

    # Wrap dataset if a single component is needed
    dataset = ComponentDataset(base_dataset, comp_idx) if comp_idx is not None else base_dataset

    train_loader, test_loader, val_loader = ds.build_loaders(
        dataset, split=(0.7, 0.15), batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    for output_mode in ("spike", "membrane"):
        label = f"{task_name}_{output_mode}"
        print(f"\n{'='*60}")
        print(f"  {label}  ({n_tasks} task(s), {output_mode} output)")
        print(f"{'='*60}")

        # ── build network ──────────────────────────────────────────────────────
        output_size = POPULATION * n_tasks
        net      = snnfn.Spiking_Net(make_net_desc(output_size, output_mode), spikegen_multi)
        pred_fn  = predict_spikefreq if output_mode == "spike" else predict_membrane
        predictor = snnfn.Predictor(pred_fn, acc_fn, population_sizes=POPULATION)

        # ── loss ───────────────────────────────────────────────────────────────
        # Single-task: use nn.MSELoss or nn.L1Loss directly on (B,) tensors.
        # Multi-task:  multi_MSELoss applies per-task MSE/L1 on (B, n_tasks).
        # set_mse drives the choice: 1 → MSE (centroid x/y), 0 → L1 (all else).
        if n_tasks == 1:
            loss_fn = nn.MSELoss() if set_mse[0] else nn.L1Loss()
        else:
            loss_fn = snnfn.multi_MSELoss(
                weights=torch.ones(n_tasks),
                set_mse=set_mse,
            )

        # ── optimizer ─────────────────────────────────────────────────────────
        optimizer = torch.optim.Adam(
            net.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-3
        )

        # ── trainer ───────────────────────────────────────────────────────────
        trainer = snnfn.Trainer(
            net, loss_fn, optimizer, predictor,
            train_loader, val_loader, test_loader,
            task="Regression",
        )

        ckpt_path = os.path.join(CKPT_DIR, f"{label}.pt")
        trainer.train(NUM_EPOCHS, checkpoint_path=ckpt_path, save_every=SAVE_EVERY)

print("\nAll tasks complete.")

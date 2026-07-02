"""train_job.py

Single-job entry point for HTCondor.  Each job trains exactly ONE of the 18
combinations (9 regression targets × 2 output modes: spike / membrane).

Usage
-----
    python train_job.py --task_idx <0-17>

Job index mapping (task_idx = 2*task_row + mode_col):
  mode_col: 0 → spike, 1 → membrane

  task_idx  task_name           output_mode
  --------  ------------------  -----------
   0         energy              spike
   1         energy              membrane
   2         centroid_x          spike
   3         centroid_x          membrane
   4         centroid_y          spike
   5         centroid_y          membrane
   6         centroid_z          spike
   7         centroid_z          membrane
   8         dispersion_x        spike
   9         dispersion_x        membrane
  10         dispersion_y        spike
  11         dispersion_y        membrane
  12         dispersion_z        spike
  13         dispersion_z        membrane
  14         energy_centroid     spike
  15         energy_centroid     membrane
  16         energy_dispersion   spike
  17         energy_dispersion   membrane

Checkpoints are saved to CKPT_DIR/<task_name>_<output_mode>.pt
with an intermediate save every SAVE_EVERY epochs.
"""

import argparse
import os
import sys

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
MAX_FILES  = 50
BATCH_SIZE = 64
NUM_EPOCHS = 10
SAVE_EVERY = 2
CKPT_DIR   = "./checkpoints"

POPULATION  = 20
INPUT_SIZE  = ds.nSensors * 4   # 100 sensors × 4 multiplicity thresholds = 400


# ── Task metadata table (no datasets loaded yet) ───────────────────────────────
#  (label, needs_energy, needs_centroid, needs_dispersion, comp_idx, n_tasks, acc_fn_name, set_mse)
#
#  acc_fn_name: "relative" → relative_error,  "absolute" → absolute_error
#  The actual dataset objects are built later, based on which task is selected.

_TASK_META = [
    # label                needs_E  needs_C  needs_D  comp  n  acc           set_mse
    ("energy",             True,    False,   False,   None, 1, "relative",   [0]),
    ("centroid_x",         False,   True,    False,   0,    1, "absolute",   [1]),
    ("centroid_y",         False,   True,    False,   1,    1, "absolute",   [1]),
    ("centroid_z",         False,   True,    False,   2,    1, "absolute",   [0]),
    ("dispersion_x",       False,   False,   True,    0,    1, "absolute",   [0]),
    ("dispersion_y",       False,   False,   True,    1,    1, "absolute",   [0]),
    ("dispersion_z",       False,   False,   True,    2,    1, "absolute",   [0]),
    ("energy_centroid",    True,    True,    False,   None, 4, "absolute",   [0, 1, 1, 0]),
    ("energy_dispersion",  True,    False,   True,    None, 4, "absolute",   [0, 0, 0, 0]),
]

OUTPUT_MODES = ["spike", "membrane"]

NUM_JOBS = len(_TASK_META) * len(OUTPUT_MODES)   # 9 tasks × 2 output modes

# ── Spike generation ────────────────────────────────────────────────────────────
def spikegen_multi(data, multiplicity=4):
    """Encode photon counts into spikes using 4 fixed thresholds (10², 10³, 10⁴, 10⁵)."""
    og_shape   = data.shape
    spike_data = torch.zeros(og_shape[1], og_shape[0], multiplicity * og_shape[2], device=data.device)
    for i in range(multiplicity):
        condition = data > np.power(10, i + 2)
        batch_idx, time_idx, sensor_idx = torch.nonzero(condition, as_tuple=True)
        spike_data[time_idx, batch_idx, multiplicity * sensor_idx + i] = 1
    return spike_data


# ── Prediction functions ────────────────────────────────────────────────────────
def predict_spikefreq(output):
    return output.sum(0).mean(1)


def predict_membrane(output):
    return output[-1].mean(1)


# ── Accuracy functions ──────────────────────────────────────────────────────────
def relative_error(prediction, targets):
    return torch.abs(targets - prediction) / torch.abs(targets).clamp(min=1e-8)


def absolute_error(prediction, targets):
    return torch.abs(targets - prediction)


# ── Network builder ─────────────────────────────────────────────────────────────
def make_net_desc(output_size: int, output_mode: str) -> dict:
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


# ── Dataset helpers ─────────────────────────────────────────────────────────────
class ComponentDataset(Dataset):
    def __init__(self, base: Dataset, idx: int):
        self.base = base
        self.idx  = idx

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        sample, target = self.base[i]
        return sample, target[self.idx]


class CombinedTargetDataset(Dataset):
    def __init__(self, ds1: Dataset, ds2: Dataset):
        if len(ds1) != len(ds2):
            raise ValueError("Both datasets must have the same number of samples.")
        self.ds1 = ds1
        self.ds2 = ds2

    def __len__(self):
        return len(self.ds1)

    def __getitem__(self, i):
        sample, t1 = self.ds1[i]
        _,     t2  = self.ds2[i]
        t1 = t1.unsqueeze(0) if t1.dim() == 0 else t1
        t2 = t2.unsqueeze(0) if t2.dim() == 0 else t2
        return sample, torch.cat([t1, t2])


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train a single SNN task.")
    parser.add_argument(
        "--task_idx", type=int, required=True,
        help=f"Job index in [0, {NUM_JOBS - 1}].  "
             "task = task_idx // 2,  output_mode = ['spike','membrane'][task_idx %% 2]",
    )
    args = parser.parse_args()

    if not (0 <= args.task_idx < NUM_JOBS):
        parser.error(f"--task_idx must be in [0, {NUM_JOBS - 1}], got {args.task_idx}")

    task_row  = args.task_idx // 2
    mode_col  = args.task_idx % 2
    output_mode = OUTPUT_MODES[mode_col]

    task_name, needs_E, needs_C, needs_D, comp_idx, n_tasks, acc_fn_name, set_mse = \
        _TASK_META[task_row]

    label = f"{task_name}_{output_mode}"
    print(f"\n{'='*60}")
    print(f"  Job {args.task_idx}: {label}  ({n_tasks} task(s), {output_mode} output)")
    print(f"{'='*60}\n")

    # ── Load only the datasets required for this task ──────────────────────────
    print("Indexing / loading dataset(s) …")
    convert_to_log = lambda x:  (x[0], torch.log10(x[1]))
    energy_data     = ds.build_dataset(DATA_PATH, MAX_FILES, lazy=True, primary_only=True,
                                       target="energy", transform=convert_to_log) if needs_E else None
    centroid_data   = ds.build_dataset(DATA_PATH, MAX_FILES, lazy=True, primary_only=True,
                                       target="centroid")   if needs_C else None
    dispersion_data = ds.build_dataset(DATA_PATH, MAX_FILES, lazy=True, primary_only=True,
                                       target="dispersion") if needs_D else None

    # Build the task dataset
    if task_name == "energy_centroid":
        dataset = CombinedTargetDataset(energy_data, centroid_data)
    elif task_name == "energy_dispersion":
        dataset = CombinedTargetDataset(energy_data, dispersion_data)
    else:
        base = energy_data or centroid_data or dispersion_data
        dataset = ComponentDataset(base, comp_idx) if comp_idx is not None else base

    print(f"  Total samples: {len(dataset)}\n")

    # ── Data loaders ───────────────────────────────────────────────────────────
    train_loader, test_loader, val_loader = ds.build_loaders(
        dataset, split=(0.7, 0.15), batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    # ── Network ────────────────────────────────────────────────────────────────
    output_size = POPULATION * n_tasks
    net         = snnfn.Spiking_Net(make_net_desc(output_size, output_mode), spikegen_multi)
    pred_fn     = predict_spikefreq if output_mode == "spike" else predict_membrane
    acc_fn      = relative_error if acc_fn_name == "relative" else absolute_error
    predictor   = snnfn.Predictor(pred_fn, acc_fn, population_sizes=POPULATION)

    # ── Loss ───────────────────────────────────────────────────────────────────
    if n_tasks == 1:
        loss_fn = nn.MSELoss() if set_mse[0] else nn.L1Loss()
    else:
        loss_fn = snnfn.multi_MSELoss(
            weights=torch.ones(n_tasks),
            set_mse=set_mse,
        )

    # ── Optimizer ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        net.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-3
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = snnfn.Trainer(
        net, loss_fn, optimizer, predictor,
        train_loader, val_loader, test_loader,
        task="Regression",
    )

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"{label}.pt")
    trainer.train(NUM_EPOCHS, checkpoint_path=ckpt_path, save_every=SAVE_EVERY)

    print(f"\nJob {args.task_idx} ({label}) complete.")


if __name__ == "__main__":
    main()

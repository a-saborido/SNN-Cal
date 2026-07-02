"""train_ann_job.py

Single-job entry point for HTCondor.  Each job trains exactly ONE of the 9
regression tasks using a plain ANN.

The ANN sums the raw photon-count input over the time dimension to produce a
[B, 100] feature vector, then applies two hidden layers of 120 units with ReLU
activation, and outputs one neuron per regression target.

Usage
-----
    python train_ann_job.py --task_idx <0-8>

  task_idx  task_name
  --------  ------------------
   0         energy
   1         centroid_x
   2         centroid_y
   3         centroid_z
   4         dispersion_x
   5         dispersion_y
   6         dispersion_z
   7         energy_centroid
   8         energy_dispersion

Checkpoints are saved to CKPT_DIR/ann_<task_name>.pt
with an intermediate save every SAVE_EVERY epochs.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import dataset as ds
import SNN_func as snnfn


# ── Hyperparameters ────────────────────────────────────────────────────────────
DATA_PATH  = "/lustre/ific.uv.es/ml/uovi123/snncalo/PhotonData/PrimaryOnly"
MAX_FILES  = 50
BATCH_SIZE = 64
NUM_EPOCHS = 10
SAVE_EVERY = 2
CKPT_DIR   = "./checkpoints"

ANN_INPUT_SIZE = ds.nSensors   # 100 sensors, summed over time


# ── Task metadata table ────────────────────────────────────────────────────────
_TASK_META = [
    # label                needs_E  needs_C  needs_D  comp  n  acc           set_mse
#    ("energy",             True,    False,   False,   None, 1, "relative",   [0]),
    ("centroid_x",         False,   True,    False,   0,    1, "absolute",   [1]),
    ("centroid_y",         False,   True,    False,   1,    1, "absolute",   [1]),
    ("centroid_z",         False,   True,    False,   2,    1, "absolute",   [0]),
    ("dispersion_x",       False,   False,   True,    0,    1, "absolute",   [0]),
    ("dispersion_y",       False,   False,   True,    1,    1, "absolute",   [0]),
    ("dispersion_z",       False,   False,   True,    2,    1, "absolute",   [0]),
    ("energy_centroid",    True,    True,    False,   None, 4, "absolute",   [0, 1, 1, 0]),
    ("energy_dispersion",  True,    False,   True,    None, 4, "absolute",   [0, 0, 0, 0]),
]

NUM_JOBS = len(_TASK_META)   # 9


# ── Network ────────────────────────────────────────────────────────────────────
class ANN(nn.Module):
    """Plain feed-forward network for calorimeter regression.

    Input: raw photon-count tensor of shape (B, T, nSensors).
    The time dimension is collapsed by summation before the dense layers.
    Output: (B, output_size).
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size,  hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        x = x.sum(dim=1).float()   # (B, T, nSensors) → (B, nSensors)
        return self.net(x)         # (B, output_size)


# ── Accuracy functions ─────────────────────────────────────────────────────────
def relative_error(prediction, targets):
    return torch.abs(targets - prediction) / torch.abs(targets).clamp(min=1e-8)


def absolute_error(prediction, targets):
    return torch.abs(targets - prediction)


# ── Dataset helpers ────────────────────────────────────────────────────────────
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


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train a single ANN task.")
    parser.add_argument(
        "--task_idx", type=int, required=True,
        help=f"Job index in [0, {NUM_JOBS - 1}], mapping directly to _TASK_META.",
    )
    args = parser.parse_args()

    if not (0 <= args.task_idx < NUM_JOBS):
        parser.error(f"--task_idx must be in [0, {NUM_JOBS - 1}], got {args.task_idx}")

    task_name, needs_E, needs_C, needs_D, comp_idx, n_tasks, acc_fn_name, set_mse = \
        _TASK_META[args.task_idx]

    label = f"{task_name}_ann"
    print(f"\n{'='*60}")
    print(f"  Job {args.task_idx}: {label}  ({n_tasks} task(s))")
    print(f"{'='*60}\n")

    # ── Load only the datasets required for this task ──────────────────────────
    print("Indexing / loading dataset(s) …")
    convert_to_log = lambda x: (x[0], torch.log10(x[1]))
    energy_data     = ds.build_dataset(DATA_PATH, MAX_FILES, lazy=True, primary_only=True,
                                       target="energy", transform=convert_to_log) if needs_E else None
    centroid_data   = ds.build_dataset(DATA_PATH, MAX_FILES, lazy=True, primary_only=True,
                                       target="centroid")   if needs_C else None
    dispersion_data = ds.build_dataset(DATA_PATH, MAX_FILES, lazy=True, primary_only=True,
                                       target="dispersion") if needs_D else None

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
    net = ANN(ANN_INPUT_SIZE, hidden_size=120, output_size=n_tasks)

    # pred_fn: the Predictor splits the output into per-task chunks of size 1,
    # each chunk is (B, 1) — squeeze to (B,) to match the target shape.
    pred_fn = lambda x: x.squeeze(-1)
    acc_fn  = relative_error if acc_fn_name == "relative" else absolute_error
    predictor = snnfn.Predictor(pred_fn, acc_fn, population_sizes=1)

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

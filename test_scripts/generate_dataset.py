#!/usr/bin/env python
"""
Reads raw binary files, builds a `CustomDataset`, flattens any composite
target into a numeric vector, and caches the result as a single .pt file.

Accepted --target values
------------------------
  energy            (single column)
  centroid          (x_c, y_c, z_c)
  dispersion        (sigma_x , sigma_y , sigma_z)
  Epos              = energy + centroid
  Edsp              = energy + dispersion
  "a,b,c"           comma-separated list of base keys
"""

import argparse, torch, numpy as np
from dataset import build_dataset

# ------------------------- CLI -------------------------
p = argparse.ArgumentParser()
p.add_argument("--data-dir", required=True,
               help="Folder containing the raw sub-directories")
p.add_argument("--out",      default="cached_dataset.pt",
               help="Output .pt filename")
p.add_argument("--target",   default="energy",
               help="Target(s) to regress - see docstring")
p.add_argument("--max-files", type=int, default=100)
args = p.parse_args()

# ------------------------- target alias / parsing -------------------------
alias = {
    "Epos": ["energy", "centroid"],       # logE, x_c, y_c, z_c
    "Edsp": ["energy", "dispersion"],     # logE, sigma_x, sigma_y, sigma_z
}

tgt_arg = args.target
if isinstance(tgt_arg, str) and "," in tgt_arg:               # "a,b,c"
    tgt_arg = [t.strip() for t in tgt_arg.split(",")]

if isinstance(tgt_arg, str) and tgt_arg in alias:             # Epos or Edsp
    tgt_arg = alias[tgt_arg]

# ------------------------- build dataset -------------------------
ds = build_dataset(args.data_dir,
                   max_files=args.max_files,
                   primary_only=False,                         # set to True if including only primary cubelets!
                   target=tgt_arg,
                   energy_threshold=10
                   )                       # tune this as needed

# ------------------------- flatten helper -------------------------
def _flatten(x):
    """Recursively flatten nested tuples / lists / ndarrays to 1-D list."""
    if isinstance(x, (list, tuple)):
        out = []
        for xi in x:
            out.extend(_flatten(xi))
        return out
    elif isinstance(x, np.ndarray):
        return x.astype(np.float32).ravel().tolist()
    else:                                   # scalar
        return [float(x)]

# ------------------------- stack samples and targets -------------------------
samples, targets = zip(*ds.data)            # lists of tensors
samples = torch.stack(samples)              # (N, T, 100) int32
targets = torch.stack([torch.tensor(_flatten(t), dtype=torch.float32)
                       for t in targets])   # (N, n_targets) float32

if "energy" in tgt_arg or (isinstance(tgt_arg, list) and "energy" in tgt_arg):
    # energy is the *first* column after flattening
    targets[:, 0] = torch.log10(targets[:, 0])          # log10(E/MeV)


# ------------------------- save -------------------------
torch.save({"samples": samples,
            "targets": targets,
            "target_name": tgt_arg},
           args.out)
print(f"Cached {len(samples):,} events in {args.out}")

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

# --- SEGMENTATION ARGUMENTS ---
p.add_argument("--x-center", type=float, default=4.5, help="X center of the region (0-9)")
p.add_argument("--y-center", type=float, default=4.5, help="Y center of the region (0-9)")
p.add_argument("--r-min", type=float, default=0.0, help="Inner (minimum) radius")
p.add_argument("--r-max", type=float, default=15.0, help="Outer (maximum) radius")
p.add_argument("--z-min", type=int, default=0, help="Starting Z layer (0-9)")
p.add_argument("--z-max", type=int, default=9, help="Ending Z layer (0-9)")
p.add_argument("--invert", action="store_true", help="If set, keep what is OUTSIDE the region")

p.add_argument("--primary-only", action="store_true", help="If set, read only the primary cubelets")

# ----------------------------------------------------

p.add_argument("--e-max", type=float, default=None, help="Maximum allowed energy (MeV)")
# ----------------------------------------------------
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
                   primary_only=args.primary_only,                         
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

# ------------------------- stack samples, cubelets and targets -------------------------
samples, cubelets, targets = zip(*ds.data)
samples = torch.stack(samples)   # (N, T, 100)

cubelets = torch.stack([
    c if torch.is_tensor(c) else torch.tensor(c, dtype=torch.long)
    for c in cubelets
]).long()                        # (N,)

targets = torch.stack([
    torch.tensor(_flatten(t), dtype=torch.float32)
    for t in targets
])                               # (N, n_targets)


# ------------------------- Maximum energy filter -------------------------
# Applied while targets[:, 0] is still in linear MeV (before log10).
if args.e_max is not None:
    mask_e = targets[:, 0] <= args.e_max

    # Combine with the spatial mask if it already exists
    if 'mask' in locals():
        mask = mask & mask_e
    else:
        mask = mask_e

    samples = samples[mask]
    cubelets = cubelets[mask]
    targets = targets[mask]
    print(f"Filter E <= {args.e_max} MeV applied. Events: {len(samples):,}")

# ------------------------- log10 conversion (always applied to energy) -------------------------
# We always train/evaluate on log10(E/MeV). The linear-energy view is recovered
# downstream (in print_predictions.py) by undoing the log10.
if "energy" in tgt_arg or (isinstance(tgt_arg, list) and "energy" in tgt_arg):
    targets[:, 0] = torch.log10(targets[:, 0])          # log10(E/MeV)



# ------------------------- Spatial filtering (segmentation) -------------------------
# With 1000 cubelets on a 10x10x10 grid, extract (Z, Y, X) from the cubelet ID
z_coord = cubelets // 100
y_coord = (cubelets % 100) // 10
x_coord = cubelets % 10

# Distance (radius) of each event with respect to the defined XY center
radios = torch.sqrt((x_coord - args.x_center)**2 + (y_coord - args.y_center)**2)

# Boolean mask combining radius and depth (Z) conditions
mask = (radios >= args.r_min) & (radios <= args.r_max) & (z_coord >= args.z_min) & (z_coord <= args.z_max)

# Invert the mask if requested (to take the rest of the detector)
if args.invert:
    mask = ~mask

samples = samples[mask]
cubelets = cubelets[mask]
targets = targets[mask]

# ------------------------- save -------------------------
torch.save({"samples": samples,
            "cubelets": cubelets,
            "targets": targets,
            "target_name": tgt_arg},
           args.out)
print(f"Cached {len(samples):,} events in {args.out}")

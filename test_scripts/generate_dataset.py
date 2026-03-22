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
import math
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


p.add_argument("--e-min", type=float, default=0.0,
               help="Maximum energy of the cubelet")
p.add_argument("--e-max", type=float, default=float('inf'),
               help="Minimum energy of the cubelet")

#the idea will be to create one dataset for each region
p.add_argument("--x-center", type=float, default=4.5, help="Centro en la coord X")
p.add_argument("--y-center", type=float, default=4.5, help="Centro en la coord Y")
p.add_argument("--r-min", type=float, default=0.0, help="Radio interior en el plano XY (0 para círculo macizo)")
p.add_argument("--r-max", type=float, default=15.0, help="Radio exterior en el plano XY")
p.add_argument("--z-min", type=int, default=0, help="Min Z (0-9)")
p.add_argument("--z-max", type=int, default=9, help="Max Z (0-9)")
p.add_argument("--invert", action="store_true", help="Coge el complementario")
#!!!!!!!!!!

args = p.parse_args()

valid_ids = []
for z in range(10): # Iteramos siempre por todo el detector (0 a 9)
    for y in range(10):
        for x in range(10):
            radio = math.sqrt((x - args.x_center)**2 + (y - args.y_center)**2)
            
            # Comprobamos si el cubelet está dentro
            dentro_r = (args.r_min <= radio <= args.r_max)
            dentro_z = (args.z_min <= z <= args.z_max)
            en_zona = dentro_r and dentro_z
            
            
            if args.invert:
                if not en_zona: # Si está FUERA de la zona, lo guardamos
                    valid_ids.append((z * 100) + (y * 10) + x)
            else:
                if en_zona:     # Si está DENTRO de la zona, lo guardamos
                    valid_ids.append((z * 100) + (y * 10) + x)

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
                   primary_only=False,                         
                   target=tgt_arg,
                   energy_threshold=None)       # ¡QUITAMOS el valid_cubelets de aquí!

# --- AÑADIMOS EL FILTRO GEOMÉTRICO AQUÍ ---
valid_ids_set = set(valid_ids)

def geometry_filter(item):
    sample, cublet_id, target_val = item
    return int(cublet_id) in valid_ids_set

ds.clean(geometry_filter)
# -------------------------------------------

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
    else:                                       # scalar
        return [float(x)]

# ------------------------- stack samples and targets -------------------------
# Tu código ya estaba genial aquí, lo mantenemos intacto:
samples, cubelets, targets = zip(*ds.data)
samples = torch.stack(samples)              # (N, T, 100) int32

cubelets = torch.stack([
    c if torch.is_tensor(c) else torch.tensor(c, dtype=torch.long)
    for c in cubelets
]).long()   

targets = torch.stack([torch.tensor(_flatten(t), dtype=torch.float32)
                       for t in targets])   # (N, n_targets) float32

if "energy" in tgt_arg or (isinstance(tgt_arg, list) and "energy" in tgt_arg):
    # energy is the *first* column after flattening
    targets[:, 0] = torch.log10(targets[:, 0])          # log10(E/MeV)

#------------------------- energías filtrradas

if args.e_min > 0:
    # Convertimos el umbral a escala logarítmica para que coincida con targets
    e_min_log = math.log10(args.e_min)
    
    # Creamos la máscara: True para los eventos que superan el mínimo
    mask = targets[:, 0] >= e_min_log
    
    # Aplicamos el filtro a todos los tensores a la vez
    samples = samples[mask]
    cubelets = cubelets[mask]
    targets = targets[mask]



# ------------------------- save -------------------------
torch.save({"samples": samples,
            "cubelet_ids": cubelets,  # Le llamamos cubelet_ids para que train_model.py lo encuentre
            "targets": targets,
            "target_name": tgt_arg},
           args.out)
print(f"Cached {len(samples):,} events in {args.out}")
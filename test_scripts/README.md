
## Run training and predicitons on the primary cubelet:

IMPORTANT: set `primary_only=True` in generate_dataset.py

`python generate_dataset.py --data-dir ../Data/PrimaryOnly/Uniform --out PrimaryOnlyUniform_Epos.pt --target Epos`  # or "energy,centroid"  or  Edsp ...

`python train_model.py --cache PrimaryOnlyUniform_Epos.pt --epochs 5 --lr 1e-2 --model-out snn_PrimaryTrained_Epos.pth`

`python print_predictions.py --cache PrimaryOnlyUniform_Epos.pt --model snn_PrimaryTrained_Epos.pth`

## Run training in primary cubelet and predict in all:

IMPORTANT: set `primary_only=False` in generate_dataset.py

`python generate_dataset.py --data-dir ../Data/All_small --out All_small_Epos.pt --target Epos`

Assuming we already trained a model called `snn_PrimaryTrained_Epos`with primary only data, we can test it on all cubelets data:

`python print_predictions.py --cache All_small_Epos.pt --model snn_PrimaryTrained_Epos.pth`

# --------------------------------------------------------------------------------

###  Dataset 1 (centre)
`python generate_dataset.py --data-dir ../Data/All --out Dataset_1.pt --target Epos --x-center 4.5 --y-center 4.5 --r-min 0.0 --r-max 1.5 --z-min 0 --z-max 5`

###  Dataset 2 (corona)

`python generate_dataset.py --data-dir ../Data/All --out Dataset_2.pt --target Epos --x-center 4.5 --y-center 4.5 --r-min 1.5 --r-max 3.5 --z-min 0 --z-max 5`

###  Dataset 3 (halo)

`python generate_dataset.py --data-dir ../Data/All --out Dataset_3.pt --target Epos --x-center 4.5 --y-center 4.5 --r-min 0.0 --r-max 3.5 --z-min 0 --z-max 5 --invert`

# Training

`python train_model.py --cache Dataset_1.pt --epochs 15 --lr 1e-2 --model-out Model_1.pth`

# Predictions

`python print_predictions.py --cache Dataset_1.pt --model Model_1.pth`

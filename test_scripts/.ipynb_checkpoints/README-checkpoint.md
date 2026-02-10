python generate_dataset.py --data-dir ../Data/PrimaryOnly/Uniform --out PrimaryOnlyUniform_Epos.pt --target Epos  # or "energy,centroid"  or  Edsp ...

python train_model.py --cache PrimaryOnlyUniform_Epos.pt --epochs 5 --lr 1e-2 --model-out snn_model_Epos.pth

python print_predictions.py --cache PrimaryOnlyUniform_Epos.pt --model snn_model_Epos.pth

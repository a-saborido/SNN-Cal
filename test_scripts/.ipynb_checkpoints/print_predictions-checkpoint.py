#!/usr/bin/env python
"""
Prediction and evaluation using a trained SNN model
"""
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import CustomDataset
from SNN_func import Spiking_Net, Predictor, Trainer, multi_MSELoss
import snntorch as snn
from snntorch import surrogate
import numpy as np
import matplotlib.pyplot as plt

# ------------------------- Helper functions -------------------------

def spikegen_multi(data, multiplicity=4):
    og_shape = data.shape
    spike_data = torch.zeros(og_shape[1], og_shape[0], multiplicity * og_shape[2])
    for i in range(multiplicity):
        condition = data > np.power(10, i+2)
        batch_idx, time_idx, sensor_idx = torch.nonzero(condition, as_tuple=True)
        spike_data[time_idx, batch_idx, multiplicity * sensor_idx + i] = 1
    return spike_data


def predict_spikefreq(output):
    # sum spikes across time and average over the population
    return output.sum(0).mean(1)


def distance(prediction, targets, absolute: bool = True, relative: bool = True,
             transform: callable = lambda *args, **kwargs: args[0]):
    p, t = transform(prediction), transform(targets)
    accuracy = t - p
    if absolute:
        accuracy = torch.abs(accuracy)
    if relative:
        accuracy /= t
    return accuracy


def make_net_desc(n_tasks: int, pop: int = 20) -> dict:
    COMMON_NEURON = {"beta": 0.5, "learn_beta": True,
                     "threshold": 1.0, "learn_threshold": True,
                     "spike_grad": surrogate.atan()}
    return dict(
        layers=[400, 120, 120, pop * n_tasks],
        timesteps=100,
        output="spike",
        model=snn.Leaky,
        neuron_params=[{}, COMMON_NEURON, COMMON_NEURON, COMMON_NEURON],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="Path to cached dataset .pt file")
    p.add_argument("--model", required=True, help="Path to trained model .pth file")
    p.add_argument("--batch", type=int, default=32, help="Batch size for evaluation")
    p.add_argument("--lr", type=float, default=1e-2, help="Learning rate (unused placeholder)")
    args = p.parse_args()

    # Load cached data
    data_file = torch.load(args.cache, map_location="cpu")
    samples, targets = data_file["samples"], data_file["targets"]
    ds = CustomDataset(filelist=[], primary_only=True, target=data_file["target_name"])
    ds.data = list(zip(samples, targets))

    # Split dataset (70% train, 15% val, 15% test)
    total = len(ds)
    tr_len = int(0.70 * total)
    va_len = int(0.15 * total)
    te_len = total - tr_len - va_len
    tr_ds, va_ds, te_ds = random_split(ds, [tr_len, va_len, te_len])

    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch)
    te_loader = DataLoader(te_ds, batch_size=args.batch)

    # Build and load network
    n_tasks = targets.shape[1] if targets.ndim > 1 else 1
    net_desc = make_net_desc(n_tasks)
    net = Spiking_Net(net_desc, lambda x: spikegen_multi(x, 4))
    net.load_state_dict(torch.load(args.model, map_location="cpu"))

    # Predictor, loss, optimizer (optimizer unused here!)
    predictor = Predictor(predict_spikefreq, distance, population_sizes=20)
    loss_fn = multi_MSELoss(weights=torch.tensor([1] * n_tasks))
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # Trainer setup
    train_Epos_spk = Trainer(net, loss_fn, optimizer, predictor,
                             tr_loader, va_loader, te_loader,
                             task="Regression")

    # Evaluate with different accuracy metrics and show results
    train_Epos_spk.predict.accuracy_fn = lambda p, t: distance(p, t, absolute=True, relative=True)
    train_Epos_spk.test("test")
    train_Epos_spk.predict.accuracy_fn = lambda p, t: distance(p, t, absolute=False, relative=False)
    #train_Epos_spk.show_results(nbins=50, title=["log(E/MeV)", r"$x_c$", r"$y_c$", r"$z_c$"])
    
    # Not using show_results(). It avoids an empty plot of the loss function evolution
    print(f"Test loss: {train_Epos_spk.loss_hist['test'][train_Epos_spk.current_epoch]}")
    print(f"Test relative error: {train_Epos_spk.acc_hist['test'][train_Epos_spk.current_epoch] * 100}%")
    train_Epos_spk.plot_pred_vs_target(
    title=["log(E/MeV)", r"$x_c$", r"$y_c$", r"$z_c$"], nbins=50)
    train_Epos_spk.plot_residuals(
    title=["log(E/MeV)", r"$x_c$", r"$y_c$", r"$z_c$"], nbins=50)

    plt.show()
if __name__ == "__main__":
    main()

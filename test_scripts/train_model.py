#!/usr/bin/env python
"""
Training
"""

import argparse, torch, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset  import CustomDataset
from SNN_func import Spiking_Net, Predictor, Trainer, multi_MSELoss, CubeletOrderedThresholdSpikeGenMulti
import snntorch as snn
from snntorch import surrogate

from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

'''
def spikegen_multi(data, cubelet_id=None, multiplicity=4):
    B, T, S = data.shape
    spike_data = torch.zeros(
        T, B, multiplicity * S,
        device=data.device,
        dtype=data.dtype
    )

    for i in range(multiplicity):
        threshold = 10.0 ** (i + 2)
        condition = data > threshold
        batch_idx, time_idx, sensor_idx = torch.nonzero(condition, as_tuple=True)
        spike_data[time_idx, batch_idx, multiplicity * sensor_idx + i] = 1.0

    return spike_data
'''

def predict_spikefreq(output):
    prediction = output.sum(0).mean(1) # sum spikes across time and average over the population
    return prediction

def distance(prediction, targets, absolute: bool = True, relative: bool = True,
             transform: Callable = lambda *args, **kwargs: args[0]):
    p, t = transform(prediction), transform(targets)
    accuracy = t - p
    if absolute:
        accuracy = torch.abs(accuracy)
    if relative:
        accuracy /= t
    return accuracy

# --------------------------------------------------
POP_SIZE = 20     # neurons per regression target
# --------------------------------------------------


# ------------------------- helper to build net description -------------------------
COMMON_NEURON = {"beta":0.5, "learn_beta":True,
                 "threshold":1.0, "learn_threshold":True,
                 "spike_grad": surrogate.atan()}

def make_net_desc(n_tasks: int, pop: int = POP_SIZE) -> dict:
    return dict(
        layers        = [400, 120, 120, pop * n_tasks],
        timesteps     = 100,
        output        = "spike",
        model         = snn.Leaky,
        neuron_params = [{},
                         COMMON_NEURON, COMMON_NEURON, COMMON_NEURON],
    )

# ------------------------- main training routine -------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache",     required=True)
    p.add_argument("--epochs",    type=int,   default=5)
    p.add_argument("--lr",        type=float, default=1e-2)
    p.add_argument("--batch",     type=int,   default=32)
    p.add_argument("--model-out", default="snn_model.pth")
    args = p.parse_args()

    # load data
    data_file = torch.load(args.cache, map_location="cpu")
    samples = data_file["samples"]
    cubelets = data_file["cubelets"]
    targets = data_file["targets"]

    ds = CustomDataset(filelist=[], primary_only=True,
                       target=data_file["target_name"])
    ds.data = list(zip(samples, cubelets, targets))

    # Fix seed for consistency with print_predictions.py
    seed = 42
    g = torch.Generator().manual_seed(seed)

    tr_len = int(0.70*len(ds))
    va_len = int(0.15*len(ds))
    te_len = len(ds) - tr_len - va_len
    tr_ds, va_ds, te_ds = random_split(ds, [tr_len, va_len, te_len], generator=g)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch)
    te_loader = DataLoader(te_ds, batch_size=args.batch)

    # ------------------------- network / loss / predictor -------------------------
    n_tasks       = targets.shape[1] if targets.ndim>1 else 1
    net_desc      = make_net_desc(n_tasks)
    #net_Epos_spk  = Spiking_Net(net_desc,spikegen_multi)
    encoder = CubeletOrderedThresholdSpikeGenMulti(
        n_cubelets=1000,
        multiplicity=4,
        alpha=5.0
    )
    net_Epos_spk = Spiking_Net(net_desc, encoder)

    # predictor
    Pred_Epos_spk = Predictor(predict_spikefreq,
                              distance,
                              population_sizes=POP_SIZE)

    # loss
    loss_Epos = multi_MSELoss(weights=torch.tensor([1]*n_tasks))

    # optimiser + scheduler
    opt_Epos_spk  = optim.Adam(net_Epos_spk.parameters(),
                               lr=args.lr,
                               betas=(0.9, 0.999),
                               weight_decay=1e-3)
    sche_Epos_spk = optim.lr_scheduler.ExponentialLR(opt_Epos_spk, gamma=0.7)

    # trainer
    train_Epos_spk = Trainer(net_Epos_spk,
                             loss_Epos,
                             opt_Epos_spk,
                             Pred_Epos_spk,
                             tr_loader, va_loader, te_loader,
                             task="Regression")

    train_Epos_spk.train(args.epochs)

    # plot loss function
    train_Epos_spk.plot_loss(validation=True, logscale=True)
    plt.savefig("loss.png", dpi=300, bbox_inches="tight")

    # save
    torch.save(net_Epos_spk.state_dict(), args.model_out)
    print(f"Trained model saved to {args.model_out}")

    # after training, print the learned thresholds of the encoder
    th_exp = net_Epos_spk.spikegen_fn.threshold_exponents().detach().cpu().numpy()
    th = net_Epos_spk.spikegen_fn.thresholds().detach().cpu().numpy()

    print("Threshold exponents (log10) =")
    print(th_exp)
    '''
    c= np.array([])
    for i in th_exp:
        c = np.append(c, i)
    print(np.sort(np.unique(c)))
    '''
    print("Threshold exponents shape:", th_exp.shape)
    print("Threshold exponents min/mean/max:", th_exp.min(), th_exp.mean(), th_exp.max())

    print("\nThresholds (linear scale) =")
    print(th)
    print("Thresholds shape:", th.shape)
    print("Thresholds min/mean/max:", th.min(), th.mean(), th.max())

    # save the learned threshold exponents of the encoder
    out_file = "learned_threshold_exponents.txt"
    np.savetxt(out_file, th_exp, fmt="%.8f", delimiter="\t")


if __name__ == "__main__":
    main()

import math
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchmetrics.classification import MulticlassConfusionMatrix
from tqdm import tqdm

from typing import Callable, Union
from collections.abc import Iterable
from itertools import accumulate


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.set_default_dtype(torch.float32)

###############################################################################
#                                                                             #
#   SPIKE GENERATION                                                          #
#   HardThresholdArctanSTE  –  custom autograd function (surrogate gradient) #
#   SpikeGenSTE              –  learnable multi-threshold spike encoder       #
#                                                                             #
###############################################################################

class HardThresholdArctanSTE(Function):
    """Hard threshold in the forward pass; arctan surrogate gradient backward."""

    @staticmethod
    def forward(ctx, input, threshold, k):
        ctx.save_for_backward(input, threshold, k)
        return (input > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, threshold, k = ctx.saved_tensors

        # surrogate: d/dx [0.5 + atan(k(x-th))/π]  →  k / (π(1 + (k(x-th))²))
        diff = k * (input - threshold)
        surrogate_grad = k / (torch.pi * (1 + diff * diff))

        grad_input     = grad_output * surrogate_grad
        grad_threshold = -grad_output * surrogate_grad
        grad_k         = None   # k is treated as fixed; implement if learnable

        return grad_input, grad_threshold, grad_k


class SpikeGenSTE(nn.Module):
    """Encodes a real-valued input into spikes using learnable thresholds.

    Each input value is compared against `multiplicity` thresholds.  The
    thresholds and the steepness parameter k are learnable via the arctan
    surrogate gradient.
    """

    def __init__(self, multiplicity: int = 4):
        super().__init__()
        self.multiplicity = multiplicity

        init_thresholds = torch.tensor(
            [10 ** (i + 2) for i in range(multiplicity)], dtype=torch.float32
        )
        self.thresholds = nn.Parameter(init_thresholds)
        self.k = nn.Parameter(torch.tensor(5.0))

    def forward(self, data):
        """
        Args:
            data: (B, T, S)
        Returns:
            spikes: (T, B, S * multiplicity)
        """
        B, T, S = data.shape

        th = self.thresholds.view(1, 1, 1, self.multiplicity).expand(B, T, S, self.multiplicity)
        x  = data.unsqueeze(-1).expand_as(th)

        spikes = HardThresholdArctanSTE.apply(x, th, self.k)
        spikes = spikes.reshape(B, T, S * self.multiplicity)
        return spikes.permute(1, 0, 2)   # (T, B, S*multiplicity)


###############################################################################
#                                                                             #
#   NETWORK ARCHITECTURE                                                      #
#   Spiking_Net  –  fully-connected SNN with variable depth and neuron model  #
#                                                                             #
###############################################################################

class Spiking_Net(nn.Module):
    """Fully-connected SNN with variable neuron model and number of layers.

    Args:
        net_desc (dict): Must contain:
            - "layers"        : list of layer widths (input → output)
            - "timesteps"     : number of simulation timesteps
            - "output"        : "spike" or "membrane"
            - "neuron_params" : per-layer neuron parameters
            - "model"         : (optional) single neuron class used for every layer
        spikegen_fn: callable that converts raw input to spike trains (T, B, features)
    """

    def __init__(self, net_desc: dict, spikegen_fn: Callable):
        super().__init__()

        self.n_neurons = net_desc["layers"]
        self.timesteps = net_desc["timesteps"]
        self.output    = net_desc["output"]
        self.spikegen_fn = spikegen_fn

        modules = []
        for i_layer in range(1, len(self.n_neurons)):
            modules.append(
                nn.Linear(self.n_neurons[i_layer - 1], self.n_neurons[i_layer])
            )
            if "model" in net_desc:
                modules.append(net_desc["model"](**net_desc["neuron_params"][i_layer]))
            else:
                modules.append(
                    net_desc["neuron_params"][i_layer][0](**net_desc["neuron_params"][i_layer][1])
                )
        self.network = nn.Sequential(*modules)

    def forward(self, data):
        """Forward pass across all timesteps.

        Returns:
            Tensor of shape (T, B, n_out) — spikes or membrane potentials of
            the last layer, depending on self.output.
        """
        x = self.spikegen_fn(data)

        # Initialise membrane potentials for every neuron layer (odd indices)
        mem = []
        for i, module in enumerate(self.network):
            if i % 2 == 1:
                res = module.reset_mem()
                mem.append(list(res) if isinstance(res, tuple) else [res])

        n_layers = len(self.network) // 2
        spk_rec = []
        mem_rec = []

        spk = None
        for step in range(self.timesteps):
            for i_layer in range(n_layers):
                cur = self.network[2 * i_layer](x[step] if i_layer == 0 else spk)
                spk, *(mem[i_layer]) = self.network[2 * i_layer + 1](cur, *(mem[i_layer]))

            # Record only the final layer output
            spk_rec.append(spk)
            mem_rec.append(mem[-1][-1])

        if self.output == "spike":
            return torch.stack(spk_rec, dim=0)
        elif self.output == "membrane":
            return torch.stack(mem_rec, dim=0)


###############################################################################
#                                                                             #
#   PREDICTION                                                                #
#   Predictor  –  wraps a prediction function and an accuracy metric,        #
#                 with support for single- and multi-task outputs             #
#                                                                             #
###############################################################################

class Predictor:
    """Applies a prediction function and accuracy metric to network output.

    Args:
        prediction_fn: maps raw network output to predictions
        accuracy_fn:   computes per-sample accuracy / error
        population_sizes: int (equal-sized populations) or list of ints
                          (variable-sized populations) for multi-task outputs.
                          Use -1 for single-task.
    """

    def __init__(
        self,
        prediction_fn: Callable,
        accuracy_fn: Callable,
        population_sizes: Union[int, Iterable[int]] = -1,
    ):
        self.prediction_fn = prediction_fn
        self.accuracy_fn   = accuracy_fn

        if isinstance(population_sizes, int):
            self.population_sizes = population_sizes
            return
        if isinstance(population_sizes, Iterable):
            if isinstance(population_sizes, (np.ndarray, torch.Tensor)):
                population_sizes = population_sizes.tolist()
            if all(isinstance(x, int) for x in population_sizes):
                self.population_sizes = population_sizes
                return
        raise TypeError("population_sizes must be an int or an iterable of int.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_singletask(self, output, targets, reduction: str = "mean"):
        prediction = self.prediction_fn(output)
        accuracy   = self.accuracy_fn(prediction, targets)
        if reduction == "mean":
            return prediction, torch.mean(accuracy, 0)
        elif reduction == "sum":
            return prediction, torch.sum(accuracy, 0)
        return prediction, accuracy

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, output, targets, reduction: str = "mean"):
        # output shape: (T, B, n_out)   targets shape: (B,) or (B, n_tasks)

        # Single-task shortcut
        if len(targets.shape) == 1:
            return self._predict_singletask(output, targets, reduction)

        n_tasks = targets.shape[-1]

        # Multi-task: variable population sizes (list/tuple of ints)
        if isinstance(self.population_sizes, (list, tuple)):
            if sum(self.population_sizes) != output.shape[-1]:
                raise ValueError(
                    f"Sum of population sizes ({sum(self.population_sizes)}) must equal "
                    f"last layer size ({output.shape[-1]})."
                )
            if len(self.population_sizes) != n_tasks:
                raise ValueError(
                    f"Number of populations ({len(self.population_sizes)}) must equal "
                    f"number of tasks ({n_tasks})."
                )

            prediction = torch.zeros(size=targets.shape, device=targets.device)
            accuracy   = (
                torch.zeros(size=targets.shape, device=targets.device)
                if reduction == "none"
                else torch.zeros(size=(n_tasks,), device=targets.device)
            )
            chunks = list(accumulate([0] + list(self.population_sizes)))
            for i in range(n_tasks):
                chunk_out = output[..., chunks[i]:chunks[i + 1]]   # (T, B, pop_size_i)
                if reduction == "none":
                    prediction[:, i], accuracy[:, i] = \
                        self._predict_singletask(chunk_out, targets[:, i], reduction)
                else:
                    prediction[:, i], accuracy[i] = \
                        self._predict_singletask(chunk_out, targets[:, i], reduction)
            return prediction, accuracy

        # Multi-task: equal population sizes (single int)
        pop_size = self.population_sizes
        if pop_size * n_tasks != output.shape[-1]:
            raise ValueError(
                f"Population size ({pop_size}) × n_tasks ({n_tasks}) must equal "
                f"last layer size ({output.shape[-1]})."
            )

        prediction = torch.zeros(size=targets.shape, device=targets.device)
        accuracy   = (
            torch.zeros(size=targets.shape, device=targets.device)
            if reduction == "none"
            else torch.zeros(size=(n_tasks,), device=targets.device)
        )
        for i in range(n_tasks):
            chunk_out = output[..., i * pop_size:(i + 1) * pop_size]   # (T, B, pop_size)
            if reduction == "none":
                prediction[:, i], accuracy[:, i] = \
                    self._predict_singletask(chunk_out, targets[:, i], reduction)
            else:
                prediction[:, i], accuracy[i] = \
                    self._predict_singletask(chunk_out, targets[:, i], reduction)
        return prediction, accuracy


###############################################################################
#                                                                             #
#   LOSS FUNCTIONS                                                            #
#   multi_MSELoss  –  per-task combination of MSE and L1 losses              #
#                                                                             #
###############################################################################

class multi_MSELoss(nn.Module):
    """Weighted combination of per-task MSE / L1 losses.

    Args:
        reduction:  passed to F.mse_loss / F.l1_loss
        weights:    per-task scalar weights
        set_mse:    list of booleans – True → MSE, False → L1 for each task
    """

    def __init__(
        self,
        reduction: str = "mean",
        weights: torch.Tensor = torch.ones(1),
        set_mse: list = [0, 1, 1, 0],
    ):
        super().__init__()
        self.reduction = reduction
        self.weights   = weights
        self.func = [F.mse_loss if flag else F.l1_loss for flag in set_mse]

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(target.shape) < 2:
            target = target.unsqueeze(-1)

        losses = torch.zeros(target.shape[-1], device=input.device)
        for i in range(target.shape[-1]):
            losses[i] = self.func[i](input[:, i], target[:, i], reduction=self.reduction)

        return (self.weights.to(input.device) * losses).sum()


###############################################################################
#                                                                             #
#   TRAINING                                                                  #
#   Trainer  –  training loop, evaluation, checkpointing, and plotting       #
#                                                                             #
###############################################################################

class Trainer:
    """Handles the full training lifecycle of a spiking network.

    Args:
        net:           the network to train
        loss_fn:       loss function
        optimizer:     PyTorch optimizer
        predict:       Predictor instance
        train_dataset: DataLoader for training
        val_dataset:   DataLoader for validation
        test_dataset:  DataLoader for testing
        task:          "Regression" or "Classification"
    """

    def __init__(
        self,
        net,
        loss_fn,
        optimizer,
        predict,
        train_dataset,
        val_dataset,
        test_dataset,
        task: str = "Regression",
    ):
        self.net       = net
        self.loss_fn   = loss_fn
        self.optimizer = optimizer
        self.predict   = predict
        self.task      = task
        self.datasets  = {
            "train":      train_dataset,
            "validation": val_dataset,
            "test":       test_dataset,
        }

        self.current_epoch = 0
        self.loss_hist = {"train": {}, "validation": {}, "test": {}}
        self.acc_hist = {"validation": {}, "test": {}}
        self.par_hist = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                self.par_hist[name] = []

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save the full training state to *path* (a .pt / .pth file).

        Saves: model weights, optimizer state, current epoch, loss and
        accuracy histories.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "epoch":                self.current_epoch,
                "model_state_dict":     self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss_hist":            self.loss_hist,
                "acc_hist":             self.acc_hist,
            },
            path,
        )
        print(f"Checkpoint saved to {path}  (epoch {self.current_epoch})")

    def load_checkpoint(self, path: str) -> None:
        """Restore training state from a checkpoint file at *path*.

        The network and optimizer must already be constructed with the same
        architecture / parameter groups before calling this method.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.loss_hist     = checkpoint["loss_hist"]
        self.acc_hist      = checkpoint["acc_hist"]
        print(f"Checkpoint loaded from {path}  (epoch {self.current_epoch})")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def test(self, dataset_name: str) -> None:
        """Evaluate loss and accuracy on *dataset_name* ("validation" or "test")."""
        if dataset_name not in ("validation", "test"):
            raise ValueError('dataset_name must be "validation" or "test".')

        temp_loss = []
        temp_acc  = []

        self.net.eval()
        with torch.no_grad():
            for data, targets in self.datasets[dataset_name]:
                data    = data.to(device)
                targets = targets.to(device)

                output = self.net(data)
                pred, acc = self.predict(output, targets)
                loss = self.loss_fn(pred, targets)

                temp_loss.append(loss.item())
                temp_acc.append(acc.detach().cpu())

        self.loss_hist[dataset_name][self.current_epoch] = np.mean(temp_loss, 0)
        self.acc_hist[dataset_name][self.current_epoch]  = np.mean(temp_acc, 0)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        num_epochs: int,
        verbosity: int = 1,
        checkpoint_path: str = None,
        save_every: int = None,
    ) -> None:
        """Train for *num_epochs* epochs, validating after each one.

        Args:
            num_epochs:       number of epochs to train for
            verbosity:        0 = silent, 1 = print per-epoch metrics
            checkpoint_path:  if provided, save checkpoints to this path;
                              a suffix ``_epochN`` is inserted before the
                              extension for intermediate saves
            save_every:       save a checkpoint every this many epochs;
                              if None and checkpoint_path is set, only the
                              final checkpoint is saved
        """
        self.net.to(device)

        task_metric = "Average Error" if self.task == "Regression" else "Accuracy"

        # Baseline validation before any training
        self.test("validation")
        if verbosity:
            print(f"Epoch {self.current_epoch}:")
            print(f"  Validation Loss     = {self.loss_hist['validation'][self.current_epoch]}")
            print(f"  Validation {task_metric} = {self.acc_hist['validation'][self.current_epoch]}")
            print("\n-------------------------------\n")

        # Save value of optimizable parameters
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                self.par_hist[name].append(param.cpu().detach().numpy())
        for epoch in tqdm(range(num_epochs), desc="Epoch"):
            self.net.train()

            for data, targets in tqdm(self.datasets["train"], desc="Batches", leave=False):
                data    = data.to(device)
                targets = targets.to(device)

                output = self.net(data)
                pred, _ = self.predict(output, targets)
                loss_val = self.loss_fn(pred, targets)

                self.optimizer.zero_grad()
                loss_val.backward()
                # Save value of optimizable parameters
                for name, param in self.net.named_parameters():
                    if param.requires_grad:
                        pipi = param.cpu().clone().detach().numpy()
                        self.par_hist[name].append(pipi)
                self.optimizer.step()

                # Accumulate per-batch training loss
                epoch_losses = self.loss_hist["train"].setdefault(self.current_epoch, [])
                epoch_losses.append(loss_val.item())

            self.current_epoch += 1
            self.test("validation")

            if verbosity:
                print(f"Epoch {self.current_epoch}:")
                print(f"  Validation Loss     = {self.loss_hist['validation'][self.current_epoch]}")
                print(f"  Validation {task_metric} = {self.acc_hist['validation'][self.current_epoch]}")
                print("\n-------------------------------\n")

            if checkpoint_path and save_every and self.current_epoch % save_every == 0:
                base, ext = os.path.splitext(checkpoint_path)
                self.save_checkpoint(f"{base}_epoch{self.current_epoch}{ext}")

        if checkpoint_path:
            self.save_checkpoint(checkpoint_path)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_loss(self, validation: bool = True, logscale: bool = True) -> None:
        """Plot training (and optionally validation) loss curves."""
        loss = [l for epoch_losses in self.loss_hist["train"].values() for l in epoch_losses]

        fig = plt.figure(facecolor="w", figsize=(4, 3))
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        if logscale:
            plt.yscale("log")
        plt.plot(loss, label="Training")

        if validation:
            x = [i * len(self.datasets["train"]) for i in self.loss_hist["validation"]]
            plt.plot(
                x,
                list(self.loss_hist["validation"].values()),
                color="orange",
                marker="o",
                linestyle="dashed",
                label="Validation",
            )

        plt.legend(loc="upper right")
        plt.show()

    def _get_all(self, transform: Callable = lambda x, **kw: x):
        """Run inference on the test set and return targets, predictions, accuracy."""
        self.net.eval()
        all_targets, all_predictions, all_accuracy = [], [], []

        with torch.no_grad():
            for data, targets in self.datasets["test"]:
                data    = data.to(device)
                targets = targets.to(device)

                output = self.net(data)
                pred, acc = self.predict(output, targets, reduction="none")

                all_targets.append(transform(targets))
                all_predictions.append(transform(pred))
                all_accuracy.append(acc)

        all_targets     = torch.cat(all_targets,     dim=0).cpu()
        all_predictions = torch.cat(all_predictions, dim=0).cpu()
        all_accuracy    = torch.cat(all_accuracy,    dim=0).cpu()

        if all_targets.dim() < 2:
            all_targets     = all_targets.unsqueeze(-1)
            all_predictions = all_predictions.unsqueeze(-1)
            all_accuracy    = all_accuracy.unsqueeze(-1)

        return all_targets, all_predictions, all_accuracy

    def _plot_results(
        self,
        targets:     torch.Tensor = torch.tensor([]),
        predictions: torch.Tensor = torch.tensor([]),
        accuracy:    torch.Tensor = torch.tensor([]),
        plot_type:   str          = "2D",
        nbins:       int          = 50,
        title:       Union[str, list] = "",
        logscale:    bool         = False,
        select:      list         = [],
        *args, **kwargs,
    ) -> None:
        n_tasks = max(targets.shape[-1], accuracy.shape[-1])
        if select:
            n_tasks = min(n_tasks, len(select))
        else:
            select = list(range(n_tasks))

        ncols = math.ceil(math.sqrt(n_tasks))
        nrows = math.ceil(n_tasks / ncols)

        fig, axs = plt.subplots(
            ncols=ncols, nrows=nrows,
            facecolor="w", figsize=(5 * ncols, 4 * nrows),
            constrained_layout=True,
        )
        if not isinstance(axs, np.ndarray):
            axs   = np.array([axs])
            title = [title]
        axs = axs.flatten()

        hist = None
        for i in range(n_tasks):
            ax = axs[i]
            if plot_type == "2D":
                ax.set_xlabel("Targets",    fontsize=15)
                ax.set_ylabel("Prediction", fontsize=15)
                ax.set_title(title[i],      fontsize=15)

                if "E" in title[i] or "sigma" in title[i]:
                    r = [
                        min(targets[:, select[i]].min(), predictions[:, select[i]].min()),
                        max(targets[:, select[i]].max(), predictions[:, select[i]].max()),
                    ]
                else:
                    r = [0, 9]
                    ax.set_xticks(range(10))
                    ax.set_yticks(range(10))

                norm = SymLogNorm(*args, **kwargs) if logscale else None
                hist = ax.hist2d(
                    targets[:, select[i]], predictions[:, select[i]],
                    nbins, norm=norm, cmap="viridis", range=[r, r],
                )
                ax.plot([0, 1e5], [0, 1e5], color="white", linewidth=1, linestyle="--")

            elif plot_type == "1D":
                ax.set_xlabel("Residuals", fontsize=15)
                ax.set_ylabel("Counts",    fontsize=15)
                ax.set_title(title[i],     fontsize=15)
                ax.hist(accuracy[:, select[i]], nbins, edgecolor="black", alpha=0.7)
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(axis="both", which="major", labelsize=12)

        for i in range(n_tasks, len(axs)):
            fig.delaxes(axs[i])

        if plot_type == "2D" and hist is not None:
            cbar = fig.colorbar(hist[3], ax=axs, orientation="vertical", fraction=0.02, pad=0.04)
            cbar.set_label("Counts", fontsize=15)

    def plot_pred_vs_target(
        self, transform: Callable = lambda x, **kw: x, *args, **kwargs
    ) -> None:
        targets, predictions, _ = self._get_all(transform=transform)
        self._plot_results(targets=targets, predictions=predictions, plot_type="2D", *args, **kwargs)

    def plot_residuals(
        self, transform: Callable = lambda x, **kw: x, *args, **kwargs
    ) -> None:
        _, _, accuracy = self._get_all(transform=transform)
        self._plot_results(accuracy=accuracy, plot_type="1D", *args, **kwargs)

    def show_results(
        self, transform: Callable = lambda x, **kw: x, *args, **kwargs
    ) -> None:
        print(f"Test loss:           {self.loss_hist['test'][self.current_epoch]}")
        print(f"Test relative error: {self.acc_hist['test'][self.current_epoch] * 100} %")
        self.plot_loss()
        targets, predictions, accuracy = self._get_all(transform=transform)
        self._plot_results(targets=targets, predictions=predictions, plot_type="2D", *args, **kwargs)
        self._plot_results(accuracy=accuracy, plot_type="1D", *args, **kwargs)

    # ------------------------------------------------------------------
    # Confusion matrix (classification only)
    # ------------------------------------------------------------------

    def ConfusionMatrix(self, *args, **kwargs) -> MulticlassConfusionMatrix:
        """Build and return a confusion matrix over the test set."""
        cm = MulticlassConfusionMatrix(*args, **kwargs)

        self.net.eval()
        with torch.no_grad():
            for data, targets in self.datasets["test"]:
                data    = data.to(device)
                targets = targets.to(device)

                output = self.net(data)
                pred, _ = self.predict(output, targets)
                cm.update(pred, targets)

        return cm

    def get_par_hist(self):
        return self.par_hist

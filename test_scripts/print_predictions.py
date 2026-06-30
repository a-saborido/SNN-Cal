#!/usr/bin/env python
"""
Prediction and evaluation using a trained SNN model.

All test-set forward passes are performed once by `collect_test_outputs`,
and every downstream plot/diagnostic reuses the cached tensors.

Energy is always handled in log10 scale (matching generate_dataset.py).
A linear-energy view is derived afterwards by undoing the log10.
"""
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import CustomDataset
from SNN_func import Spiking_Net, Predictor, Trainer, multi_MSELoss, CubeletOrderedThresholdSpikeGenMulti
import snntorch as snn
from snntorch import surrogate
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# ------------------------- Helper functions -------------------------
def predict_spikefreq(output):
    # Sum spikes across time and average over the population
    return output.sum(0).mean(1)


def distance(prediction, targets, absolute: bool = True, relative: bool = True,
             transform: callable = lambda *args, **kwargs: args[0]):
    p, t = transform(prediction), transform(targets)
    accuracy = t - p
    if absolute:
        accuracy = torch.abs(accuracy)
    if relative:
        accuracy /= (t + 1e-8)
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


########################################################
# ------------------- Single forward pass --------------
########################################################

def collect_test_outputs(net, predictor, te_loader, pop_size):
    """
    Propagate the whole test set through the network and return
    every tensor the downstream diagnostics need.

    Returns
    -------
    targets : np.ndarray, shape (N, n_tasks)
        Ground-truth regression targets.
    predictions : np.ndarray, shape (N, n_tasks)
        Aggregated predictions (population-averaged spike frequency).
    spikes_per_neuron : np.ndarray, shape (N, pop, n_tasks)
        Total spikes emitted by each neuron of every population, per event.
        Used for population-variance and per-neuron activity diagnostics.
    """
    device = next(net.parameters()).device
    net.eval()

    all_targets = []
    all_preds = []
    all_spikes = []

    with torch.no_grad():
        for batch in te_loader:
            samples_batch, cubelets_batch, targets_batch = batch
            samples_batch = samples_batch.to(device)
            cubelets_batch = cubelets_batch.to(device)
            targets_batch = targets_batch.to(device)

            # Raw spike train: (T, B, pop * n_tasks)
            spk_out = net((samples_batch, cubelets_batch))

            # Aggregated prediction via the same Predictor used in training
            preds_batch, _ = predictor(spk_out, targets_batch)

            # Per-neuron spike counts: reshape to (T, B, pop, n_tasks),
            # then sum over time -> (B, pop, n_tasks)
            n_tasks = targets_batch.shape[1] if targets_batch.ndim > 1 else 1
            spk_reshaped = spk_out.reshape(spk_out.shape[0], spk_out.shape[1],
                                           pop_size, n_tasks)
            spikes_batch = spk_reshaped.sum(dim=0)  # (B, pop, n_tasks)

            all_targets.append(targets_batch.cpu().numpy())
            all_preds.append(preds_batch.cpu().numpy())
            all_spikes.append(spikes_batch.cpu().numpy())

    targets = np.concatenate(all_targets, axis=0)
    predictions = np.concatenate(all_preds, axis=0)
    spikes_per_neuron = np.concatenate(all_spikes, axis=0)

    return targets, predictions, spikes_per_neuron


def check_output_shapes(targets, predictions, spikes_per_neuron, pop_size):
    """
    Sanity-check the shapes of the collected test outputs (point 5).
    Raises AssertionError on any mismatch so problems surface immediately
    instead of producing silently wrong plots.
    """
    assert targets.ndim == 2, f"targets must be 2D (N, n_tasks), got {targets.shape}"
    assert predictions.ndim == 2, f"predictions must be 2D (N, n_tasks), got {predictions.shape}"
    assert spikes_per_neuron.ndim == 3, \
        f"spikes_per_neuron must be 3D (N, pop, n_tasks), got {spikes_per_neuron.shape}"

    N, n_tasks = targets.shape
    assert predictions.shape == (N, n_tasks), \
        f"predictions {predictions.shape} != targets {targets.shape}"
    assert spikes_per_neuron.shape == (N, pop_size, n_tasks), \
        f"spikes_per_neuron {spikes_per_neuron.shape} != (N={N}, pop={pop_size}, n_tasks={n_tasks})"

    # The population mean of per-neuron spikes must reproduce the aggregated
    # prediction (predict_spikefreq averages over the population). This ties
    # the two code paths together and catches reshape/axis-order mistakes.
    recomputed = spikes_per_neuron.mean(axis=1)  # (N, n_tasks)
    assert np.allclose(recomputed, predictions, atol=1e-4), \
        "Population mean of per-neuron spikes does not match the aggregated prediction"

    print(f"[shape check] OK  -  N={N}, n_tasks={n_tasks}, pop={pop_size}")


########################################################
# ------------------- Deep diagnostics -----------------
########################################################

def analyze_and_profile_predictions(targets, predictions,
                                     var_names=['log(E/MeV)', 'x_c', 'y_c', 'z_c'],
                                     num_bins=50):
    """
    Profile predictions, compute figures of merit, and assess the bias.
    Operates purely on precomputed arrays (no network propagation).
    """
    metrics_list = []

    # Two separate figures
    fig_prof, axes_prof = plt.subplots(2, 2, figsize=(14, 12))
    axes_prof = axes_prof.flatten()

    fig_bv, axes_bv = plt.subplots(2, 2, figsize=(14, 12))
    axes_bv = axes_bv.flatten()

    for i, var in enumerate(var_names):
        t = targets[:, i]
        p = predictions[:, i]
        residuals = p - t

        # ---------------------------------------------------------
        # Figures of merit
        # ---------------------------------------------------------
        r_corr, _ = stats.pearsonr(t, p)
        r2 = r2_score(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))
        mae = mean_absolute_error(t, p)
        global_bias = np.median(residuals)
        global_variance = np.var(residuals)

        metrics_list.append({
            'Variable': var,
            'Pearson (r)': r_corr,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'Bias': global_bias,
            'Residual Variance': global_variance
        })

        # ---------------------------------------------------------
        # Binned statistics (for the plots)
        # ---------------------------------------------------------
        bins = np.linspace(np.min(t), np.max(t), num_bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        pred_mean, _, _ = stats.binned_statistic(t, p, statistic='mean', bins=bins)
        pred_std, _, _ = stats.binned_statistic(t, p, statistic='std', bins=bins)
        bias_mean, _, _ = stats.binned_statistic(t, residuals, statistic='median', bins=bins)
        bias_var, _, _ = stats.binned_statistic(t, residuals, statistic=np.std, bins=bins)

        # =========================================================
        # Figure 1: combined 2D histogram + profile
        # =========================================================
        ax_p = axes_prof[i]
        h = ax_p.hist2d(t, p, bins=num_bins, cmap='viridis')
        ax_p.errorbar(bin_centers, pred_mean, yerr=pred_std, fmt='r.',
                      label=r'Profile (Mean $\pm$ Std)', markersize=8, capsize=3)
        ax_p.plot([np.min(t), np.max(t)], [np.min(t), np.max(t)],
                  'w--', alpha=0.7, label='Ideal')

        ax_p.set_title(f'Reconstruction vs True: {var}', fontsize=20)
        ax_p.set_xlabel('True value (target)', fontsize=16)
        ax_p.set_ylabel('Prediction', fontsize=16)
        ax_p.legend(loc='upper left', fontsize=16)
        ax_p.tick_params(axis='both', which='major', labelsize=14)
        fig_prof.colorbar(h[3], ax=ax_p, label='Counts')

        # =========================================================
        # Figure 2: bias vs standard deviation
        # =========================================================
        ax_bv = axes_bv[i]
        ax_bv.step(bin_centers, bias_mean, where='mid', color='magenta',
                   linewidth=2.5, label='Bias')
        ax_bv.step(bin_centers, bias_var, where='mid', color='blue', linestyle='--',
                   linewidth=2.5, alpha=0.8, label='Standard deviation')

        ax_bv.axhline(0, color='black', linestyle=':', alpha=0.6)  # zero baseline
        ax_bv.set_title(f'Bias vs Variance: {var}', fontsize=20)
        ax_bv.set_xlabel('True value (target)', fontsize=18)
        ax_bv.set_ylabel('Error', fontweight='bold', fontsize=18)
        ax_bv.tick_params(axis='both', which='major', labelsize=14)
        ax_bv.legend(loc='best', fontsize=18)

    # ---------------------------------------------------------
    # Save and show
    # ---------------------------------------------------------
    fig_prof.tight_layout()
    fig_prof.savefig("diagnostic_profile.png", dpi=300, bbox_inches="tight")

    fig_bv.tight_layout()
    fig_bv.savefig("diagnostic_bias_var.png", dpi=300, bbox_inches="tight")

    plt.close(fig_prof)
    plt.close(fig_bv)

    # ---------------------------------------------------------
    # LaTeX table of the figures of merit
    # ---------------------------------------------------------
    df_metrics = pd.DataFrame(metrics_list)
    print("\n--- Figures of merit ---")
    print(df_metrics.to_string(index=False))

    print("\n--- LaTeX code ---")
    latex_table = df_metrics.to_latex(
        index=False, float_format="%.3f",
        caption="Figures of merit for the reconstruction of calorimeter parameters.",
        label="tab:reconstruction_metrics")
    print(latex_table)

    return df_metrics


def plot_empirical_vs_expected_cdf(targets, predictions,
                                   var_names=['log(E/MeV)', 'x_c', 'y_c', 'z_c']):
    """
    Compare the cumulative distribution function (CDF) of the true values
    (expected) against the CDF of the network predictions (empirical).
    Operates purely on precomputed arrays.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor='white')
    axes = axes.flatten()

    for i, var in enumerate(var_names):
        t = targets[:, i]
        p = predictions[:, i]

        # Sort the data to build the CDF
        t_sorted = np.sort(t)
        p_sorted = np.sort(p)

        # Cumulative probability (0 to 1)
        cdf_t = np.arange(1, len(t_sorted) + 1) / len(t_sorted)
        cdf_p = np.arange(1, len(p_sorted) + 1) / len(p_sorted)

        ax = axes[i]
        ax.plot(t_sorted, cdf_t, label='Expected CDF (true data)',
                color='black', linewidth=2.5)
        ax.plot(p_sorted, cdf_p, label='Empirical CDF (predictions)',
                color='royalblue', linestyle='--', linewidth=2.5)

        ax.set_title(f'Cumulative distribution comparison: {var}', fontsize=16)
        ax.set_xlabel('Variable value', fontsize=14)
        ax.set_ylabel('Cumulative probability', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("diagnostic_cdf_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_population_variance(spikes_per_neuron, targets, task_idx=0,
                             title="Intra-population variance of the neural ensemble",
                             xlabel="True deposited energy log(E/MeV)",
                             ylabel="Population variance (Spikes^2)",
                             filename="diagnostic_population_variance.png"):
    """
    Scatter of intra-population spike variance vs the true target, per event.
    Reuses precomputed per-neuron spike counts (no network propagation).
    """
    # Variance across the population (axis=1) for the chosen task
    variances = spikes_per_neuron[:, :, task_idx].var(axis=1)
    t_np = targets[:, task_idx]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="w")
    ax.scatter(t_np, variances, alpha=0.5, s=15, color='royalblue')
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_neuron_activity_distribution(spikes_per_neuron, task_idx=0,
                                      title="Spike distribution per neuron",
                                      filename="diagnostic_neuron_activity.png"):
    """
    Boxplot of the total spikes emitted by each neuron of the ensemble.
    Reuses precomputed per-neuron spike counts (no network propagation).
    """
    spikes_task = spikes_per_neuron[:, :, task_idx]  # (N, pop)
    pop = spikes_task.shape[1]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="w")
    ax.boxplot(spikes_task, patch_artist=True, notch=False,
               boxprops=dict(facecolor='lightblue', color='blue', alpha=0.7),
               medianprops=dict(color='red', linewidth=2),
               flierprops=dict(marker='o', color='black', alpha=0.1, markersize=3))

    ax.set_xlabel(f"Neuron ID in the ensemble (1 to {pop})", fontsize=14)
    ax.set_ylabel("Total spikes emitted per event", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_energy_linear(targets, predictions,
                       filename_scatter="scatter_target_vs_predict_energy.png",
                       filename_profile="scatter_target_vs_predict_energy_v2.png",
                       num_bins=50):
    """
    Linear-energy VIEW derived from the log10 predictions (column 0).
    This only undoes the log10 for visualization; the model is always trained
    and evaluated in log space.
    """
    E_targets_log = targets[:, 0]
    E_preds_log = predictions[:, 0]

    targets_E = 10 ** E_targets_log      # undo log10
    predictions_E = 10 ** E_preds_log

    # --- Simple scatter ---
    plt.figure(figsize=(8, 6))
    plt.scatter(targets_E, predictions_E, alpha=0.5, color='royalblue', s=10)
    max_val = max(np.max(targets_E), np.max(predictions_E))
    plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal')
    plt.xlabel('True energy E (MeV)', fontsize=16)
    plt.ylabel('Predicted energy E (MeV)', fontsize=16)
    plt.title('Linear energy regression', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(filename_scatter, dpi=300)
    plt.close()

    # --- 2D histogram + profile (same format as the log plots) ---
    bins = np.linspace(np.min(targets_E), np.max(targets_E), num_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    pred_mean, _, _ = stats.binned_statistic(targets_E, predictions_E, statistic='mean', bins=bins)
    pred_std, _, _ = stats.binned_statistic(targets_E, predictions_E, statistic='std', bins=bins)

    fig, ax = plt.subplots(figsize=(8, 6))
    h = ax.hist2d(targets_E, predictions_E, bins=num_bins, cmap='viridis', cmin=1)
    ax.errorbar(bin_centers, pred_mean, yerr=pred_std, fmt='r.',
                label=r'Profile (Mean $\pm$ Std)', markersize=8, capsize=3)
    min_E, max_E = np.min(targets_E), np.max(targets_E)
    ax.plot([min_E, max_E], [min_E, max_E], 'w--', alpha=0.7, label='Ideal')

    ax.set_title('Reconstruction vs True: E (MeV) [linear scale]', fontsize=16)
    ax.set_xlabel('True value (target) E (MeV)', fontsize=14)
    ax.set_ylabel('Predicted E (MeV)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(loc='upper left', fontsize=10)
    fig.colorbar(h[3], ax=ax, label='Counts')

    plt.tight_layout()
    plt.savefig(filename_profile, dpi=300, bbox_inches="tight")
    plt.close()


########################################################

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="Path to cached dataset .pt file")
    p.add_argument("--model", required=True, help="Path to trained model .pth file")
    p.add_argument("--batch", type=int, default=32, help="Batch size for evaluation")
    p.add_argument("--lr", type=float, default=1e-2, help="Learning rate (unused placeholder)")
    p.add_argument("--analyze", action="store_true", help="Run deep diagnostics and profiling")
    args = p.parse_args()

    POP_SIZE = 20

    # Load cached data
    data_file = torch.load(args.cache, map_location="cpu")
    samples = data_file["samples"]
    cubelets = data_file["cubelets"]
    targets = data_file["targets"]

    ds = CustomDataset(filelist=[], primary_only=True, target=data_file["target_name"])
    ds.data = list(zip(samples, cubelets, targets))

    # Split dataset (70% train, 15% val, 15% test)
    total = len(ds)

    # Fix seed for consistency with train_model.py
    seed = 42
    g = torch.Generator().manual_seed(seed)

    tr_len = int(0.70 * total)
    va_len = int(0.15 * total)
    te_len = total - tr_len - va_len
    tr_ds, va_ds, te_ds = random_split(ds, [tr_len, va_len, te_len], generator=g)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch)
    te_loader = DataLoader(te_ds, batch_size=args.batch)

    # Build and load network
    n_tasks = targets.shape[1] if targets.ndim > 1 else 1
    net_desc = make_net_desc(n_tasks)
    encoder = CubeletOrderedThresholdSpikeGenMulti(
        n_cubelets=1000,
        multiplicity=4,
        alpha=5.0
    )
    net = Spiking_Net(net_desc, encoder)
    net.load_state_dict(torch.load(args.model, map_location="cpu"))

    # Predictor, loss, optimizer (optimizer unused here)
    predictor = Predictor(predict_spikefreq, distance, population_sizes=POP_SIZE)
    loss_fn = multi_MSELoss(weights=torch.tensor([1] * n_tasks))
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # Trainer setup (reused only for the basic diagnostics in SNN_func.py)
    train_Epos_spk = Trainer(net, loss_fn, optimizer, predictor,
                             tr_loader, va_loader, te_loader,
                             task="Regression")

    # --- Single forward pass over the test set: everything reuses this ---
    test_targets, test_predictions, spikes_per_neuron = collect_test_outputs(
        net, predictor, te_loader, POP_SIZE)
    check_output_shapes(test_targets, test_predictions, spikes_per_neuron, POP_SIZE)

    var_names = ["log(E/MeV)", r"$x_c$", r"$y_c$", r"$z_c$"]

    # ---------------------------------------------------------
    # Basic test metrics (loss / relative error)
    # ---------------------------------------------------------
    train_Epos_spk.predict.accuracy_fn = lambda p, t: distance(p, t, absolute=True, relative=True)
    train_Epos_spk.test("test")
    print(f"Test loss: {train_Epos_spk.loss_hist['test'][train_Epos_spk.current_epoch]}")
    print(f"Test relative error: {train_Epos_spk.acc_hist['test'][train_Epos_spk.current_epoch] * 100}%")

    # ---------------------------------------------------------
    # Basic diagnostics kept in SNN_func.py (they reuse _get_all internally;
    # cheap compared to the deep diagnostics and left as-is by design)
    # ---------------------------------------------------------
    train_Epos_spk.predict.accuracy_fn = lambda p, t: distance(p, t, absolute=False, relative=False)
    train_Epos_spk.plot_pred_vs_target(title=var_names, nbins=50)
    plt.savefig("pred_vs_target.png", dpi=300, bbox_inches="tight")
    plt.close()
    train_Epos_spk.plot_residuals(title=var_names, nbins=50)
    plt.savefig("residuals.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------
    # Linear-energy view (derived from the log10 predictions)
    # ---------------------------------------------------------
    plot_energy_linear(test_targets, test_predictions)

    # ---------------------------------------------------------
    # Deep diagnostics (all fed by the single forward pass above)
    # ---------------------------------------------------------
    if args.analyze:
        print("\n--- Running deep diagnostics on the test set ---")
        diag_var_names = ["log(E/MeV)", "x_c", "y_c", "z_c"]

        analyze_and_profile_predictions(
            targets=test_targets,
            predictions=test_predictions,
            var_names=diag_var_names,
            num_bins=50,
        )

        plot_empirical_vs_expected_cdf(
            targets=test_targets,
            predictions=test_predictions,
            var_names=diag_var_names,
        )

        # --- Structural diagnostics of the neurons ---
        plot_population_variance(
            spikes_per_neuron, test_targets, task_idx=0,
            title="Intra-population variance of the neural ensemble",
            xlabel="True deposited energy log(E/MeV)",
            ylabel="Population variance (Spikes^2)",
            filename="diagnostic_population_variance.png",
        )

        plot_neuron_activity_distribution(
            spikes_per_neuron, task_idx=0,
            title="Activity distribution of the 20 neurons (energy)",
            filename="diagnostic_neuron_activity.png",
        )


if __name__ == "__main__":
    main()

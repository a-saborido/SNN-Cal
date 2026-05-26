#!/usr/bin/env python
"""
Prediction and evaluation using a trained SNN model
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

########################################################
#-------------------Diagnostics-------------------------
########################################################

def analyze_and_profile_predictions(targets, predictions, var_names=['log(E/MeV)', 'x_c', 'y_c', 'z_c'], num_bins=50):
    """
    Realiza el perfilado (profiling), calcula métricas de error y evalúa el bias.
    """
    metrics_list = []
    
    # Preparamos DOS figuras separadas
    fig_prof, axes_prof = plt.subplots(2, 2, figsize=(14, 12))
    axes_prof = axes_prof.flatten()
    
    fig_bv, axes_bv = plt.subplots(2, 2, figsize=(14, 12))
    axes_bv = axes_bv.flatten()

    for i, var in enumerate(var_names):
        t = targets[:, i]
        p = predictions[:, i]
        residuals = p - t
        
        # ---------------------------------------------------------
        # TAREA 2: Calcular Coeficientes y Figuras de Mérito
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
            'Varianza Residuos': global_variance
        })

        # ---------------------------------------------------------
        # ESTADÍSTICA POR BINS (Para las gráficas)
        # ---------------------------------------------------------
        bins = np.linspace(np.min(t), np.max(t), num_bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        pred_mean, _, _ = stats.binned_statistic(t, p, statistic='mean', bins=bins)
        pred_std, _, _ = stats.binned_statistic(t, p, statistic='std', bins=bins)
        bias_mean, _, _ = stats.binned_statistic(t, residuals, statistic='median', bins=bins)
        bias_var, _, _ = stats.binned_statistic(t, residuals, statistic=np.std, bins=bins)

        # =========================================================
        # FIGURA 1: Gráfica Combinada 2D + Perfil (Limpia)
        # =========================================================
        ax_p = axes_prof[i]
        h = ax_p.hist2d(t, p, bins=num_bins, cmap='viridis', cmin=1)
        ax_p.errorbar(bin_centers, pred_mean, yerr=pred_std, fmt='r.', label=r'Perfil (Media $\pm$ Std)', markersize=8,                  capsize=3)
        ax_p.plot([np.min(t), np.max(t)], [np.min(t), np.max(t)], 'w--', alpha=0.7, label='Ideal')
        
        ax_p.set_title(f'Reconstrucción vs Real: {var}')
        ax_p.set_xlabel('Valor Real (Target)')
        ax_p.set_ylabel('Predicción')
        ax_p.legend(loc='upper left')
        fig_prof.colorbar(h[3], ax=ax_p, label='Counts')

        # =========================================================
        # FIGURA 2: Bias vs Varianza en solitario
        # =========================================================
        ax_bv = axes_bv[i]
        
        # Pintamos el Bias (Magenta) y la Desviación (Azul) en el MISMO eje
        ax_bv.step(bin_centers, bias_mean, where='mid', color='magenta', linewidth=2.5, label='Bias (Sesgo)')
        ax_bv.step(bin_centers, bias_var, where='mid', color='blue', linestyle='--', linewidth=2.5, alpha=0.8, label='Desviación Estándar')
        
        ax_bv.axhline(0, color='black', linestyle=':', alpha=0.6) # Línea base 0
        ax_bv.set_ylabel('Error (Mismas unidades que la variable)', fontweight='bold')
        ax_bv.set_title(f'Descomposición del Error: {var}')
        ax_bv.set_xlabel('Valor Real (Target)')
        ax_bv.legend(loc='best', fontsize=10)
        ax_bv.set_title(f'Comparativa Bias vs Varianza: {var}')
        ax_bv.set_xlabel('Valor Real (Target)')
        

    # ---------------------------------------------------------
    # GUARDADO Y MUESTRA
    # ---------------------------------------------------------
    fig_prof.tight_layout()
    fig_prof.savefig("diagnostico_perfil.png", dpi=300, bbox_inches="tight")
    
    fig_bv.tight_layout()
    fig_bv.savefig("diagnostico_bias_var.png", dpi=300, bbox_inches="tight")
    
    plt.show()

    # ---------------------------------------------------------
    # Generar Tabla para el PDF (LaTeX)
    # ---------------------------------------------------------
    df_metrics = pd.DataFrame(metrics_list)
    print("\n--- Tabla de Figuras de Mérito ---")
    print(df_metrics.to_string(index=False))
    
    print("\n--- Código LaTeX para tu Memoria ---")
    # Formateamos los números a 3 decimales para que el PDF quede profesional
    latex_table = df_metrics.to_latex(index=False, float_format="%.3f", caption="Figuras de mérito para la reconstrucción de parámetros del calorímetro.", label="tab:metricas_reconstruccion")
    print(latex_table)

    return df_metrics

########################################################

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="Path to cached dataset .pt file")
    p.add_argument("--model", required=True, help="Path to trained model .pth file")
    p.add_argument("--batch", type=int, default=32, help="Batch size for evaluation")
    p.add_argument("--lr", type=float, default=1e-2, help="Learning rate (unused placeholder)")
    p.add_argument("--analyze", action="store_true", help="Ejecutar análisis de diagnóstico y profiling")
    args = p.parse_args()

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
    #net = Spiking_Net(net_desc, spikegen_multi)
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
    # title=["log(E/MeV)", r"$x_c$", r"$y_c$", r"$z_c$"], nbins=50)
	title=["logE (MeV)", r"$x_c$", r"$y_c$", r"$z_c$"],nbins=50)
    plt.savefig("pred_vs_target.png", dpi=300, bbox_inches="tight")
    train_Epos_spk.plot_residuals(
    # title=["log(E/MeV)", r"$x_c$", r"$y_c$", r"$z_c$"], nbins=50)
	title=["logE (MeV)", r"$x_c$", r"$y_c$", r"$z_c$"],nbins=50)	
    plt.savefig("residuals.png", dpi=300, bbox_inches="tight")

    plt.show()

	
# === PLOT LINEAL ===
    test_targets, test_predictions, _ = train_Epos_spk._get_all(transform=lambda x: x)
    E_targets_log = test_targets[:, 0].cpu().numpy()
    E_preds_log = test_predictions[:, 0].cpu().numpy()

    targets_E = 10**(E_targets_log) # Deshace el logaritmo natural
    predictions_E = 10**(E_preds_log)

    plt.figure(figsize=(8, 6))
    plt.scatter(targets_E, predictions_E, alpha=0.5, color='royalblue', s=10)
    max_val = max(np.max(targets_E), np.max(predictions_E))
    plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal')
    plt.xlabel('Energía Real E (MeV)')
    plt.ylabel('Energía Predicha E (MeV)')
    plt.title('Regresión Lineal de Energía')
    plt.grid(True, alpha=0.3)
    plt.savefig('scatter_target_vs_predict_Energy.png', dpi=300)

# === PLOT LINEAL MISMO FORMATO QUE ANTES ===

	# 1. Calculamos la estadística por bines para el perfil
    num_bins = 50
    bins = np.linspace(np.min(targets_E), np.max(targets_E), num_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    pred_mean, _, _ = stats.binned_statistic(targets_E, predictions_E, statistic='mean', bins=bins)
    pred_std, _, _ = stats.binned_statistic(targets_E, predictions_E, statistic='std', bins=bins)

    # 2. Creamos la figura con el mismo formato que el logE
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Histograma 2D
    h = ax.hist2d(targets_E, predictions_E, bins=num_bins, cmap='viridis', cmin=1)
    
    # Perfil (Media +- Std) superpuesto
    ax.errorbar(bin_centers, pred_mean, yerr=pred_std, fmt='r.', 
                label=r'Perfil (Media $\pm$ Std)', markersize=8, capsize=3)
    
    # Línea ideal (y = x)
    min_E, max_E = np.min(targets_E), np.max(targets_E)
    ax.plot([min_E, max_E], [min_E, max_E], 'w--', alpha=0.7, label='Ideal')
    
    # Títulos y formato
    ax.set_title('Reconstrucción vs Real: E (MeV) [Escala Lineal]')
    ax.set_xlabel('Valor Real (Target) E (MeV)')
    ax.set_ylabel('Predicción E (MeV)')
    ax.legend(loc='upper left')
    fig.colorbar(h[3], ax=ax, label='Counts')

    plt.tight_layout()
    plt.savefig('scatter_target_vs_predict_Energyv2.png', dpi=300, bbox_inches="tight")
    # plt.show() # Descomentar si quieres verlo en pantalla al vuelo
    plt.close()
	

    # ---------------------------------------------------------
    # EJECUCIÓN DEL DIAGNÓSTICO (NUEVO BLOQUE)
    # ---------------------------------------------------------
    if args.analyze:
        print("\n--- Extrayendo predicciones del Test Set para Diagnóstico ---")
        net.eval()
        all_targets = []
        all_preds = []
        
        with torch.no_grad():
            # Iteramos sobre el dataloader de test (te_loader)
            for samples_batch, cubelets_batch, targets_batch in te_loader:
                # Asegurarnos de que los datos están en la CPU (o GPU si lo usas)
                # Asegurarnos de que los datos están en la CPU (o GPU si lo usas)
                device = next(net.parameters()).device
                samples_batch = samples_batch.to(device)
                cubelets_batch = cubelets_batch.to(device)  

                # Pasamos los datos por la red (COMO TUPLA) y el predictor
                spk_out = net((samples_batch, cubelets_batch))  
                preds_batch, _ = predictor(spk_out, targets_batch)
                
                all_targets.append(targets_batch.cpu().numpy())
                all_preds.append(preds_batch.cpu().numpy())
                
        # Concatenamos todas las listas en arrays de NumPy de forma (N, 4)
        targets_np = np.concatenate(all_targets, axis=0)
        preds_np = np.concatenate(all_preds, axis=0)
        
        # Llamamos a nuestra función de diagnóstico
        # Cambiamos log(E/MeV) por E (MeV) si corresponde a tu target
        analyze_and_profile_predictions(
            targets=targets_np, 
            predictions=preds_np, 
            var_names=["log(E/MeV)", "x_c", "y_c", "z_c"], 
            num_bins=50
        )

# --- DIAGNÓSTICO ESTRUCTURAL DE LAS NEURONAS ---
        
        # train_Epos_spk.plot_population_variance(
        #     task_idx=0, 
        #     title="Varianza Intra-Poblacional del Ensamble Neuronal",
        #     xlabel="Energía Real Depositada log(E/MeV)",
        #     ylabel="Varianza Poblacional (Spikes²)",
        #     filename="diagnostico_varianza_poblacional.png"
        # )
        
        # train_Epos_spk.plot_neuron_activity_distribution(
        #     task_idx=0,
        #     title="Distribución de Actividad de las 20 Neuronas (Energía)",
        #     filename="diagnostico_actividad_neuronas.png"
        # )
        train_Epos_spk.plot_population_variance(
            task_idx=0, 
            title="Intra-Population Variance of the 20 Neurons",
            xlabel="True Deposited Energy log(E/MeV)",
            ylabel="Population Variance (Spikes²)",
            filename="diagnostico_varianza_poblacional.png"
        )
        
        train_Epos_spk.plot_neuron_activity_distribution(
            task_idx=0,
            title="Activity Distribution of the 20 Neurons (Energy)",
            filename="diagnostico_actividad_neuronas.png"
        )
if __name__ == "__main__":
    main()

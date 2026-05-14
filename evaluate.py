"""
Evaluate and compare ConvLSTM vs U-Net + ConvLSTM results.
Loads CV results and generates comparison plots and tables for paper.
"""

import pickle
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import config


def load_results(model_name):
    """Load CV results for a model."""
    results_file = config.RESULTS_DIR / f"{model_name}_cv_results.pkl"
    if not results_file.exists():
        raise FileNotFoundError(f"Results not found: {results_file}")
    
    with open(results_file, "rb") as f:
        results = pickle.load(f)
    
    return results


def comparison_table():
    """Generate comparison table."""
    print("\n" + "="*70)
    print("MODEL COMPARISON TABLE")
    print("="*70 + "\n")
    
    try:
        convlstm_results = load_results("convlstm")
    except FileNotFoundError:
        print("[ERROR] ConvLSTM results not found. Train baseline first.")
        return
    
    try:
        unet_results = load_results("unet_convlstm")
    except FileNotFoundError:
        print("[ERROR] U-Net results not found. Train proposed model first.")
        return
    
    # Extract metrics
    data = {
        "Metric": ["RMSE (m)", "MAE (m)", "CSI", "SSIM"],
        "ConvLSTM (Baseline)": [
            f"{convlstm_results['mean_metrics']['rmse']:.4f} ± {convlstm_results['std_metrics']['rmse']:.4f}",
            f"{convlstm_results['mean_metrics']['mae']:.4f} ± {convlstm_results['std_metrics']['mae']:.4f}",
            f"{convlstm_results['mean_metrics']['csi']:.4f} ± {convlstm_results['std_metrics']['csi']:.4f}",
            f"{convlstm_results['mean_metrics']['ssim']:.4f} ± {convlstm_results['std_metrics']['ssim']:.4f}",
        ],
        "U-Net + ConvLSTM (Proposed)": [
            f"{unet_results['mean_metrics']['rmse']:.4f} ± {unet_results['std_metrics']['rmse']:.4f}",
            f"{unet_results['mean_metrics']['mae']:.4f} ± {unet_results['std_metrics']['mae']:.4f}",
            f"{unet_results['mean_metrics']['csi']:.4f} ± {unet_results['std_metrics']['csi']:.4f}",
            f"{unet_results['mean_metrics']['ssim']:.4f} ± {unet_results['std_metrics']['ssim']:.4f}",
        ],
    }
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print()
    
    # Calculate improvements
    print("Improvements (U-Net vs ConvLSTM):")
    rmse_improvement = (
        (convlstm_results['mean_metrics']['rmse'] - unet_results['mean_metrics']['rmse']) 
        / convlstm_results['mean_metrics']['rmse'] * 100
    )
    mae_improvement = (
        (convlstm_results['mean_metrics']['mae'] - unet_results['mean_metrics']['mae']) 
        / convlstm_results['mean_metrics']['mae'] * 100
    )
    csi_improvement = (
        (unet_results['mean_metrics']['csi'] - convlstm_results['mean_metrics']['csi']) 
        / convlstm_results['mean_metrics']['csi'] * 100
    )
    ssim_improvement = (
        (unet_results['mean_metrics']['ssim'] - convlstm_results['mean_metrics']['ssim']) 
        / convlstm_results['mean_metrics']['ssim'] * 100
    )
    
    print(f"  RMSE: {rmse_improvement:+.2f}%")
    print(f"  MAE:  {mae_improvement:+.2f}%")
    print(f"  CSI:  {csi_improvement:+.2f}%")
    print(f"  SSIM: {ssim_improvement:+.2f}%")
    print()


def per_fold_table():
    """Per-fold breakdown for paper appendix."""
    print("\n" + "="*70)
    print("PER-FOLD RESULTS")
    print("="*70 + "\n")
    
    try:
        convlstm_results = load_results("convlstm")
        unet_results = load_results("unet_convlstm")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return
    
    folds = [f"Fold {i+1}" for i in range(5)]
    data = {
        "Fold": folds,
        "ConvLSTM RMSE": [f"{m['rmse']:.4f}" for m in convlstm_results['metrics_per_fold']],
        "U-Net RMSE": [f"{m['rmse']:.4f}" for m in unet_results['metrics_per_fold']],
        "ConvLSTM CSI": [f"{m['csi']:.4f}" for m in convlstm_results['metrics_per_fold']],
        "U-Net CSI": [f"{m['csi']:.4f}" for m in unet_results['metrics_per_fold']],
    }
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print()


def plot_comparison():
    """Generate comparison plots."""
    try:
        convlstm_results = load_results("convlstm")
        unet_results = load_results("unet_convlstm")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return
    
    metrics = ["rmse", "mae", "csi", "ssim"]
    convlstm_means = [convlstm_results['mean_metrics'][m] for m in metrics]
    unet_means = [unet_results['mean_metrics'][m] for m in metrics]
    
    convlstm_stds = [convlstm_results['std_metrics'][m] for m in metrics]
    unet_stds = [unet_results['std_metrics'][m] for m in metrics]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Model Comparison: ConvLSTM vs U-Net + ConvLSTM")
    
    metric_labels = ["RMSE (m)", "MAE (m)", "CSI", "SSIM"]
    x = np.arange(1)
    width = 0.35
    
    for idx, (ax, metric, label) in enumerate(zip(axes.flat, metrics, metric_labels)):
        conv_mean = convlstm_results['mean_metrics'][metric]
        conv_std = convlstm_results['std_metrics'][metric]
        unet_mean = unet_results['mean_metrics'][metric]
        unet_std = unet_results['std_metrics'][metric]
        
        ax.bar(x - width/2, [conv_mean], width, label="ConvLSTM", 
               yerr=[conv_std], capsize=5, alpha=0.8)
        ax.bar(x + width/2, [unet_mean], width, label="U-Net + ConvLSTM", 
               yerr=[unet_std], capsize=5, alpha=0.8)
        
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks([])
        if idx == 0:
            ax.legend()
    
    plt.tight_layout()
    plot_file = config.RESULTS_DIR / "model_comparison.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {plot_file}")
    plt.close()


def export_latex_table():
    """Export comparison table in LaTeX format for paper."""
    print("\n" + "="*70)
    print("LaTeX TABLE (for paper)")
    print("="*70 + "\n")
    
    try:
        convlstm_results = load_results("convlstm")
        unet_results = load_results("unet_convlstm")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return
    
    latex = r"""
\begin{table}[ht]
\centering
\caption{Flood Depth Prediction Performance (5-Fold Leave-One-Out CV)}
\begin{tabular}{ccc}
\hline
\textbf{Metric} & \textbf{ConvLSTM (Baseline)} & \textbf{U-Net + ConvLSTM (Proposed)} \\
\hline
"""
    
    metrics_display = [
        ("rmse", "RMSE (m)"),
        ("mae", "MAE (m)"),
        ("csi", "CSI"),
        ("ssim", "SSIM"),
    ]
    
    for metric_key, metric_label in metrics_display:
        conv_mean = convlstm_results['mean_metrics'][metric_key]
        conv_std = convlstm_results['std_metrics'][metric_key]
        unet_mean = unet_results['mean_metrics'][metric_key]
        unet_std = unet_results['std_metrics'][metric_key]
        
        latex += f"{metric_label} & ${conv_mean:.4f} \\pm {conv_std:.4f}$ & ${unet_mean:.4f} \\pm {unet_std:.4f}$ \\\\\n"
    
    latex += r"""\hline
\end{tabular}
\end{table}
"""
    
    print(latex)
    
    latex_file = config.RESULTS_DIR / "comparison_table.tex"
    with open(latex_file, "w") as f:
        f.write(latex)
    print(f"Saved LaTeX: {latex_file}")


def main():
    """Generate all evaluation outputs."""
    print("\n" + "="*70)
    print("FLOOD DEPTH PREDICTION - MODEL EVALUATION")
    print("="*70)
    
    comparison_table()
    per_fold_table()
    plot_comparison()
    export_latex_table()
    
    print("\n" + "="*70)
    print("Evaluation complete.")
    print("="*70)


if __name__ == "__main__":
    main()

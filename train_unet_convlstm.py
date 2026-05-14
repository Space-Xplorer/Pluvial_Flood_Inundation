"""
Train U-Net + ConvLSTM proposed model.
Leave-one-out cross-validation over 5 simulations.
"""

import os
from pathlib import Path
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.cuda.amp as amp

import config
from dataset import FloodDataset
from metrics import MetricsTracker
from models import UNetConvLSTM


def setup():
    """Random seed + devices."""
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = True


def load_dataset(test_sim_idx):
    """Load dataset with leave-one-out split."""
    if not config.DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {config.DATASET_FILE}")
    
    data = np.load(config.DATASET_FILE, allow_pickle=True)
    X = data["X"]
    Y = data["Y"]
    sim_ids = data["sim_id"]
    
    # Compute normalization from training set
    train_mask = sim_ids != test_sim_idx
    X_train = X[train_mask]
    mean_depth, std_depth = FloodDataset.compute_statistics(X_train)
    
    # Split indices
    train_idx = np.where(sim_ids != test_sim_idx)[0]
    test_idx = np.where(sim_ids == test_sim_idx)[0]
    
    # Create datasets
    train_dataset = FloodDataset(
        X, Y, sim_ids,
        augment=True,
        augmentation_config=config.AUGMENTATION_CONFIG,
        normalize=True,
        mean_depth=mean_depth,
        std_depth=std_depth,
    )
    
    test_dataset = FloodDataset(
        X, Y, sim_ids,
        augment=False,
        normalize=True,
        mean_depth=mean_depth,
        std_depth=std_depth,
    )
    
    # Create subsets
    train_subset = Subset(train_dataset, train_idx)
    test_subset = Subset(test_dataset, test_idx)
    
    return train_subset, test_subset


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for X, Y, sim_id in loader:
        X = X.to(device)
        Y = Y.to(device)
        
        optimizer.zero_grad()
        
        with amp.autocast(enabled=config.USE_AMP):
            pred = model(X)
            loss = criterion(pred, Y)
        
        scaler.scale(loss).backward()
        if config.GRADIENT_CLIP > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """Evaluate on dataset."""
    model.eval()
    total_loss = 0.0
    metrics_tracker = MetricsTracker(threshold=config.FLOOD_THRESHOLD)
    
    with torch.no_grad():
        for X, Y, sim_id in loader:
            X = X.to(device)
            Y = Y.to(device)
            
            pred = model(X)
            loss = criterion(pred, Y)
            total_loss += loss.item()
            
            # Update metrics
            metrics_tracker.update(pred, Y)
    
    metrics = metrics_tracker.get_metrics()
    return total_loss / len(loader), metrics


def train_fold(fold_idx, test_sim_idx):
    """Train one fold of leave-one-out CV."""
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx+1}/5: Test on Simulation {test_sim_idx}")
    print(f"{'='*60}")
    
    setup()
    
    # Load data
    train_dataset, test_dataset = load_dataset(test_sim_idx)
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    
    # Model, optimizer, loss
    device = torch.device(config.DEVICE)
    model = UNetConvLSTM(**config.UNET_CONVLSTM_CONFIG).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, 
                           weight_decay=config.WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scaler = amp.GradScaler(enabled=config.USE_AMP)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=config.COSINE_TMIN
    )
    
    # Training loop
    best_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "test_loss": [], "test_metrics": []}
    
    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        test_loss, metrics = evaluate(model, test_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_metrics"].append(metrics)
        
        scheduler.step()
        
        if (epoch + 1) % config.LOG_FREQ == 0:
            print(f"Epoch {epoch+1}/{config.EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
            print(f"  Metrics: {metrics}")
        
        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            if config.SAVE_BEST_MODEL:
                checkpoint_path = config.CHECKPOINT_DIR / f"unet_convlstm_fold{fold_idx}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model and final evaluation
    checkpoint_path = config.CHECKPOINT_DIR / f"unet_convlstm_fold{fold_idx}.pt"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path))
    
    final_loss, final_metrics = evaluate(model, test_loader, criterion, device)
    history["final_metrics"] = final_metrics
    
    print(f"\nFold {fold_idx+1} Final Metrics:")
    print(f"  RMSE: {final_metrics['rmse']:.4f}")
    print(f"  MAE:  {final_metrics['mae']:.4f}")
    print(f"  CSI:  {final_metrics['csi']:.4f}")
    print(f"  SSIM: {final_metrics['ssim']:.4f}")
    
    return history


def main():
    """Leave-one-out cross-validation."""
    print("Training U-Net + ConvLSTM Proposed Model")
    print(f"Device: {config.DEVICE}")
    print(f"Grid Size: {config.GRID_SIZE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    
    all_histories = {}
    all_metrics = []
    
    for fold_idx, test_sim_idx in enumerate(config.SIMULATE_IDS):
        history = train_fold(fold_idx, test_sim_idx)
        all_histories[f"fold_{fold_idx}"] = history
        all_metrics.append(history["final_metrics"])
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("5-Fold Leave-One-Out CV Results (U-Net + ConvLSTM)")
    print(f"{'='*60}")
    
    rmse_scores = [m["rmse"] for m in all_metrics]
    mae_scores = [m["mae"] for m in all_metrics]
    csi_scores = [m["csi"] for m in all_metrics]
    ssim_scores = [m["ssim"] for m in all_metrics]
    
    print(f"RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"MAE:  {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"CSI:  {np.mean(csi_scores):.4f} ± {np.std(csi_scores):.4f}")
    print(f"SSIM: {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")
    
    # Save results
    results_file = config.RESULTS_DIR / "unet_convlstm_cv_results.pkl"
    with open(results_file, "wb") as f:
        pickle.dump(
            {
                "histories": all_histories,
                "metrics_per_fold": all_metrics,
                "mean_metrics": {
                    "rmse": float(np.mean(rmse_scores)),
                    "mae": float(np.mean(mae_scores)),
                    "csi": float(np.mean(csi_scores)),
                    "ssim": float(np.mean(ssim_scores)),
                },
                "std_metrics": {
                    "rmse": float(np.std(rmse_scores)),
                    "mae": float(np.std(mae_scores)),
                    "csi": float(np.std(csi_scores)),
                    "ssim": float(np.std(ssim_scores)),
                },
            },
            f,
        )
    
    print(f"\nResults saved: {results_file}")


if __name__ == "__main__":
    main()

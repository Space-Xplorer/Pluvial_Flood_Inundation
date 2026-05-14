"""
Evaluation metrics for flood depth prediction.
Includes RMSE, MAE, CSI (hydrology standard), and SSIM (structure preservation).
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import convolve2d


def rmse(pred, target):
    """Root Mean Squared Error."""
    return float(torch.sqrt(torch.mean((pred - target) ** 2)).cpu().numpy())


def mae(pred, target):
    """Mean Absolute Error."""
    return float(torch.mean(torch.abs(pred - target)).cpu().numpy())


def csi(pred, target, threshold=0.3):
    """
    Critical Success Index (Hanssen & Kuipers, used in hydrology).
    
    CSI = (hits) / (hits + false_alarms + misses)
    
    Good for flood extent prediction.
    Values: 0 (no skill) to 1 (perfect).
    
    Args:
        pred: (N, H, W) predictions
        target: (N, H, W) target (ground truth)
        threshold: float - depth threshold for "flooded"
    
    Returns:
        csi_score: float [0, 1]
    """
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    
    hits = torch.sum(pred_bin * target_bin)
    false_alarms = torch.sum(pred_bin * (1 - target_bin))
    misses = torch.sum((1 - pred_bin) * target_bin)
    
    csi_score = hits / (hits + false_alarms + misses + 1e-6)
    return float(csi_score.cpu().numpy())


def ssim(pred, target, data_range=1.0):
    """
    Structural Similarity Index (SSIM).
    
    Measures perceived structural similarity between images.
    Range [-1, 1], where 1 = perfect similarity.
    Good for flood spatial pattern preservation.
    
    Args:
        pred: (N, H, W) predictions
        target: (N, H, W) target
        data_range: float - max value in data (for normalization)
    
    Returns:
        mean_ssim: float
    """
    # Simple per-frame SSIM
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    
    pred_mean = F.avg_pool2d(pred.unsqueeze(0), kernel_size=11, stride=1, padding=5)
    target_mean = F.avg_pool2d(
        target.unsqueeze(0), kernel_size=11, stride=1, padding=5
    )
    
    pred_sq = F.avg_pool2d(
        (pred**2).unsqueeze(0), kernel_size=11, stride=1, padding=5
    )
    target_sq = F.avg_pool2d(
        (target**2).unsqueeze(0), kernel_size=11, stride=1, padding=5
    )
    pred_target = F.avg_pool2d(
        (pred * target).unsqueeze(0), kernel_size=11, stride=1, padding=5
    )
    
    pred_var = pred_sq - pred_mean**2
    target_var = target_sq - target_mean**2
    pred_target_cov = pred_target - pred_mean * target_mean
    
    ssim_map = (
        (2 * pred_mean * target_mean + c1)
        * (2 * pred_target_cov + c2)
        / (
            (pred_mean**2 + target_mean**2 + c1)
            * (pred_var + target_var + c2)
        )
    )
    
    return float(torch.mean(ssim_map).cpu().numpy())


class MetricsTracker:
    """Track and aggregate metrics across batches."""

    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.rmse_list = []
        self.mae_list = []
        self.csi_list = []
        self.ssim_list = []

    def update(self, pred, target):
        """
        Args:
            pred: (B, T, 1, H, W) or (H, W)
            target: (B, T, 1, H, W) or (H, W)
        """
        # Flatten batch and time dimensions
        if pred.dim() == 5:
            B, T, C, H, W = pred.shape
            pred = pred.view(-1, H, W)
            target = target.view(-1, H, W)
        
        self.rmse_list.append(rmse(pred, target))
        self.mae_list.append(mae(pred, target))
        self.csi_list.append(csi(pred, target, threshold=self.threshold))
        self.ssim_list.append(ssim(pred, target))

    def get_metrics(self):
        """Return aggregated metrics."""
        metrics = {
            "rmse": float(np.mean(self.rmse_list)) if self.rmse_list else 0.0,
            "mae": float(np.mean(self.mae_list)) if self.mae_list else 0.0,
            "csi": float(np.mean(self.csi_list)) if self.csi_list else 0.0,
            "ssim": float(np.mean(self.ssim_list)) if self.ssim_list else 0.0,
        }
        return metrics

    def __str__(self):
        metrics = self.get_metrics()
        return (
            f"RMSE: {metrics['rmse']:.4f} | "
            f"MAE: {metrics['mae']:.4f} | "
            f"CSI: {metrics['csi']:.4f} | "
            f"SSIM: {metrics['ssim']:.4f}"
        )

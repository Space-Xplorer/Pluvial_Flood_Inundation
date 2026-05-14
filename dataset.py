"""
PyTorch Dataset class with augmentation for flood depth sequences.
Supports train/test split and normalization.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


class FloodDataset(Dataset):
    """
    PyTorch Dataset for flood depth prediction.
    Loads from pre-built dl_dataset.npz and supports augmentation.
    """

    def __init__(
        self,
        X,
        Y,
        sim_ids=None,
        augment=False,
        augmentation_config=None,
        normalize=False,
        mean_depth=None,
        std_depth=None,
    ):
        """
        Args:
            X: (N, T, C, H, W) float32 - input sequences
            Y: (N, P, 1, H, W) float32 - target depth frames
            sim_ids: (N,) int32 - simulation indices (for tracking)
            augment: bool - apply augmentation
            augmentation_config: dict - augmentation parameters
            normalize: bool - normalize depth inputs
            mean_depth: float - normalization mean
            std_depth: float - normalization std
        """
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.sim_ids = sim_ids if sim_ids is not None else np.arange(len(X))
        
        self.augment = augment
        self.augmentation_config = augmentation_config or {}
        self.normalize = normalize
        self.mean_depth = mean_depth
        self.std_depth = std_depth
        
        # Build augmentation pipeline
        self.augmentation = self._build_augmentation() if augment else None

    def _build_augmentation(self):
        """Create albumentations augmentation pipeline for spatial grids."""
        aug_list = []
        
        # Horizontal flip
        if self.augmentation_config.get("horizontal_flip", True):
            aug_list.append(A.HorizontalFlip(p=0.5))
        
        # Vertical flip
        if self.augmentation_config.get("vertical_flip", True):
            aug_list.append(A.VerticalFlip(p=0.5))
        
        # Rotation
        max_rot = self.augmentation_config.get("max_rotation", 10)
        if max_rot > 0:
            aug_list.append(A.Rotate(limit=max_rot, p=0.5, border_mode=0))
        
        # Scale
        max_scale = self.augmentation_config.get("max_scale", 0.1)
        if max_scale > 0:
            scale_lim = (1 - max_scale, 1 + max_scale)
            aug_list.append(A.RandomScale(scale_limit=max_scale, p=0.5))

        return A.ReplayCompose(aug_list + [A.Resize(256, 256)], p=0.5)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns:
            x: (T, C, H, W) - input sequence
            y: (P, 1, H, W) - target sequence
            sim_id: int - simulation index
        """
        x = self.X[idx]  # (T, C, H, W)
        y = self.Y[idx]  # (P, 1, H, W)
        sim_id = self.sim_ids[idx]
        
        # Apply augmentation to spatial dimensions (H, W)
        if self.augment and self.augmentation is not None:
            x, y = self._apply_augmentation(x, y)
        
        # Normalize depth channel (first channel)
        if self.normalize and self.mean_depth is not None and self.std_depth is not None:
            x[:, 0, :, :] = (x[:, 0, :, :] - self.mean_depth) / (self.std_depth + 1e-6)
            y[:, 0, :, :] = (y[:, 0, :, :] - self.mean_depth) / (self.std_depth + 1e-6)
        
        return x, y, torch.tensor(sim_id, dtype=torch.int32)

    def _apply_augmentation(self, x, y):
        """Apply augmentation consistently to input and target sequences."""
        # x: (T, C, H, W), y: (P, 1, H, W)
        T, C, H, W = x.shape
        P = y.shape[0]
        
        # Stack all frames (history + target) for consistent augmentation
        # We'll augment the spatial dimensions
        all_frames = torch.cat(
            [x[:, 0:1, :, :], y[:, 0:1, :, :]],  # depth only (first channel)
            dim=0
        )  # (T+P, 1, H, W)
        
        # Apply one random spatial transform and replay it across all frames.
        aug_list = []
        first_frame = np.nan_to_num(all_frames[0, 0].cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        first_frame = first_frame.astype(np.float32)

        if self.augmentation is not None:
            replayed = self.augmentation(image=first_frame)
            augmented_first = replayed["image"].astype(np.float32)
            replay = replayed["replay"]
        else:
            augmented_first = first_frame
            replay = None

        aug_list.append(torch.from_numpy(augmented_first).float())

        for t in range(1, T + P):
            frame = np.nan_to_num(all_frames[t, 0].cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
            frame = frame.astype(np.float32)
            if replay is not None:
                frame = A.ReplayCompose.replay(replay, image=frame)["image"].astype(np.float32)
            aug_list.append(torch.from_numpy(frame).float())
        
        aug_frames = torch.stack(aug_list, dim=0)  # (T+P, H, W)
        
        x_aug = x.clone()
        x_aug[:, 0, :, :] = aug_frames[:T]
        
        y_aug = y.clone()
        y_aug[:, 0, :, :] = aug_frames[T:T+P]
        
        return x_aug, y_aug

    @staticmethod
    def compute_statistics(X):
        """Compute mean and std of depth channel for normalization."""
        depth_channel = X[:, :, 0, :, :]  # (N, T, H, W)
        mean_depth = float(np.mean(depth_channel))
        std_depth = float(np.std(depth_channel))
        return mean_depth, std_depth

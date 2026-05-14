"""
Hyperparameter configuration for flood forecasting models.
All hyperparameters in one place for reproducibility and easy tuning.
"""

import os
from pathlib import Path

# ============================================================================
# DIRECTORIES
# ============================================================================
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Results" / "dataset"
DATASET_FILE = DATA_DIR / "dl_dataset.npz"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
RESULTS_DIR = PROJECT_DIR / "results"
LOGS_DIR = PROJECT_DIR / "logs"

for d in [CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DEVICE & PRECISION
# ============================================================================
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
USE_AMP = True  # Automatic Mixed Precision (faster, lower VRAM)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
GRID_SIZE = 256  # Can reduce to 128 for more memory headroom
NUM_CHANNELS = 3  # depth_history, dem, rainfall
SEQ_LENGTH = 6  # historical timesteps
PRED_LENGTH = 1  # predict 1 step ahead

# Data augmentation
AUGMENTATION_PROB = 0.5  # Probability of applying augmentation
AUGMENTATION_CONFIG = {
    "horizontal_flip": True,
    "vertical_flip": True,
    "max_rotation": 10,  # degrees
    "max_scale": 0.1,  # ±10%
}

# ============================================================================
# DATASET SPLIT (Leave-One-Out CV)
# ============================================================================
# With 5 simulations, we do 5-fold leave-one-out
# Each fold: train on 4 sims, test on 1 sim
SIMULATE_IDS = [0, 1, 2, 3, 4]  # Adjust to match your .hdf files

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 2  # Reduced to avoid CUDA OOM on 8GB laptop GPUs
EPOCHS = 150
PATIENCE = 20  # Early stopping patience
GRADIENT_CLIP = 1.0
WARMUP_EPOCHS = 5

# LR Scheduler
LR_SCHEDULER = "cosine"  # Options: "cosine", "exponential", "step"
COSINE_TMIN = 1e-5
COSINE_TMAX = 0.1

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

# ConvLSTM (Baseline)
CONVLSTM_CONFIG = {
    "input_channels": NUM_CHANNELS,
    "hidden_channels": [64, 128, 128],
    "kernel_size": 3,
    "num_layers": 3,
    "output_channels": 1,
}

# U-Net + ConvLSTM (Proposed)
UNET_CONVLSTM_CONFIG = {
    "input_channels": NUM_CHANNELS,
    "unet_channels": [64, 128, 256],
    "convlstm_hidden": 128,
    "convlstm_layers": 2,
    "output_channels": 1,
}

# ============================================================================
# LOSS FUNCTION
# ============================================================================
LOSS_FN = "mse"  # Options: "mse", "l1", "huber"
LOSS_WEIGHTS = {
    "depth": 1.0,  # Primary target
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================
# Thresholds for flood detection
FLOOD_THRESHOLD = 0.3  # meters (used for CSI)

METRICS_TO_TRACK = [
    "rmse",
    "mae",
    "csi",  # Critical Success Index
    "ssim",  # Structural Similarity
]

# ============================================================================
# CHECKPOINTING & LOGGING
# ============================================================================
SAVE_BEST_MODEL = True
SAVE_FREQ = 10  # Save checkpoint every N epochs
LOG_FREQ = 50  # Print metrics every N batches
TENSORBOARD_LOG = True

# ============================================================================
# RANDOM SEEDS (Reproducibility)
# ============================================================================
SEED = 42

# ============================================================================
# NUMERICAL STABILITY
# ============================================================================
# Clip predictions to realistic flood depths
MIN_DEPTH = 0.0
MAX_DEPTH = 10.0  # Adjust based on your simulation domain

# Normalization (compute on training data)
NORMALIZE_INPUTS = True
MEAN_DEPTH = None  # Computed during training
STD_DEPTH = None  # Computed during training

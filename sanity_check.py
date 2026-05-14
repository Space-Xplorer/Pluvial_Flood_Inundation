"""
Quick sanity check: list all training scripts and verify dataset is ready.
"""

from pathlib import Path
import sys

import config


def check_dataset():
    """Verify dataset exists and show stats."""
    if not config.DATASET_FILE.exists():
        print("[ERROR] Dataset not found. Run the preprocessing pipeline first:")
        print("  1. python extract_hdf.py")
        print("  2. python mesh_to_grid.py")
        print("  3. python build_dl_dataset.py")
        return False
    
    print(f"✓ Dataset found: {config.DATASET_FILE}")
    return True


def check_directories():
    """Verify output directories exist."""
    for d in [config.CHECKPOINT_DIR, config.RESULTS_DIR, config.LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory ready: {d}")


def list_scripts():
    """Show available training scripts."""
    scripts = [
        "train_convlstm.py",
        "train_unet_convlstm.py",
        "evaluate.py",
    ]
    
    print("\n[AVAILABLE TRAINING SCRIPTS]")
    for script in scripts:
        script_path = config.PROJECT_DIR / script
        if script_path.exists():
            print(f"  ✓ {script}")
        else:
            print(f"  ✗ {script} (NOT FOUND)")


def show_config():
    """Display key hyperparameters."""
    print("\n[TRAINING CONFIG]")
    print(f"  Device: {config.DEVICE}")
    print(f"  Grid Size: {config.GRID_SIZE}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Sequence Length: {config.SEQ_LENGTH}")
    print(f"  Prediction Length: {config.PRED_LENGTH}")


def main():
    """Sanity check."""
    print("\n" + "="*70)
    print("TRAINING SANITY CHECK")
    print("="*70 + "\n")
    
    ok = check_dataset()
    if not ok:
        sys.exit(1)
    
    check_directories()
    list_scripts()
    show_config()
    
    print("\n" + "="*70)
    print("Setup OK. Ready to train.")
    print("\nRun models:")
    print("  python train_convlstm.py")
    print("  python train_unet_convlstm.py")
    print("\nThen compare:")
    print("  python evaluate.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

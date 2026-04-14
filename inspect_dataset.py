from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_FILE = PROJECT_DIR / "Results" / "dataset" / "dl_dataset.npz"


def main():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_FILE}")

    data = np.load(DATASET_FILE, allow_pickle=True)
    print(f"Dataset: {DATASET_FILE}")
    print(f"Keys: {list(data.keys())}")

    X = data["X"]
    Y = data["Y"]

    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    if "channel_names" in data:
        print(f"Channels: {list(data['channel_names'])}")
    if "seq_len" in data:
        print(f"seq_len: {int(data['seq_len'][0])}")
    if "pred_len" in data:
        print(f"pred_len: {int(data['pred_len'][0])}")
    if "stride" in data:
        print(f"stride: {int(data['stride'][0])}")


if __name__ == "__main__":
    main()
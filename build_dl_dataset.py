from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
GRID_DIR = PROJECT_DIR / "Results" / "gridded"
DATASET_DIR = PROJECT_DIR / "Results" / "dataset"
EXTERNAL_DIR = PROJECT_DIR / "Results" / "external"

# Sequence settings
SEQ_LEN = 6
PRED_LEN = 1
STRIDE = 1


def numeric_suffix(path_obj):
    stem = path_obj.stem
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return 10**9


def load_optional_dem(grid_shape):
    dem_path = EXTERNAL_DIR / "dem_grid.npy"
    if dem_path.exists():
        dem = np.load(dem_path)
        if dem.shape != grid_shape:
            raise ValueError(
                f"DEM shape mismatch. Expected {grid_shape}, got {dem.shape}: {dem_path}"
            )
        return dem.astype(np.float32), str(dem_path)

    return np.zeros(grid_shape, dtype=np.float32), None


def load_optional_rainfall(sim_count, max_timesteps):
    rain_path = EXTERNAL_DIR / "rainfall_timeseries.npy"
    if not rain_path.exists():
        return np.zeros((sim_count, max_timesteps), dtype=np.float32), None

    rainfall = np.load(rain_path)
    if rainfall.ndim == 1:
        rainfall = np.tile(rainfall[None, :], (sim_count, 1))
    if rainfall.ndim != 2:
        raise ValueError(
            "rainfall_timeseries.npy must be 1D (time,) or 2D (sim, time)"
        )
    if rainfall.shape[0] not in (1, sim_count):
        raise ValueError(
            f"Rainfall simulation axis mismatch: expected 1 or {sim_count}, got {rainfall.shape[0]}"
        )

    if rainfall.shape[0] == 1:
        rainfall = np.tile(rainfall, (sim_count, 1))

    if rainfall.shape[1] < max_timesteps:
        pad = np.zeros((sim_count, max_timesteps - rainfall.shape[1]), dtype=rainfall.dtype)
        rainfall = np.concatenate([rainfall, pad], axis=1)

    return rainfall[:, :max_timesteps].astype(np.float32), str(rain_path)


def build_samples(depth_grid, dem_grid, rainfall_series, sim_idx):
    t_steps, h, w = depth_grid.shape
    max_start = t_steps - SEQ_LEN - PRED_LEN + 1
    if max_start <= 0:
        return [], [], [], []

    x_list = []
    y_list = []
    sim_id_list = []
    start_t_list = []

    dem_seq = np.repeat(dem_grid[None, :, :], SEQ_LEN, axis=0)

    for start_t in range(0, max_start, STRIDE):
        hist = depth_grid[start_t : start_t + SEQ_LEN]
        target = depth_grid[start_t + SEQ_LEN : start_t + SEQ_LEN + PRED_LEN]

        rain_seq_1d = rainfall_series[start_t : start_t + SEQ_LEN]
        rain_seq = np.repeat(rain_seq_1d[:, None, None], h, axis=1)
        rain_seq = np.repeat(rain_seq, w, axis=2)

        x = np.stack([hist, dem_seq, rain_seq], axis=1).astype(np.float32)
        y = target[:, None, :, :].astype(np.float32)

        x_list.append(x)
        y_list.append(y)
        sim_id_list.append(sim_idx)
        start_t_list.append(start_t)

    return x_list, y_list, sim_id_list, start_t_list


def main():
    if not GRID_DIR.exists():
        raise FileNotFoundError(f"Gridded directory not found: {GRID_DIR}")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    depth_files = sorted(GRID_DIR.glob("depth_grid_*.npy"), key=numeric_suffix)
    if not depth_files:
        raise FileNotFoundError("No depth_grid_*.npy found. Run mesh_to_grid.py first.")

    sim_indices = [numeric_suffix(path_obj) for path_obj in depth_files]
    all_depth = [np.load(path_obj).astype(np.float32) for path_obj in depth_files]

    grid_shape = all_depth[0].shape[1:]
    for sim_idx, arr in zip(sim_indices, all_depth):
        if arr.ndim != 3:
            raise ValueError(f"depth_grid_{sim_idx}.npy must be 3D (time, H, W)")
        if arr.shape[1:] != grid_shape:
            raise ValueError("All depth_grid files must have identical (H, W)")

    min_timesteps = min(arr.shape[0] for arr in all_depth)
    dem_grid, dem_source = load_optional_dem(grid_shape)
    rainfall_all, rain_source = load_optional_rainfall(len(all_depth), min_timesteps)

    x_samples = []
    y_samples = []
    sim_ids = []
    start_ts = []

    for i, (sim_idx, depth_grid) in enumerate(zip(sim_indices, all_depth)):
        depth_grid = depth_grid[:min_timesteps]
        rainfall_series = rainfall_all[i]

        x_list, y_list, sim_id_list, start_t_list = build_samples(
            depth_grid, dem_grid, rainfall_series, sim_idx
        )

        x_samples.extend(x_list)
        y_samples.extend(y_list)
        sim_ids.extend(sim_id_list)
        start_ts.extend(start_t_list)

        print(
            f"Simulation {sim_idx}: timesteps={depth_grid.shape[0]}, samples={len(x_list)}"
        )

    if not x_samples:
        raise ValueError("No samples were built. Reduce SEQ_LEN/PRED_LEN or check data.")

    X = np.stack(x_samples, axis=0)
    Y = np.stack(y_samples, axis=0)
    sim_ids = np.array(sim_ids, dtype=np.int32)
    start_ts = np.array(start_ts, dtype=np.int32)
    channel_names = np.array(["depth_history", "dem", "rainfall"], dtype=object)

    out_file = DATASET_DIR / "dl_dataset.npz"
    np.savez_compressed(
        out_file,
        X=X,
        Y=Y,
        sim_id=sim_ids,
        start_t=start_ts,
        channel_names=channel_names,
        seq_len=np.array([SEQ_LEN], dtype=np.int32),
        pred_len=np.array([PRED_LEN], dtype=np.int32),
        stride=np.array([STRIDE], dtype=np.int32),
        dem_source=np.array([dem_source if dem_source else "zeros"], dtype=object),
        rainfall_source=np.array([rain_source if rain_source else "zeros"], dtype=object),
    )

    print(f"Saved dataset: {out_file}")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")


if __name__ == "__main__":
    main()
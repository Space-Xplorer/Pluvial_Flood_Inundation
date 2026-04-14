from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree


PROJECT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_DIR / "Results" / "processed"
GRID_DIR = PROJECT_DIR / "Results" / "gridded"
GRID_SIZE = 256


def numeric_suffix(path_obj):
    stem = path_obj.stem
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return 10**9


def build_grid(coords, grid_size):
    x = coords[:, 0]
    y = coords[:, 1]

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    gx = np.linspace(x_min, x_max, grid_size, dtype=np.float64)
    gy = np.linspace(y_min, y_max, grid_size, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(gx, gy)

    return grid_x, grid_y, (x_min, x_max, y_min, y_max)


def interpolate_frame(values, coords, grid_x, grid_y):
    linear = griddata(coords, values, (grid_x, grid_y), method="linear")
    nearest = griddata(coords, values, (grid_x, grid_y), method="nearest")
    return np.where(np.isnan(linear), nearest, linear)


def build_coverage_mask(coords, grid_x, grid_y):
    tree = cKDTree(coords)
    query_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    dists, _ = tree.query(query_points, k=1)

    dx = float(np.mean(np.diff(grid_x[0, :])))
    dy = float(np.mean(np.diff(grid_y[:, 0])))
    threshold = 1.5 * np.sqrt(dx * dx + dy * dy)

    mask = (dists.reshape(grid_x.shape) <= threshold).astype(np.uint8)
    return mask


def convert_one(sim_idx):
    depth_path = PROCESSED_DIR / f"depth_{sim_idx}.npy"
    wse_path = PROCESSED_DIR / f"wse_{sim_idx}.npy"
    coords_path = PROCESSED_DIR / f"coords_{sim_idx}.npy"

    if not depth_path.exists() or not coords_path.exists():
        raise FileNotFoundError(f"Missing required files for index {sim_idx}")

    depth = np.load(depth_path)
    coords = np.load(coords_path)

    if depth.ndim != 2:
        raise ValueError(f"depth_{sim_idx}.npy must be 2D (time, cells)")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords_{sim_idx}.npy must be 2D with shape (cells, 2)")
    if depth.shape[1] != coords.shape[0]:
        raise ValueError(f"Cell count mismatch for index {sim_idx}: depth vs coords")

    t_steps = depth.shape[0]
    grid_x, grid_y, bounds = build_grid(coords, GRID_SIZE)
    depth_grid = np.zeros((t_steps, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    for t in range(t_steps):
        frame = interpolate_frame(depth[t], coords, grid_x, grid_y)
        depth_grid[t] = frame.astype(np.float32)

    np.save(GRID_DIR / f"depth_grid_{sim_idx}.npy", depth_grid)

    if wse_path.exists():
        wse = np.load(wse_path)
        if wse.shape == depth.shape:
            wse_grid = np.zeros((t_steps, GRID_SIZE, GRID_SIZE), dtype=np.float32)
            for t in range(t_steps):
                frame = interpolate_frame(wse[t], coords, grid_x, grid_y)
                wse_grid[t] = frame.astype(np.float32)
            np.save(GRID_DIR / f"wse_grid_{sim_idx}.npy", wse_grid)

    return grid_x, grid_y, bounds, coords


def main():
    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(f"Processed directory not found: {PROCESSED_DIR}")

    GRID_DIR.mkdir(parents=True, exist_ok=True)

    depth_files = sorted(PROCESSED_DIR.glob("depth_*.npy"), key=numeric_suffix)
    if not depth_files:
        raise FileNotFoundError("No depth_*.npy files found. Run extract_hdf.py first.")

    seen_indices = [numeric_suffix(path_obj) for path_obj in depth_files]
    print(f"Found {len(seen_indices)} simulation(s): {seen_indices}")

    grid_x = grid_y = bounds = coords = None
    failures = []

    for sim_idx in seen_indices:
        try:
            print(f"Converting simulation index {sim_idx} to {GRID_SIZE}x{GRID_SIZE} grid")
            grid_x, grid_y, bounds, coords = convert_one(sim_idx)
        except Exception as exc:
            failures.append((sim_idx, str(exc)))
            print(f"ERROR in simulation {sim_idx}: {exc}")

    if grid_x is not None and grid_y is not None and coords is not None:
        mask = build_coverage_mask(coords, grid_x, grid_y)
        np.save(GRID_DIR / "coverage_mask.npy", mask)
        np.savez_compressed(
            GRID_DIR / "grid_meta.npz",
            grid_x=grid_x.astype(np.float32),
            grid_y=grid_y.astype(np.float32),
            bounds=np.array(bounds, dtype=np.float64),
            grid_size=np.array([GRID_SIZE], dtype=np.int32),
        )

    if failures:
        print("\nCompleted with errors:")
        for sim_idx, err in failures:
            print(f" - simulation {sim_idx}: {err}")
    else:
        print("\nDONE")


if __name__ == "__main__":
    main()
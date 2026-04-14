import os
from pathlib import Path

import h5py
import numpy as np


# This script is configured for the current repository layout.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Results"
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_dataset(area, candidates):
    """Return the first matching dataset key from candidates."""
    keys_lower = {k.lower(): k for k in area.keys()}
    for candidate in candidates:
        key = keys_lower.get(candidate.lower())
        if key is not None:
            return key
    return None


def extract(file_path, idx):
    print(f"\nProcessing: {file_path.name}")

    with h5py.File(file_path, "r") as f:
        try:
            base = f["Results"]["Unsteady"]["Output"]["Output Blocks"]["Base Output"][
                "2D Flow Areas"
            ]
        except KeyError as exc:
            raise KeyError(
                f"Expected HEC-RAS path not found in {file_path.name}: {exc}"
            ) from exc

        area_names = list(base.keys())
        if not area_names:
            raise ValueError(f"No 2D flow areas found in {file_path.name}")

        area_name = area_names[0]
        print(f"Area: {area_name}")
        area = base[area_name]
        print(f"Available keys: {list(area.keys())}")

        depth_key = find_dataset(area, ["Depth"])
        wse_key = find_dataset(area, ["Water Surface", "Water Surface Elevation"])
        coords_key = find_dataset(
            area,
            [
                "Cells Center Coordinate",
                "Cell Center Coordinate",
                "Cell Center Coordinates",
            ],
        )

        missing = [
            name
            for name, key in {
                "Depth": depth_key,
                "Water Surface": wse_key,
                "Cells Center Coordinate": coords_key,
            }.items()
            if key is None
        ]
        if missing:
            raise KeyError(
                f"Missing required dataset(s) in {file_path.name}: {', '.join(missing)}"
            )

        depth = area[depth_key][:]  # (time, cells)
        wse = area[wse_key][:]  # (time, cells)
        coords = area[coords_key][:]  # (cells, 2)

    print(f"depth shape: {depth.shape}")
    print(f"coords shape: {coords.shape}")

    np.save(OUT_DIR / f"depth_{idx}.npy", depth)
    np.save(OUT_DIR / f"wse_{idx}.npy", wse)
    np.save(OUT_DIR / f"coords_{idx}.npy", coords)


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    files = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() == ".hdf")
    if not files:
        raise FileNotFoundError(f"No .hdf files found in: {DATA_DIR}")

    print(f"Found {len(files)} HDF file(s) in: {DATA_DIR}")
    print(f"Output directory: {OUT_DIR}")

    failures = []
    for i, file_path in enumerate(files):
        try:
            extract(file_path, i)
        except Exception as exc:  # Keep processing remaining files.
            failures.append((file_path.name, str(exc)))
            print(f"ERROR in {file_path.name}: {exc}")

    if failures:
        print("\nCompleted with errors:")
        for name, err in failures:
            print(f" - {name}: {err}")
    else:
        print("\nDONE")


if __name__ == "__main__":
    main()
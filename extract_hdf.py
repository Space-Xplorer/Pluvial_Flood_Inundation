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


def get_dataset_by_path(file_handle, path_parts):
    current = file_handle
    for part in path_parts:
        if part not in current:
            return None
        current = current[part]
    return current if isinstance(current, h5py.Dataset) else None


def extract(file_path, idx):
    print(f"\nProcessing: {file_path.name}")

    with h5py.File(file_path, "r") as f:
        try:
            geometry_areas = f["Geometry"]["2D Flow Areas"]
        except KeyError as exc:
            raise KeyError(
                f"Expected geometry path not found in {file_path.name}: {exc}"
            ) from exc

        area_names = [
            key
            for key in geometry_areas.keys()
            if isinstance(geometry_areas[key], h5py.Group)
        ]
        if not area_names:
            raise ValueError(f"No 2D flow areas found in {file_path.name}")

        area_name = area_names[0]
        print(f"Area: {area_name}")
        area = geometry_areas[area_name]
        print(f"Available keys: {list(area.keys())}")

        depth_key = find_dataset(area, ["Depth"])
        min_elev_key = find_dataset(
            area,
            ["Cells Minimum Elevation", "Cell Minimum Elevation"],
        )
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
                "Cells Center Coordinate": coords_key,
                "Cells Minimum Elevation": min_elev_key,
            }.items()
            if key is None
        ]
        if missing:
            raise KeyError(
                f"Missing required dataset(s) in {file_path.name}: {', '.join(missing)}"
            )

        coords = area[coords_key][:]  # (cells, 2)
        min_elev = area[min_elev_key][:].astype(np.float32)  # (cells,)

        if depth_key is not None:
            depth = area[depth_key][:].astype(np.float32)  # (time, cells)
            wse = None
        else:
            wse_path = get_dataset_by_path(
                f,
                [
                    "Results",
                    "Unsteady",
                    "Output",
                    "Output Blocks",
                    "Base Output",
                    "Unsteady Time Series",
                    "2D Flow Areas",
                    area_name,
                    "Water Surface",
                ],
            )
            if wse_path is None:
                raise KeyError(
                    f"Water Surface dataset not found for {file_path.name} in Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{area_name}/Water Surface"
                )
            wse = wse_path[:].astype(np.float32)  # (time, cells)
            depth = np.maximum(wse - min_elev[None, :], 0.0).astype(np.float32)

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        if wse is not None:
            wse = np.nan_to_num(wse, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"depth shape: {depth.shape}")
    print(f"coords shape: {coords.shape}")

    np.save(OUT_DIR / f"depth_{idx}.npy", depth)
    if wse is not None:
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
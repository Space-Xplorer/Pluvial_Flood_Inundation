# Flood DL Dataset Pipeline

## 1) Project Goal

Convert HEC-RAS 2D simulation outputs from HDF format into deep-learning-ready tensors.

The implemented pipeline performs:

1. HDF extraction (time x mesh-cells arrays)
2. Mesh-to-grid conversion (regular 256 x 256 image-like tensors)
3. Sequence dataset creation for model training
4. Dataset inspection utilities

This is the exact bridge from simulation outputs to ML inputs/targets.

---

## 2) Current Workspace Layout

Input data expected in:

- Results/
  - VNR_Flood_Model.p01.hdf
  - VNR_Flood_Model.p02.hdf
  - VNR_Flood_Model.p18.hdf
  - VNR_Flood_Model.p19.hdf
  - VNR_Flood_Model.p20.hdf

New scripts added at project root:

- extract_hdf.py
- mesh_to_grid.py
- build_dl_dataset.py
- inspect_dataset.py
- requirements.txt
- PROJECT_PIPELINE.md

Generated outputs will be created automatically in:

- Results/processed/
- Results/gridded/
- Results/dataset/

Optional external inputs (if available):

- Results/external/dem_grid.npy
- Results/external/rainfall_timeseries.npy

If these optional files are missing, the pipeline uses zero placeholders for those channels.

---

## 3) File-by-File Breakdown

## extract_hdf.py

Purpose:
- Read each HDF file under Results/
- Safely navigate the HEC-RAS hierarchy
- Extract Depth, Water Surface, and Cell Center Coordinates
- Save NumPy arrays per simulation index

Key behavior:
- Dynamically picks first 2D flow area
- Validates required datasets exist
- Prints shape checks:
  - depth shape: (time, cells)
  - coords shape: (cells, 2)
- Continues processing remaining files even if one fails

Outputs:
- Results/processed/depth_i.npy
- Results/processed/wse_i.npy
- Results/processed/coords_i.npy

---

## mesh_to_grid.py

Purpose:
- Convert irregular mesh values into regular image grid tensors (256 x 256)

How it works:
- Loads depth_i.npy and coords_i.npy from Results/processed/
- Builds a regular coordinate grid from min/max coordinate extents
- Interpolates each timestep using:
  - linear interpolation
  - nearest-neighbor fill for NaNs
- Saves gridded depth (and wse if available)
- Builds and saves a coverage mask using nearest-neighbor distance threshold

Outputs:
- Results/gridded/depth_grid_i.npy
- Results/gridded/wse_grid_i.npy (if water surface exists)
- Results/gridded/coverage_mask.npy
- Results/gridded/grid_meta.npz

Notes:
- Grid size is currently fixed by GRID_SIZE = 256.
- If you need 128 or 512, change the constant in mesh_to_grid.py.

---

## build_dl_dataset.py

Purpose:
- Build sequence training samples for deep learning from gridded depth data

Default sequence config:
- SEQ_LEN = 6
- PRED_LEN = 1
- STRIDE = 1

Input channels created per timestep:
1. Depth history (from depth_grid)
2. DEM (from Results/external/dem_grid.npy or zeros)
3. Rainfall (from Results/external/rainfall_timeseries.npy or zeros)

Target:
- Future depth frame(s) after the history window

Tensor formats:
- X: (samples, seq_len, 3, H, W)
- Y: (samples, pred_len, 1, H, W)

Additional metadata saved:
- sim_id
- start_t
- channel_names
- sequence settings
- source info for DEM/rainfall

Output:
- Results/dataset/dl_dataset.npz

---

## inspect_dataset.py

Purpose:
- Quick sanity check for generated dataset

Prints:
- Available keys in npz
- X and Y shapes
- channel names and sequence settings

---

## requirements.txt

Dependencies used by scripts:
- h5py
- numpy
- scipy

---

## 4) End-to-End Data Flow

Step 1:
- Raw HDF -> processed arrays
- Script: extract_hdf.py

Step 2:
- Processed arrays -> regular image grids
- Script: mesh_to_grid.py

Step 3:
- Gridded arrays -> DL training dataset
- Script: build_dl_dataset.py

Step 4:
- Verify output tensor structure
- Script: inspect_dataset.py

---

## 5) Run Order

Run in this order:

1. python extract_hdf.py
2. python mesh_to_grid.py
3. python build_dl_dataset.py
4. python inspect_dataset.py

---

## 6) Expected Output Shapes

Typical extraction stage:
- depth: (T, Ncells)
- coords: (Ncells, 2)

After gridding:
- depth_grid_i: (T, 256, 256)

After dataset build:
- X: (Nsamples, SEQ_LEN, 3, 256, 256)
- Y: (Nsamples, PRED_LEN, 1, 256, 256)

Where:
- T = number of time steps in a simulation
- Ncells = number of mesh cells
- Nsamples depends on T, SEQ_LEN, PRED_LEN, STRIDE, and number of simulations

---

## 7) Robustness and Error Handling Added

- Safe navigation of HDF hierarchy and clear path errors
- Required dataset validation with descriptive failure messages
- Per-file isolation so one bad file does not stop full run
- Shape validations at each stage
- Interpolation fallback to nearest when linear leaves NaNs
- Optional input handling (DEM/rainfall) with zero fallbacks

---

## 8) What Is Ready vs What Is Placeholder

Ready now:
- Full preprocessing pipeline from HDF to DL tensors
- Reproducible output structure and metadata
- Sanity inspection script

Placeholder behavior by design:
- DEM channel defaults to zeros if dem_grid.npy not provided
- Rainfall channel defaults to zeros if rainfall_timeseries.npy not provided

To make physically richer training inputs, provide:
- Results/external/dem_grid.npy shaped (256, 256)
- Results/external/rainfall_timeseries.npy shaped (time,) or (simulations, time)

---

## 9) Suggested Next Technical Step

Now that data is DL-ready, the next milestone is model training, for example:

- U-Net style encoder-decoder for one-step depth prediction
- ConvLSTM for temporal sequence forecasting

The generated dataset format already supports both approaches.

---

## 10) Implementation Summary

What was added to this codebase in this update:

1. Completed extraction stage script (already present): extract_hdf.py
2. Added mesh-to-grid conversion script: mesh_to_grid.py
3. Added sequence dataset builder script: build_dl_dataset.py
4. Added dataset inspector: inspect_dataset.py
5. Added dependency file: requirements.txt
6. Added this full technical documentation: PROJECT_PIPELINE.md

No scripts were executed in this update.
# Flood Depth Prediction for Pluvial Inundation

Minor Project for BTech in CSE-AIML

This repository contains an end-to-end deep learning workflow for flood depth prediction using HEC-RAS 2D simulation outputs. The project converts raw HDF simulation files into deep-learning-ready tensors, trains two forecasting models, and compares their performance with flood-specific evaluation metrics.

## Project Objective

The goal of this project is to predict flood depth maps from simulation history using deep learning. The pipeline is designed for a small dataset of flood simulations and is organized around two stages:

1. Data preprocessing from HDF to regular grid tensors.
2. Model training and evaluation using leave-one-out cross-validation.

The repository also includes a batch utility for generating HEC-RAS unsteady flow input files.

## What This Project Contains

- HDF extraction from HEC-RAS simulation outputs
- Mesh-to-grid interpolation into 256 x 256 raster maps
- Sequence dataset creation for temporal learning
- ConvLSTM baseline training
- U-Net + ConvLSTM proposed model training
- Cross-validation evaluation and comparison plots
- Optional batch generation of HEC-RAS unsteady flow files

## Repository Structure

```text
DL_Training/
├── build_dl_dataset.py        # Build sequence tensors from gridded outputs
├── config.py                  # Central training and data configuration
├── dataset.py                 # PyTorch dataset + augmentation
├── evaluate.py                # Compare trained models and export tables
├── extract_hdf.py             # Extract depth, WSE, and coordinates from HDF files
├── inspect_dataset.py         # Quick dataset inspection utility
├── main.py                    # Entry point for batch HEC-RAS run generation
├── mesh_to_grid.py            # Convert irregular mesh data into regular grids
├── metrics.py                 # RMSE, MAE, CSI, SSIM metrics
├── models.py                  # ConvLSTM and U-Net + ConvLSTM architectures
├── run_batch.py               # Batch generator for unsteady flow input files
├── sanity_check.py            # Verify dataset and training setup
├── train_convlstm.py          # Baseline training script
├── train_unet_convlstm.py     # Proposed model training script
├── TRAINING_GUIDE.md          # Stage 2 training instructions
├── PROJECT_PIPELINE.md        # Stage 1 preprocessing documentation
├── Results/                   # Raw and processed simulation data
├── checkpoints/               # Saved model checkpoints
├── logs/                      # Optional TensorBoard logs
└── results/                   # Training outputs and comparison files
```

## Data Pipeline

The preprocessing pipeline transforms the raw flood simulations into tensors that can be used directly for model training.

```text
Raw HDF files
  -> extract_hdf.py
Processed NumPy arrays
  -> mesh_to_grid.py
Regular 256 x 256 grids
  -> build_dl_dataset.py
Sequence dataset in dl_dataset.npz
```

### Input Data

Place the raw HEC-RAS HDF files inside `Results/`:

- `Results/VNR_Flood_Model.p01.hdf`
- `Results/VNR_Flood_Model.p02.hdf`
- `Results/VNR_Flood_Model.p18.hdf`
- `Results/VNR_Flood_Model.p19.hdf`
- `Results/VNR_Flood_Model.p20.hdf`

Optional external inputs can also be supplied:

- `Results/external/dem_grid.npy`
- `Results/external/rainfall_timeseries.npy`

If these optional files are missing, the pipeline falls back to zero-valued placeholder channels.

## Dataset Format

The generated dataset is saved at `Results/dataset/dl_dataset.npz` and follows this structure:

- `X`: input sequences with shape `(N, T, 3, H, W)`
- `Y`: target sequences with shape `(N, P, 1, H, W)`
- `sim_id`: simulation index for each sample

Default configuration used by the training code:

- Sequence length: 6
- Prediction length: 1
- Grid size: 256 x 256
- Input channels: depth history, DEM, rainfall

## Models

Two architectures are implemented and compared:

- ConvLSTM baseline
- U-Net + ConvLSTM proposed model

Both models are trained with 5-fold leave-one-out cross-validation, where each fold holds out one simulation for testing and uses the remaining four for training.

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The project uses packages such as PyTorch, NumPy, SciPy, pandas, Matplotlib, Albumentations, and TorchMetrics.

## Setup

1. Clone or open the repository in VS Code.
2. Install the dependencies.
3. Make sure the raw HDF files are present in `Results/`.
4. Run the preprocessing pipeline to create `Results/dataset/dl_dataset.npz`.
5. Use the sanity check before starting training.

## How To Run

### 1. Optional: Generate HEC-RAS unsteady flow files

This project includes a batch utility for generating `.uXX` files from a template:

```bash
python main.py --num-runs 10 --start-suffix 31 --template-suffix 1
```

Useful flags:

- `--num-runs`: number of flow files to generate
- `--start-suffix`: first output suffix, such as `31` for `u31`
- `--template-suffix`: suffix of the template flow file
- `--no-project-register`: skip writing the generated file into the `.prj` project

### 2. Extract HDF Data

```bash
python extract_hdf.py
```

This reads each HDF file, extracts depth, water surface elevation, and cell coordinates, and stores them as NumPy arrays in `Results/processed/`.

### 3. Convert Mesh Data to Grids

```bash
python mesh_to_grid.py
```

This interpolates the irregular mesh onto a regular 256 x 256 grid and saves outputs to `Results/gridded/`.

### 4. Build the Deep Learning Dataset

```bash
python build_dl_dataset.py
```

This creates `Results/dataset/dl_dataset.npz` with training samples, targets, and metadata.

### 5. Inspect the Dataset

```bash
python inspect_dataset.py
```

This prints the dataset keys, tensor shapes, and sequence settings for a quick sanity check.

### 6. Verify the Training Setup

```bash
python sanity_check.py
```

This checks that the dataset exists, output directories are ready, and the training scripts are available.

### 7. Train the Baseline Model

```bash
python train_convlstm.py
```

This trains the ConvLSTM baseline with leave-one-out cross-validation and stores checkpoints in `checkpoints/`.

### 8. Train the Proposed Model

```bash
python train_unet_convlstm.py
```

This trains the U-Net + ConvLSTM model with the same fold split and saves results in `results/`.

### 9. Compare Both Models

```bash
python evaluate.py
```

This generates the comparison table, per-fold summary, comparison plot, and LaTeX table for reporting.

## Output Files

After training, the main generated artifacts are:

- `checkpoints/convlstm_fold*.pt`
- `checkpoints/unet_convlstm_fold*.pt`
- `results/convlstm_cv_results.pkl`
- `results/unet_convlstm_cv_results.pkl`
- `results/model_comparison.png`
- `results/comparison_table.tex`

## Metrics Used

The project evaluates flood prediction using metrics that matter for both numerical accuracy and flood-extent quality:

- RMSE: primary regression error
- MAE: average absolute error
- CSI: critical success index for flood detection
- SSIM: structural similarity for spatial pattern preservation

## Key Configuration

Most training and data settings are centralized in `config.py`, including:

- Grid size
- Sequence length
- Batch size
- Learning rate
- Early stopping patience
- Flood threshold for CSI
- Model architecture parameters

If you want to reduce memory use, lower `BATCH_SIZE` or `GRID_SIZE` in `config.py`.

## Notes For This Repository

- The raw and intermediate simulation data live in `Results/`.
- The training scripts write model outputs to `checkpoints/` and `results/`.
- `sanity_check.py` is the fastest way to confirm the project is ready before a long training run.
- This repository is set up for research-style evaluation, not a generic image classification task.

## Academic Context

This project was developed as a minor project for a BTech in CSE-AIML degree and focuses on flood inundation forecasting using deep learning. It demonstrates a full workflow from hydrodynamic simulation data to model comparison and reporting.

## License

See `LICENSE` for details.
# MRI Reconstruction with VarNet and Pixelwise Uncertainty

This repository contains a clean PyTorch Lightning implementation for **2D accelerated MRI reconstruction** using variational networks, together with a separate pixelwise uncertainty estimation pipeline.

The codebase is intentionally focused on the stable reconstruction and uncertainty workflows:

- End-to-end VarNet (`e2e_varnet`)
- Feature-image VarNet (`fi_varnet`)
- Fixed Cartesian undersampling masks
- Pixelwise uncertainty estimation using quantile regression and conformal calibration

The repository does **not** include the experimental 3D/multi-slice pipeline, learnable sampling masks, or YOLO/perceptual-loss training path.

---

## Repository structure

```text
data/
  data_module.py              Lightning data module and fastMRI-style slice dataset
  data_transforms.py          VarNet slice transform
  undersampling_patterns.py   Fixed Cartesian mask definitions

varnets/
  VarNet.py                   E2E VarNet, FI VarNet, and uncertainty network
  VarNet_module.py            Lightning reconstruction module
  Network_module.py           Validation, testing, and logging utilities
  uncertainty_module.py       Uncertainty training, calibration, and testing module

utilities/
  functions.py                FFTs, complex operations, cropping, and reconstruction saving
  losses.py                   SSIM and pinball losses
  evaluation.py               MSE, NMSE, PSNR, and SSIM evaluation

runner.py                     Reconstruction training/testing entry point
runner_uncertainty.py         Uncertainty training/calibration/testing entry point
batch_run.sh                  SLURM launcher for reconstruction and uncertainty jobs
```

---

## Reconstruction models

The reconstruction pipeline supports two 2D multicoil VarNet backbones.

### `e2e_varnet`

Standard end-to-end variational network reconstruction:

```text
undersampled k-space
    -> sensitivity estimation
    -> cascaded image-domain regularization and data consistency
    -> reconstructed magnitude image
```

### `fi_varnet`

Feature-image VarNet reconstruction. This follows the same broad reconstruction structure but performs part of the regularization in a learned feature-image domain.

The reconstruction loss is SSIM loss.

---

## Pixelwise uncertainty estimation

The uncertainty model is trained after a reconstruction model has already been trained.

The reconstruction model is frozen, and a separate uncertainty network predicts lower and upper bounds around the VarNet reconstruction. These bounds are trained using pinball loss and then calibrated with conformal prediction.

The workflow is:

```text
train VarNet
    -> test VarNet and save reconstructions
    -> freeze VarNet
    -> train uncertainty model
    -> calibrate uncertainty intervals
    -> test uncertainty model
    -> append uncertainty maps to reconstruction HDF5 files
```

During uncertainty testing, each reconstruction HDF5 file is updated to contain:

```text
reconstruction
uncertainty_map
```

---

## Dataset format

The code expects fastMRI-style HDF5 files.

A typical folder structure is:

```text
/path/to/data/
  knee/
    multicoil_train/
    multicoil_val/
    multicoil_cal/
    multicoil_test/

  brain/
    multicoil_train/
    multicoil_val/
    multicoil_cal/
    multicoil_test/
```

Each HDF5 file should contain multicoil k-space data and the corresponding RSS reconstruction target.

---

## Before running: update `batch_run.sh`

`batch_run.sh` currently contains local paths from the original development environment. Before running the code on another machine or cluster, update the path section near the top of `batch_run.sh`.

The main paths to change are:

```bash
HOME_DIR=/path/to/project_or_data_root
VARNET_DIR=/path/to/this/repository
LOG_PATH=/path/to/output/logs
DATA_DIR=/path/to/data/${ANATOMY}/
VN_TRAIN_PATH=/path/to/data/${ANATOMY}/multicoil_train/
VN_VAL_PATH=/path/to/data/${ANATOMY}/multicoil_val/
VN_CAL_PATH=/path/to/data/${ANATOMY}/multicoil_cal/
VN_TEST_PATH=/path/to/data/${ANATOMY}/multicoil_test/
```

Then the default dataset paths are built automatically from `HOME_DIR` and `ANATOMY`.

You may also need to edit the SLURM header in `batch_run.sh` depending on your cluster:

```bash
#SBATCH --partition=radiology
#SBATCH --cpus-per-task=10
#SBATCH --time=3-00:00:00
#SBATCH --mem=300G
#SBATCH --gres=gpu:a100:1
```

For example, change the partition name, GPU type, memory, or wall time to match your system.

---

## Main `batch_run.sh` variables

Common experiment controls:

```bash
ANATOMY=knee                     # knee or brain
MODEL_NAME=e2e_varnet            # e2e_varnet or fi_varnet
ACCELERATION=4                   # acceleration factor
CENTER_FRACTION=0.08             # optional; inferred if left empty
FIXED_MASK_TYPE=equispaced_fraction
PRECISION=32
BATCH_SIZE=1
NUM_WORKERS=4
```

Reconstruction architecture variables:

```bash
NUM_CASCADES=12
CHANS=32
POOLS=4
SENS_CHANS=8
SENS_POOLS=4
VN_LR=0.0003
VN_WEIGHT_DECAY=0.0
VN_MAX_STEPS=210000
VN_RAMP_STEPS=7500
VN_COSINE_DECAY_START=150000
```

Uncertainty variables:

```bash
UNC_ALPHA=0.1
UNC_HEAD_CHANS=32
UNC_HEAD_POOLS=4
UNC_DROP_PROB=0.0
UNC_LR=0.0003
UNC_WEIGHT_DECAY=0.0
UNC_MAX_STEPS=210000
UNC_RAMP_STEPS=7500
UNC_COSINE_DECAY_START=150000
```

Calibration variables:

```bash
UNC_CAL_DELTA=0.1
UNC_CAL_LAM_START=1.0
UNC_CAL_LAM_END=2.0
UNC_CAL_LAM_STEPS=20
UNC_CAL_SAMPLE_RATE=1.0
```

Run-control variables:

```bash
RUN_TRAINING=true
RUN_TESTING=false
RUN_EVALUATION=false
RUN_UNCERTAINTY_TRAINING=false
RUN_UNCERTAINTY_CALIBRATION=false
RUN_UNCERTAINTY_TESTING=false
```

---

## Basic usage

Most experiments are launched through `batch_run.sh`.

### Train a reconstruction model

```bash
sbatch \
  --job-name=VN_K_4x_train \
  --export=ALL,ANATOMY=knee,MODEL_NAME=e2e_varnet,ACCELERATION=4,RUN_TRAINING=true,RUN_TESTING=false,RUN_EVALUATION=false \
  batch_run.sh
```

### Test and evaluate a reconstruction model

```bash
sbatch \
  --job-name=VN_K_4x_test \
  --export=ALL,ANATOMY=knee,MODEL_NAME=e2e_varnet,ACCELERATION=4,RUN_TRAINING=false,RUN_TESTING=true,RUN_EVALUATION=true \
  batch_run.sh
```

Reconstruction outputs are saved under:

```text
${LOG_PATH}/${MODEL_NAME}/reconstructions_${VN_MODEL}/
```

where `VN_MODEL` is generated automatically from the anatomy, model, acceleration, loss, and mask type.

---

## Uncertainty workflow

The uncertainty workflow assumes that a trained reconstruction checkpoint already exists.

### Train uncertainty model

```bash
sbatch \
  --job-name=VN_K_4x_unc_train \
  --export=ALL,ANATOMY=knee,MODEL_NAME=e2e_varnet,ACCELERATION=4,RUN_TRAINING=false,RUN_TESTING=false,RUN_EVALUATION=false,RUN_UNCERTAINTY_TRAINING=true,RUN_UNCERTAINTY_CALIBRATION=false,RUN_UNCERTAINTY_TESTING=false \
  batch_run.sh
```

### Calibrate uncertainty intervals

```bash
sbatch \
  --job-name=VN_K_4x_unc_cal \
  --export=ALL,ANATOMY=knee,MODEL_NAME=e2e_varnet,ACCELERATION=4,RUN_TRAINING=false,RUN_TESTING=false,RUN_EVALUATION=false,RUN_UNCERTAINTY_TRAINING=false,RUN_UNCERTAINTY_CALIBRATION=true,RUN_UNCERTAINTY_TESTING=false \
  batch_run.sh
```

### Test uncertainty model

Run reconstruction testing first so that the reconstruction HDF5 files exist. Then run:

```bash
sbatch \
  --job-name=VN_K_4x_unc_test \
  --export=ALL,ANATOMY=knee,MODEL_NAME=e2e_varnet,ACCELERATION=4,RUN_TRAINING=false,RUN_TESTING=false,RUN_EVALUATION=false,RUN_UNCERTAINTY_TRAINING=false,RUN_UNCERTAINTY_CALIBRATION=false,RUN_UNCERTAINTY_TESTING=true \
  batch_run.sh
```

This appends `uncertainty_map` to the reconstruction files saved by the reconstruction test step.

---

## Direct Python entry points

The SLURM launcher is the recommended workflow, but the Python entry points are:

```bash
python3 runner.py --help
python3 runner_uncertainty.py --help
```

Evaluation can be run directly with:

```bash
python3 -m utilities.evaluation \
  --target-path /path/to/multicoil_test \
  --predictions-path /path/to/reconstructions
```

---

## Recommended first run

A stable first experiment is:

```bash
ANATOMY=knee
MODEL_NAME=e2e_varnet
ACCELERATION=4
FIXED_MASK_TYPE=equispaced_fraction
PRECISION=32
```

Recommended sequence:

1. Train the reconstruction model.
2. Test the reconstruction model.
3. Evaluate reconstruction metrics.
4. Train the uncertainty model.
5. Calibrate uncertainty intervals.
6. Test uncertainty and append uncertainty maps.

---

## Citations

For the original E2E VarNet implementation, refer to:

- https://github.com/facebookresearch/fastMRI

If you use E2E VarNet, cite:

- Sriram, Anuroop, et al. "End-to-end variational networks for accelerated MRI reconstruction." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2020.

If you use FI VarNet, cite:

- Giannakopoulos, Ilias I., et al. "Accelerated MRI reconstructions via variational network and feature domain learning." Scientific Reports 14.1 (2024): 10991.

If you use the uncertainty method, cite:

- Giannakopoulos, Ilias I., et al. "Pixelwise Uncertainty Quantification of Accelerated MRI Reconstruction." arXiv preprint arXiv:2601.13236 (2026).

## Issues and Disclaimer

This is an in-house MRI reconstruction codebase intended for research purposes. While every effort has been made to ensure its quality, the code may contain bugs or unexpected behavior. We do not assume responsibility for any errors, issues, data loss, or conclusions resulting from the software's use.

Users are encouraged to report problems, ask questions, or suggest improvements by contacting:

**ilias[dot]giannakopoulos[at]nyulangone[dot]org**

## Acknowledgments

This work was supported in part by the National Institutes of Health (NIBIB) under awards **K99 EB035163** and **R01 EB024536**.

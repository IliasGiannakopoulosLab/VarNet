# MRI Reconstruction with VarNet, Learnable Sampling, and Pixelwise Uncertainty

This repository contains PyTorch Lightning implementations for accelerated MRI reconstruction using variational networks and pixelwise uncertainty estimation.
The stable functionality is focused on 2D reconstruction. 

## Main features

- 2D accelerated MRI reconstruction
- Feature-image and end-to-end VarNet models
- Fixed Cartesian undersampling masks
- Pixelwise uncertainty estimation using quantile regression and conformal calibration

## Repository structure

```text
data/
  data_module.py              Lightning data module and fastMRI-style datasets
  data_transforms.py          2D and volume VarNet transforms
  learnable_mask.py           Learnable Cartesian masks
  undersampling_patterns.py   Fixed Cartesian mask definitions

varnets/
  VarNet.py                   VarNet, FI-VarNet, learnable-mask wrappers
  VarNet_module.py            2D Lightning reconstruction module
  Network_module.py           2D validation, testing, and logging utilities
  uncertainty_module.py       2D uncertainty module

utilities/
  functions.py                FFTs, complex operations, cropping, saving reconstructions
  losses.py                   SSIM, perceptual, pinball, and 3D-beta losses
  evaluation.py               MSE, NMSE, PSNR, SSIM evaluation

runner.py                     Main reconstruction training/testing entry point
runner_uncertainty.py         Uncertainty training/calibration/testing entry point
```

## 2D VarNet reconstruction

The main reconstruction pipeline supports 2D accelerated MRI reconstruction from undersampled multicoil k-space data.

Supported reconstruction backbones include:

- `e2e_varnet`
- `fi_varnet`

The standard reconstruction flow is:

```text
undersampled k-space
    -> sensitivity estimation
    -> cascaded image-domain regularization and data consistency
    -> reconstructed magnitude image
```
The default reconstruction loss is SSIM loss.

## Pixelwise uncertainty estimation

The repository includes a separate uncertainty pipeline built on top of a pretrained reconstruction model.

The reconstruction model is trained first. Then, during uncertainty training, the reconstruction model is frozen and a separate uncertainty network predicts lower and upper uncertainty bounds around the reconstruction.

The uncertainty workflow is:

```text
train VarNet
    -> freeze VarNet
    -> train uncertainty network
    -> calibrate uncertainty intervals
    -> test and save uncertainty maps
```

The uncertainty pipeline includes:

1. loading a pretrained reconstruction model
2. freezing the reconstruction model
3. training an uncertainty network using quantile regression
4. using pinball loss for lower and upper quantiles
5. conformal calibration of the predicted intervals
6. saving calibrated uncertainty maps during testing

This is useful for identifying spatial regions where the reconstructed image is likely to be unreliable.

## Dataset format

The code expects fastMRI-style HDF5 files.

Typical folder structure:

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

Each HDF5 file should contain multicoil k-space data and the corresponding reconstruction target.

## Basic usage

Most experiments are launched through `batch_run.sh`.

### 2D fixed-mask training

```bash
sbatch \
  --job-name=VN_2D_K_4x_train \
  --export=ALL,EXPERIMENT_DIM=2D,ANATOMY=knee,ACCELERATION=4,MASK_MODE=fixed,FIXED_MASK_TYPE=equispaced_fraction,RUN_TRAINING=true,RUN_TESTING=false,RUN_EVALUATION=false \
  batch_run.sh
```

### 2D testing and evaluation

```bash
sbatch \
  --job-name=VN_2D_K_4x_test \
  --export=ALL,EXPERIMENT_DIM=2D,ANATOMY=knee,ACCELERATION=4,MASK_MODE=fixed,RUN_TRAINING=false,RUN_TESTING=true,RUN_EVALUATION=true \
  batch_run.sh
```

## Important SLURM variables

Common experiment controls:

```bash
EXPERIMENT_DIM=2D
ANATOMY=knee
ACCELERATION=4
MASK_MODE=fixed
FIXED_MASK_TYPE=equispaced_fraction
TRAIN_MODE=train
PRECISION=bf16
```

2D architecture variables:

```bash
MODEL_NAME_2D=e2e_varnet
NUM_CASCADES_2D=12
CHANS_2D=32
POOLS_2D=4
SENS_CHANS_2D=8
SENS_POOLS_2D=4
VN_LR_2D=0.0003
```

Uncertainty variables:

```bash
RUN_UNCERTAINTY_TRAINING=true
RUN_UNCERTAINTY_CALIBRATION=true
RUN_UNCERTAINTY_TESTING=true

UNC_ALPHA=0.1
UNC_HEAD_CHANS_2D=32
UNC_HEAD_POOLS_2D=4
UNC_LR_2D=0.0003
```

## Training uncertainty

The uncertainty model should be trained after a reconstruction checkpoint exists.

### Train uncertainty network

```bash
sbatch \
  --job-name=VN_2D_K_4x_unc_train \
  --export=ALL,EXPERIMENT_DIM=2D,ANATOMY=knee,ACCELERATION=4,MASK_MODE=fixed,RUN_TRAINING=false,RUN_TESTING=false,RUN_EVALUATION=false,RUN_UNCERTAINTY_TRAINING=true,RUN_UNCERTAINTY_CALIBRATION=false,RUN_UNCERTAINTY_TESTING=false \
  batch_run.sh
```

### Calibrate uncertainty intervals

```bash
sbatch \
  --job-name=VN_2D_K_4x_unc_cal \
  --export=ALL,EXPERIMENT_DIM=2D,ANATOMY=knee,ACCELERATION=4,MASK_MODE=fixed,RUN_TRAINING=false,RUN_TESTING=false,RUN_EVALUATION=false,RUN_UNCERTAINTY_TRAINING=false,RUN_UNCERTAINTY_CALIBRATION=true,RUN_UNCERTAINTY_TESTING=false \
  batch_run.sh
```

### Test uncertainty model

```bash
sbatch \
  --job-name=VN_2D_K_4x_unc_test \
  --export=ALL,EXPERIMENT_DIM=2D,ANATOMY=knee,ACCELERATION=4,MASK_MODE=fixed,RUN_TRAINING=false,RUN_TESTING=false,RUN_EVALUATION=false,RUN_UNCERTAINTY_TRAINING=false,RUN_UNCERTAINTY_CALIBRATION=false,RUN_UNCERTAINTY_TESTING=true \
  batch_run.sh
```

## Outputs

Reconstruction outputs are saved as HDF5 files containing:

```text
reconstruction
```

Uncertainty testing saves or appends uncertainty maps depending on the selected pipeline.

## Evaluation

The evaluation script computes:

- MSE
- NMSE
- PSNR
- SSIM

Example:

```bash
python3 -m utilities.evaluation \
  --target-path /path/to/multicoil_test \
  --predictions-path /path/to/reconstructions
```

## Recommended starting point

For a stable first experiment, start with:

```bash
EXPERIMENT_DIM=2D
MODEL_NAME_2D=e2e_varnet
MASK_MODE=fixed
ACCELERATION=4
LOSS_FLAG_2D=ssim
```

Then test the VN.  

Finally, train and calibrate uncertainty on top of the trained reconstruction checkpoint.

## Citation
- For the original E2E VarNet implementation repository please refer to https://github.com/facebookresearch/fastMRI
- If you use the E2E VarNet please cite Sriram, Anuroop, et al. "End-to-end variational networks for accelerated MRI reconstruction." International conference on medical image computing and computer-assisted intervention. Cham: Springer International Publishing, 2020.
- If you use the FI VarNet please cite Giannakopoulos, Ilias I., et al. "Accelerated MRI reconstructions via variational network and feature domain learning." Scientific Reports 14.1 (2024): 10991.
- If you use the Uncertainty please cite Giannakopoulos, Ilias I., et al. "Pixelwise Uncertainty Quantification of Accelerated MRI Reconstruction." arXiv preprint arXiv:2601.13236 (2026).
- If you use the Learnable Mask or the 3D version of the VarNet please cite the repository 

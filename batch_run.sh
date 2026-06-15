#!/bin/bash
#SBATCH --partition=radiology
#SBATCH --job-name=VN_recon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=3-00:00:00
#SBATCH --mem=300G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

ANATOMY=${ANATOMY:-knee}
ACCELERATION=${ACCELERATION:-4}
CENTER_FRACTION=${CENTER_FRACTION:-}
MASK_MODE=${MASK_MODE:-fixed}
FIXED_MASK_TYPE=${FIXED_MASK_TYPE:-equispaced_fraction}
NUM_LOGITS=${NUM_LOGITS:-320}
TRAIN_MODE=${TRAIN_MODE:-train}
FINE_TUNE_CKPT=${FINE_TUNE_CKPT:-your_checkpoint.ckpt}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
PRECISION=${PRECISION:-32}
RUN_TRAINING=${RUN_TRAINING:-true}
RUN_TESTING=${RUN_TESTING:-false}
RUN_EVALUATION=${RUN_EVALUATION:-false}
RUN_UNCERTAINTY_TRAINING=${RUN_UNCERTAINTY_TRAINING:-false}
RUN_UNCERTAINTY_CALIBRATION=${RUN_UNCERTAINTY_CALIBRATION:-false}
RUN_UNCERTAINTY_TESTING=${RUN_UNCERTAINTY_TESTING:-false}

# reconstruction
MODEL_NAME=${MODEL_NAME:-e2e_varnet}
NUM_CASCADES=${NUM_CASCADES:-12}
CHANS=${CHANS:-32}
POOLS=${POOLS:-4}
SENS_CHANS=${SENS_CHANS:-8}
SENS_POOLS=${SENS_POOLS:-4}
VN_LR=${VN_LR:-0.0003}
VN_LR_BASE=${VN_LR_BASE:-}
VN_LR_MASK=${VN_LR_MASK:-}
VN_WEIGHT_DECAY=${VN_WEIGHT_DECAY:-0.0}
VN_MAX_STEPS=${VN_MAX_STEPS:-210000}
VN_RAMP_STEPS=${VN_RAMP_STEPS:-7500}
VN_COSINE_DECAY_START=${VN_COSINE_DECAY_START:-150000}

# uncertainty
UNC_ALPHA=${UNC_ALPHA:-0.1}
UNC_CAL_DELTA=${UNC_CAL_DELTA:-0.1}
UNC_CAL_LAM_START=${UNC_CAL_LAM_START:-1.0}
UNC_CAL_LAM_END=${UNC_CAL_LAM_END:-2.0}
UNC_CAL_LAM_STEPS=${UNC_CAL_LAM_STEPS:-20}
UNC_CAL_SAMPLE_RATE=${UNC_CAL_SAMPLE_RATE:-1.0}
UNC_HEAD_CHANS=${UNC_HEAD_CHANS:-32}
UNC_HEAD_POOLS=${UNC_HEAD_POOLS:-4}
UNC_DROP_PROB=${UNC_DROP_PROB:-0.0}
UNC_LR=${UNC_LR:-0.0003}
UNC_WEIGHT_DECAY=${UNC_WEIGHT_DECAY:-0.0}
UNC_MAX_STEPS=${UNC_MAX_STEPS:-210000}
UNC_RAMP_STEPS=${UNC_RAMP_STEPS:-7500}
UNC_COSINE_DECAY_START=${UNC_COSINE_DECAY_START:-150000}

# repository location
VARNET_DIR=${VARNET_DIR:-/gpfs/home/gianni02/VarNet-main}

# read-only fastMRI dataset location
DATA_ROOT=${DATA_ROOT:-/gpfs/data/gianni02lab/Team/Datasets/FastMRI}
DATA_DIR=${DATA_DIR:-${DATA_ROOT}/${ANATOMY}}
VN_TRAIN_PATH=${VN_TRAIN_PATH:-${DATA_DIR}/multicoil_train}
VN_VAL_PATH=${VN_VAL_PATH:-${DATA_DIR}/multicoil_val}
VN_CAL_PATH=${VN_CAL_PATH:-${DATA_DIR}/multicoil_cal}
VN_TEST_PATH=${VN_TEST_PATH:-${DATA_DIR}/multicoil_test}

# user-owned output location
OUTPUT_ROOT=${OUTPUT_ROOT:-/gpfs/data/gianni02lab/Ilias/Image_Reconstruction/Learnable_Mask_Models}

if [ -z "${CENTER_FRACTION}" ]; then
    case ${ACCELERATION} in
        2) CENTER_FRACTION=0.16 ;;
        4) CENTER_FRACTION=0.08 ;;
        6) CENTER_FRACTION=0.06 ;;
        8) CENTER_FRACTION=0.04 ;;
        10) CENTER_FRACTION=0.02 ;;
        *) CENTER_FRACTION=0.08 ;;
    esac
fi

if [ "${MASK_MODE}" = "learnable" ]; then
    MASK_TAG=learnable_mask
else
    MASK_TAG=${FIXED_MASK_TYPE}
fi

VN_MODEL=Model_${ANATOMY}_${MODEL_NAME}_${ACCELERATION}x_SSIM_${MASK_TAG}
UNC_MODEL=${VN_MODEL}_uncertainty

# Every generated file is stored below this model-specific directory.
MODEL_OUTPUT_DIR=${OUTPUT_ROOT}/${VN_MODEL}
LOG_PATH=${MODEL_OUTPUT_DIR}
RUN_ROOT_DIR=${MODEL_OUTPUT_DIR}/${MODEL_NAME}
UNC_ROOT_DIR=${RUN_ROOT_DIR}/uncertainty
CHECKPOINT_DIR=${RUN_ROOT_DIR}/checkpoints/${VN_MODEL}
UNC_CHECKPOINT_DIR=${UNC_ROOT_DIR}/checkpoints/${UNC_MODEL}
VN_PREDICTIONS_PATH=${RUN_ROOT_DIR}/reconstructions_${VN_MODEL}
VARNET_CKPT_PREFIX=fi_varnet.

mkdir -p "${MODEL_OUTPUT_DIR}"

# SLURM stdout/stderr is also stored inside the model directory.
SLURM_LOG_FILE=${MODEL_OUTPUT_DIR}/slurm_${SLURM_JOB_NAME:-VN_recon}_${SLURM_JOB_ID:-manual}.out
exec > >(tee -a "${SLURM_LOG_FILE}") 2>&1

# Make the repository importable while keeping the working directory inside
# MODEL_OUTPUT_DIR. This also stores dataset_cache.pkl in the model directory
# instead of the repository or the read-only dataset directory.
export PYTHONPATH="${VARNET_DIR}:${PYTHONPATH:-}"
cd "${MODEL_OUTPUT_DIR}"

resolve_preferred_ckpt() {
    local ckpt_dir="$1"
    local preferred=""
    local fallback=""
    preferred=$(find "${ckpt_dir}" -maxdepth 1 -type f -name "*.ckpt" ! -name "last.ckpt" 2>/dev/null | sort | tail -n 1 || true)
    fallback="${ckpt_dir}/last.ckpt"
    if [ -n "${preferred}" ] && [ -f "${preferred}" ]; then
        echo "${preferred}"
    elif [ -f "${fallback}" ]; then
        echo "${fallback}"
    else
        echo ""
    fi
}

MASK_ARGS=(--mask_mode "${MASK_MODE}")
if [ "${MASK_MODE}" = "fixed" ]; then
    MASK_ARGS+=(--mask_type "${FIXED_MASK_TYPE}")
else
    MASK_ARGS+=(--num_logits "${NUM_LOGITS}")
fi

TRAIN_MODE_ARGS=(--mode "${TRAIN_MODE}")
if [ "${TRAIN_MODE}" = "fine_tune" ]; then
    TRAIN_MODE_ARGS+=(--fine_tune_ckpt "${FINE_TUNE_CKPT}")
fi

COMMON_ARGS=(
    --data_path "${DATA_DIR}"
    --data_path_train "${VN_TRAIN_PATH}"
    --data_path_val "${VN_VAL_PATH}"
    --log_path "${LOG_PATH}"
    --accelerations "${ACCELERATION}"
    --center_fractions "${CENTER_FRACTION}"
    --varnet_type "${MODEL_NAME}"
    --model_name "${VN_MODEL}"
    --num_cascades "${NUM_CASCADES}"
    --chans "${CHANS}"
    --pools "${POOLS}"
    --sens_chans "${SENS_CHANS}"
    --sens_pools "${SENS_POOLS}"
    --precision "${PRECISION}"
    --lr "${VN_LR}"
    --weight_decay "${VN_WEIGHT_DECAY}"
    --max_steps "${VN_MAX_STEPS}"
    --ramp_steps "${VN_RAMP_STEPS}"
    --cosine_decay_start "${VN_COSINE_DECAY_START}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
)

if [ -n "${VN_LR_BASE}" ]; then
    COMMON_ARGS+=(--lr_base "${VN_LR_BASE}")
fi
if [ -n "${VN_LR_MASK}" ]; then
    COMMON_ARGS+=(--lr_mask "${VN_LR_MASK}")
fi

UNC_ARGS=(
    --data_path "${DATA_DIR}"
    --data_path_train "${VN_TRAIN_PATH}"
    --data_path_val "${VN_VAL_PATH}"
    --data_path_cal "${VN_CAL_PATH}"
    --log_path "${LOG_PATH}"
    --accelerations "${ACCELERATION}"
    --center_fractions "${CENTER_FRACTION}"
    --varnet_type "${MODEL_NAME}"
    --model_name "${UNC_MODEL}"
    --varnet_ckpt_prefix "${VARNET_CKPT_PREFIX}"
    --num_cascades "${NUM_CASCADES}"
    --chans "${CHANS}"
    --pools "${POOLS}"
    --sens_chans "${SENS_CHANS}"
    --sens_pools "${SENS_POOLS}"
    --precision "${PRECISION}"
    --alpha "${UNC_ALPHA}"
    --head_chans "${UNC_HEAD_CHANS}"
    --head_pools "${UNC_HEAD_POOLS}"
    --uncertainty_drop_prob "${UNC_DROP_PROB}"
    --lr "${UNC_LR}"
    --weight_decay "${UNC_WEIGHT_DECAY}"
    --max_steps "${UNC_MAX_STEPS}"
    --ramp_steps "${UNC_RAMP_STEPS}"
    --cosine_decay_start "${UNC_COSINE_DECAY_START}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --calibration_delta "${UNC_CAL_DELTA}"
    --calibration_lambdas_start "${UNC_CAL_LAM_START}"
    --calibration_lambdas_end "${UNC_CAL_LAM_END}"
    --calibration_lambdas_steps "${UNC_CAL_LAM_STEPS}"
    --calibration_sample_rate "${UNC_CAL_SAMPLE_RATE}"
)

echo "============================================================"
echo "VN_MODEL              : ${VN_MODEL}"
echo "UNC_MODEL             : ${UNC_MODEL}"
echo "ANATOMY               : ${ANATOMY}"
echo "ACCELERATION          : ${ACCELERATION}"
echo "CENTER_FRACTION       : ${CENTER_FRACTION}"
echo "MASK_MODE             : ${MASK_MODE}"
echo "FIXED_MASK_TYPE       : ${FIXED_MASK_TYPE}"
echo "NUM_LOGITS            : ${NUM_LOGITS}"
echo "BATCH_SIZE            : ${BATCH_SIZE}"
echo "NUM_CASCADES          : ${NUM_CASCADES}"
echo "CHANS                 : ${CHANS}"
echo "POOLS                 : ${POOLS}"
echo "SENS_CHANS            : ${SENS_CHANS}"
echo "SENS_POOLS            : ${SENS_POOLS}"
echo "PRECISION             : ${PRECISION}"
echo "VARNET_DIR            : ${VARNET_DIR}"
echo "DATA_DIR              : ${DATA_DIR}"
echo "MODEL_OUTPUT_DIR      : ${MODEL_OUTPUT_DIR}"
echo "CHECKPOINT_DIR        : ${CHECKPOINT_DIR}"
echo "VN_PREDICTIONS_PATH   : ${VN_PREDICTIONS_PATH}"
echo "SLURM_LOG_FILE        : ${SLURM_LOG_FILE}"
echo "============================================================"

if [ "${RUN_TRAINING}" = "true" ]; then
    srun python3 "${VARNET_DIR}/runner.py" "${COMMON_ARGS[@]}" "${MASK_ARGS[@]}" "${TRAIN_MODE_ARGS[@]}"
fi

VN_CHECKPOINT_PATH=$(resolve_preferred_ckpt "${CHECKPOINT_DIR}")

if [ "${RUN_TESTING}" = "true" ]; then
    srun python3 "${VARNET_DIR}/runner.py" "${COMMON_ARGS[@]}" "${MASK_ARGS[@]}" --mode test --test_path "${VN_TEST_PATH}"
fi

if [ "${RUN_EVALUATION}" = "true" ]; then
    srun python3 -m utilities.evaluation --target-path "${VN_TEST_PATH}" --predictions-path "${VN_PREDICTIONS_PATH}"
fi

if [ "${RUN_UNCERTAINTY_TRAINING}" = "true" ]; then
    srun python3 "${VARNET_DIR}/runner_uncertainty.py" "${UNC_ARGS[@]}" "${MASK_ARGS[@]}" --mode train --varnet_ckpt "${VN_CHECKPOINT_PATH}"
fi

UNC_CHECKPOINT_PATH=$(resolve_preferred_ckpt "${UNC_CHECKPOINT_DIR}")

if [ "${RUN_UNCERTAINTY_CALIBRATION}" = "true" ]; then
    srun python3 "${VARNET_DIR}/runner_uncertainty.py" "${UNC_ARGS[@]}" "${MASK_ARGS[@]}" --mode calibrate --devices 1 --varnet_ckpt "${VN_CHECKPOINT_PATH}" --uncertainty_ckpt "${UNC_CHECKPOINT_PATH}" --output_uncertainty_ckpt "${UNC_CHECKPOINT_PATH}"
fi

if [ "${RUN_UNCERTAINTY_TESTING}" = "true" ]; then
    srun python3 "${VARNET_DIR}/runner_uncertainty.py" "${UNC_ARGS[@]}" "${MASK_ARGS[@]}" --mode test --varnet_ckpt "${VN_CHECKPOINT_PATH}" --uncertainty_ckpt "${UNC_CHECKPOINT_PATH}" --reconstructions_dir "${VN_PREDICTIONS_PATH}" --test_path "${VN_TEST_PATH}"
fi

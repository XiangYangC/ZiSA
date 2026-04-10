#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTION_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DETECTION_DIR}/.." && pwd)"

TRAIN_PY="${DETECTION_DIR}/tools/train.py"
DIST_TRAIN_SH="${DETECTION_DIR}/tools/dist_train.sh"
CONFIG_DIR="${SCRIPT_DIR}/configs"
WORK_ROOT="${SCRIPT_DIR}/work_dirs"
LOG_ROOT="${SCRIPT_DIR}/logs"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPUS="${GPUS:-2}"
PORT="${PORT:-29500}"

mkdir -p "${WORK_ROOT}" "${LOG_ROOT}"

EXPERIMENTS=(
  base
  zisa_zoom
  zisa_mhsa
  zisa_zoom_mhsa
  zisa_zoom_mhsa_se
  zisa_full
)

cd "${PROJECT_ROOT}"

for exp in "${EXPERIMENTS[@]}"; do
  echo "============================================================"
  echo "Running experiment: ${exp}"
  echo "============================================================"

  cfg="${CONFIG_DIR}/${exp}.py"
  work_dir="${WORK_ROOT}/${exp}"
  log_file="${LOG_ROOT}/${exp}.log"
  extra_args=()

  mkdir -p "${work_dir}"

  case "${exp}" in
    zisa_mhsa|zisa_zoom_mhsa|zisa_zoom_mhsa_se|zisa_full)
      extra_args+=(--cfg-options train_dataloader.batch_size=1)
      ;;
  esac

  if [ "${GPUS}" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PORT="${PORT}" bash "${DIST_TRAIN_SH}" "${cfg}" "${GPUS}" --work-dir "${work_dir}" "${extra_args[@]}" >>"${log_file}" 2>&1
  else
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${TRAIN_PY}" "${cfg}" --work-dir "${work_dir}" "${extra_args[@]}" >>"${log_file}" 2>&1
  fi
done

echo "All ablation experiments completed."

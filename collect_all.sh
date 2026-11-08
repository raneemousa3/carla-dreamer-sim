#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Two Epic servers at a time; the ninth dataset runs alone.
outdir="${1:-dataset/all_conditions_$(date +%Y%m%d_%H%M%S)}"
python="${PYTHON:-.venv/bin/python}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
mkdir -p -- "$(dirname -- "$outdir")"
mkdir -p -- "$outdir"
if [[ -n "$(ls -A -- "$outdir")" ]]; then
  echo "Output directory must be empty: $outdir" >&2
  exit 1
fi

conditions=(
  day:sunny day:cloudy day:rainy
  sunset:sunny sunset:cloudy sunset:rainy
  night:sunny night:cloudy night:rainy
)

for ((i=0; i<${#conditions[@]}; i+=2)); do
  IFS=: read -r first_time first_weather <<< "${conditions[i]}"
  args=(--n-traj 2000 --outdir "$outdir/batch_$((i / 2 + 1))"
        --first-time-of-day "$first_time" --first-weather "$first_weather")
  if ((i + 1 < ${#conditions[@]})); then
    IFS=: read -r second_time second_weather <<< "${conditions[i+1]}"
    args+=(--second-time-of-day "$second_time" --second-weather "$second_weather")
  else
    args+=(--single)
  fi
  "$python" -B -u collect_parallel.py "${args[@]}"
done

echo "All nine datasets completed: $outdir"

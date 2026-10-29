#!/usr/bin/env bash
# backup_env.sh -- backs up a conda env (default: jersey_env) so it can be re-created
# on another machine. Read-only on the env. Creates  env_backup_<name>_<date>/ (+ .tar.gz)
#
#   bash backup_env.sh                 # backs up jersey_env
#   bash backup_env.sh my_other_env    # any env name
#   PACK=0 bash backup_env.sh          # skip the full conda-pack copy (smaller backup)
set -u
ENV_NAME="${1:-jersey_env}"
PACK="${PACK:-1}"

# make "conda activate" work inside a script
CONDA_BASE="$(conda info --base 2>/dev/null)" || { echo "conda not found in PATH"; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh"

ENV_PATH="$(conda env list | awk -v n="$ENV_NAME" '$1==n {print $NF}')"
[ -z "$ENV_PATH" ] && { echo "env '$ENV_NAME' not found. Available:"; conda env list; exit 1; }

OUT="env_backup_${ENV_NAME}_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT"
echo "env      : $ENV_NAME"
echo "location : $ENV_PATH"
echo "backup   : $PWD/$OUT"

# 1) info about the old machine (to compare later)
{
  echo "date: $(date)"; echo "host: $(hostname)"; uname -a
  echo "conda: $(conda --version)"; echo "env path: $ENV_PATH"
  conda run -n "$ENV_NAME" python --version
  conda run -n "$ENV_NAME" python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'gpu',torch.cuda.is_available())" 2>/dev/null
  nvidia-smi --query-gpu=name,driver_version --format=csv 2>/dev/null
} > "$OUT/system_info.txt" 2>&1

# 2) recipes to re-create the env
conda env export -n "$ENV_NAME" --no-builds | grep -v "^prefix:" > "$OUT/environment.yml"          # full, portable
conda env export -n "$ENV_NAME" --from-history | grep -v "^prefix:" > "$OUT/environment_history.yml" # only what you installed
conda list -n "$ENV_NAME" --explicit > "$OUT/conda_explicit.txt"                                     # exact conda packages (same OS)
conda list -n "$ENV_NAME" > "$OUT/conda_list.txt"                                                    # readable list
conda run -n "$ENV_NAME" python -m pip freeze > "$OUT/requirements_pip.txt" 2>/dev/null              # pip packages

# 3) exact copy of the whole env (works on another Linux x86_64 machine as-is)
if [ "$PACK" = "1" ]; then
  if ! conda run -n base conda-pack --version >/dev/null 2>&1; then
    echo "installing conda-pack into base ..."
    conda install -y -n base -c conda-forge conda-pack >/dev/null 2>&1 || \
      conda run -n base python -m pip install conda-pack >/dev/null 2>&1
  fi
  echo "packing full env (can take a few minutes) ..."
  conda run -n base conda-pack -n "$ENV_NAME" -o "$OUT/${ENV_NAME}_packed.tar.gz" --ignore-editable-packages \
    || echo "WARNING: conda-pack failed -> use environment.yml / requirements_pip.txt instead"
fi

cp "$(dirname "$0")/restore_env.sh" "$OUT/" 2>/dev/null
tar -czf "$OUT.tar.gz" "$OUT"
echo
echo "DONE. Copy this ONE file to the new machine:  $PWD/$OUT.tar.gz"
ls -lh "$OUT" "$OUT.tar.gz"

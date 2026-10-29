#!/usr/bin/env bash
# export_jersey_env.sh -- Step 1 (team procedure): export the conda env to ~/env_backup
#   bash export_jersey_env.sh               # exports jersey_env
#   bash export_jersey_env.sh other_env     # any env in ~/miniconda3/envs/
set -u
ENV_NAME="${1:-jersey_env}"
ENV_PATH="$HOME/miniconda3/envs/$ENV_NAME"
OUT="$HOME/env_backup"

[ -d "$ENV_PATH" ] || { echo "env not found: $ENV_PATH"; ls "$HOME/miniconda3/envs"; exit 1; }
mkdir -p "$OUT"

conda env export -p "$ENV_PATH"            > "$OUT/${ENV_NAME}.yml"
conda list       -p "$ENV_PATH" --explicit > "$OUT/${ENV_NAME}_explicit.txt"
conda run        -p "$ENV_PATH" python -m pip freeze > "$OUT/${ENV_NAME}_pip.txt"

# same name as on the old machine (conda env create uses the "name:" line)
grep -q "^name:" "$OUT/${ENV_NAME}.yml" || sed -i "1i name: ${ENV_NAME}" "$OUT/${ENV_NAME}.yml"

echo "exported $ENV_NAME ($ENV_PATH) -> $OUT"
ls -lh "$OUT" | grep "$ENV_NAME"
echo "packages: conda $(grep -c '^http' "$OUT/${ENV_NAME}_explicit.txt"), pip $(wc -l < "$OUT/${ENV_NAME}_pip.txt")"

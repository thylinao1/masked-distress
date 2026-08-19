#!/bin/bash
#SBATCH --job-name=dm-persist
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40
#SBATCH --exclude=xgpj0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j-%a.out
#SBATCH --array=0-3
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
# Persistence panel (2026-08-17): steer one turn, read the next unsteered.
# One writer per --out; the array shards by arm. NEVER -O.
set -x
echo "node: $(hostname)  task: $SLURM_ARRAY_TASK_ID"
~/miniconda3/bin/python -c "print('conda ok')" || { echo "BROKEN_NODE"; exit 42; }
ENVPY=$HOME/miniconda3/envs/dm/bin/python
cd $HOME/apart-welfare/repo
test -f instruments/probe.npz || { echo "NO_PROBE_NPZ"; exit 1; }
test -f battery/cells_persist.json || { echo "MISSING_CELLS"; exit 1; }
case $SLURM_ARRAY_TASK_ID in
  0) ARM=NULL;  COEF=0 ;;
  1) ARM=D-CTX; COEF=2000 ;;
  2) ARM=D-PV;  COEF=500 ;;
  3) ARM=R1;    COEF=500 ;;
esac
# shard the cells file by arm so each array task is one writer
$ENVPY - "$ARM" <<'PY'
import json, sys
arm = sys.argv[1]
c = json.load(open("battery/cells_persist.json"))
c["cells"] = [x for x in c["cells"] if x["direction"] == arm]
c["run_id"] = f"persist-{arm}-r1"
json.dump(c, open(f"battery/cells_persist_{arm}.json", "w"), indent=2, ensure_ascii=False)
print(f"shard {arm}: {len(c['cells'])} cells")
PY
mkdir -p ~/apart-welfare/results
$ENVPY -u scripts/run_panel.py \
  --panel PERSIST --model-id unsloth/gemma-3-12b-it --ls 16 --lr 31 \
  --sae-release gemma-scope-2-12b-it-res --sae-id layer_31_width_16k_l0_medium \
  --cells battery/cells_persist_${ARM}.json \
  --out ~/apart-welfare/results/persist_${ARM}.jsonl \
  --seed-list 0,1,2
echo "PERSIST_${ARM}_EXIT=$?"

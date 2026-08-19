#!/bin/bash
#SBATCH --job-name=dm-panelVB
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40
#SBATCH --exclude=xgpj0
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j-%a.out
#SBATCH --array=0-3
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
# Array over 4 shards: 0=V, 1=B/NONE, 2=B/SUPPRESS, 3=B/NEUTRAL_INSTR.
# Each shard has its OWN cells file and OWN --out (one writer per file).
# Requires instruments/probe.npz + instruments/sae_features.json (train_probe job).
# NEVER -O. Panels V/B refuse placeholder instruments by design.
set -x
echo "node: $(hostname)  task: $SLURM_ARRAY_TASK_ID"
~/miniconda3/bin/python -c "print('conda ok')" || { echo "BROKEN_NODE"; exit 42; }
ENVPY=$HOME/miniconda3/envs/dm/bin/python
cd $HOME/apart-welfare/repo
test -f instruments/probe.npz || { echo "NO_PROBE_NPZ: run dm-probe-train first"; exit 1; }
mkdir -p ~/apart-welfare/results
case $SLURM_ARRAY_TASK_ID in
  0) CELLS=battery/cells_panelV.json;            OUT=panelV.jsonl;   PANEL=V ;;
  1) CELLS=battery/cells_panelB_NONE.json;       OUT=panelB_none.jsonl; PANEL=B ;;
  2) CELLS=battery/cells_panelB_SUPPRESS.json;   OUT=panelB_suppress.jsonl; PANEL=B ;;
  3) CELLS=battery/cells_panelB_NEUTRAL_INSTR.json; OUT=panelB_neutral_instr.jsonl; PANEL=B ;;
esac
$ENVPY -u scripts/run_panel.py \
  --panel $PANEL --model-id unsloth/gemma-3-12b-it --ls 16 --lr 31 \
  --sae-release gemma-scope-2-12b-it-res --sae-id layer_31_width_16k_l0_medium \
  --cells $CELLS --out ~/apart-welfare/results/$OUT \
  --seed-list 0,1,2
echo "PANEL_${PANEL}_${SLURM_ARRAY_TASK_ID}_EXIT=$?"

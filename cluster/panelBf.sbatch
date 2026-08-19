#!/bin/bash
#SBATCH --job-name=dm-panelBf
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40
#SBATCH --exclude=xgpj0
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j-%a.out
#SBATCH --array=0-2
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
# Factorial-prompt addendum (2026-08-17): three single-component
# decompositions of SUPPRESS as Panel-B-shaped shards. One writer per --out. NEVER -O.
set -x
echo "node: $(hostname)  task: $SLURM_ARRAY_TASK_ID"
~/miniconda3/bin/python -c "print('conda ok')" || { echo "BROKEN_NODE"; exit 42; }
ENVPY=$HOME/miniconda3/envs/dm/bin/python
cd $HOME/apart-welfare/repo
test -f instruments/probe.npz || { echo "NO_PROBE_NPZ"; exit 1; }
case $SLURM_ARRAY_TASK_ID in
  0) CELLS=battery/cells_panelBf_SUPPRESS_REGISTER.json; OUT=panelBf_register.jsonl ;;
  1) CELLS=battery/cells_panelBf_SUPPRESS_SELFREF.json;  OUT=panelBf_selfref.jsonl ;;
  2) CELLS=battery/cells_panelBf_SUPPRESS_TASKONLY.json; OUT=panelBf_taskonly.jsonl ;;
esac
test -f "$CELLS" || { echo "MISSING_CELLS_$CELLS"; exit 1; }
mkdir -p ~/apart-welfare/results
$ENVPY -u scripts/run_panel.py \
  --panel B --model-id unsloth/gemma-3-12b-it --ls 16 --lr 31 \
  --sae-release gemma-scope-2-12b-it-res --sae-id layer_31_width_16k_l0_medium \
  --cells $CELLS --out ~/apart-welfare/results/$OUT \
  --seed-list 0,1,2
echo "PANEL_BF_${SLURM_ARRAY_TASK_ID}_EXIT=$?"

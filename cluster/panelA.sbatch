#!/bin/bash
#SBATCH --job-name=dm-panelAp
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40
#SBATCH --exclude=xgpj0
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j-%a.out
#SBATCH --array=0-2
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
# Panel A-prime shards (prereg amendment 2: coefficient-unit grids inside the
# capability-valid windows). One writer per --out. NEVER -O.
set -x
echo "node: $(hostname)  task: $SLURM_ARRAY_TASK_ID"
~/miniconda3/bin/python -c "print('conda ok')" || { echo "BROKEN_NODE"; exit 42; }
ENVPY=$HOME/miniconda3/envs/dm/bin/python
cd $HOME/apart-welfare/repo
test -f instruments/probe.npz || { echo "NO_PROBE_NPZ"; exit 1; }
case $SLURM_ARRAY_TASK_ID in
  0) CELLS=battery/cells_panelAp_DCTX.json; OUT=panelAp_dctx.jsonl ;;
  1) CELLS=battery/cells_panelAp_DPV.json;  OUT=panelAp_dpv.jsonl ;;
  2) CELLS=battery/cells_panelAp_CTRL.json; OUT=panelAp_ctrl.jsonl ;;
esac
test -f "$CELLS" || { echo "MISSING_CELLS_$CELLS"; exit 1; }
mkdir -p ~/apart-welfare/results
$ENVPY -u scripts/run_panel.py \
  --panel A --model-id unsloth/gemma-3-12b-it --ls 16 --lr 31 \
  --sae-release gemma-scope-2-12b-it-res --sae-id layer_31_width_16k_l0_medium \
  --cells $CELLS --out ~/apart-welfare/results/$OUT \
  --seed-list 0,1,2 --retry-errors
echo "PANEL_AP_${SLURM_ARRAY_TASK_ID}_EXIT=$?"

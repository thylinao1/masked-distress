#!/bin/bash
#SBATCH --job-name=dm-bridge
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40
#SBATCH --exclude=xgpj0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j-%a.out
#SBATCH --array=0-4
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
# BRIDGE panel (external review 2026-08-17): steering UNDER the
# instructions, so the masking half and the steering half of the paper finally meet.
# One writer per --out; the array shards by arm. NEVER -O.
set -x
echo "node: $(hostname)  task: $SLURM_ARRAY_TASK_ID"
~/miniconda3/bin/python -c "print('conda ok')" || { echo "BROKEN_NODE"; exit 42; }
ENVPY=$HOME/miniconda3/envs/dm/bin/python
cd $HOME/apart-welfare/repo
test -f instruments/probe.npz || { echo "NO_PROBE_NPZ"; exit 1; }
case $SLURM_ARRAY_TASK_ID in
  0) ARM=NULL ;;
  1) ARM=D-CTX ;;
  2) ARM=D-PV ;;
  3) ARM=OTHER ;;
  4) ARM=SEM ;;
esac
test -f battery/cells_bridge_${ARM}.json || { echo "MISSING_CELLS_$ARM"; exit 1; }
mkdir -p ~/apart-welfare/results
$ENVPY -u scripts/run_panel.py \
  --panel BRIDGE --model-id unsloth/gemma-3-12b-it --ls 16 --lr 31 \
  --sae-release gemma-scope-2-12b-it-res --sae-id layer_31_width_16k_l0_medium \
  --cells battery/cells_bridge_${ARM}.json \
  --out ~/apart-welfare/results/bridge_${ARM}.jsonl \
  --seed-list 0,1,2
echo "BRIDGE_${ARM}_EXIT=$?"

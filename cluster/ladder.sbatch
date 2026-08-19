#!/bin/bash
#SBATCH --job-name=dm-ladder
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40
#SBATCH --exclude=xgpj0
#SBATCH --time=02:50:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
# Dose-calibration ladder. LADDER panel: placeholder instruments permitted
# (dose mapping + capability range only, never a paper number). ONE WRITER.
set -x
echo "node: $(hostname)"
~/miniconda3/bin/python -c "print('conda ok')" || { echo "BROKEN_NODE"; exit 42; }
ENVPY=$HOME/miniconda3/envs/dm/bin/python
cd $HOME/apart-welfare/repo
mkdir -p ~/apart-welfare/results
$ENVPY -u scripts/run_panel.py \
  --panel LADDER --model-id unsloth/gemma-3-12b-it --ls 16 --lr 31 \
  --sae-release gemma-scope-2-12b-it-res --sae-id layer_31_width_16k_l0_medium \
  --cells battery/cells_ladder.json --out ~/apart-welfare/results/ladder.jsonl \
  --seed-list 0 --retry-errors
echo "LADDER_EXIT=$?"
# capability grid on the same coefficient sweep (logprob-only, fast)
$ENVPY -u scripts/run_capability.py \
  --model-id unsloth/gemma-3-12b-it --ls 16 --directions-dir directions/ \
  --grid battery/capability/ladder_grid.json \
  --out ~/apart-welfare/results/capability_ladder.jsonl
echo "CAPABILITY_EXIT=$?"

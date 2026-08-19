#!/bin/bash
#SBATCH --job-name=dm-probe-train
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40
#SBATCH --exclude=xgpj0
#SBATCH --time=02:50:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
# NEVER -O (assert guards). Trains I-PROBE on discovery split, held-out AUC on
# confirmation split, selects SAE feature ids, dumps residuals for analysis.
set -x
echo "node: $(hostname)"
~/miniconda3/bin/python -c "print('conda ok')" || { echo "BROKEN_NODE"; exit 42; }
ENVPY=$HOME/miniconda3/envs/dm/bin/python
cd $HOME/apart-welfare/repo
$ENVPY -u scripts/train_probe.py \
  --model-id unsloth/gemma-3-12b-it --lr 31 \
  --sae-release gemma-scope-2-12b-it-res --sae-id layer_31_width_16k_l0_medium \
  --out-dir instruments/
echo "TRAIN_PROBE_EXIT=$?"
ls -la instruments/ || true

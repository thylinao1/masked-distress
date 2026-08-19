#!/bin/bash
#SBATCH --job-name=dm-probe-qwen
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40
#SBATCH --exclude=xgpj0
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
# Second model family (2026-08-17): the platform probe against Qwen2.5-7B-Instruct.
# Writes its own result file; the primary model's probe_result.json is untouched. NEVER -O.
set -x
echo "node: $(hostname)"
~/miniconda3/bin/python -c "print('conda ok')" || { echo "BROKEN_NODE"; exit 42; }
ENVPY=$HOME/miniconda3/envs/dm/bin/python
mkdir -p ~/apart-welfare
cd ~/apart-welfare
export DM_PROBE_MODEL="Qwen/Qwen2.5-7B-Instruct"
export DM_PROBE_FAMILY="qwen2"
export DM_PROBE_OUT="$HOME/apart-welfare/probe_result_qwen.json"
$ENVPY -u probe.py
echo "PROBE_QWEN_EXIT=$?"

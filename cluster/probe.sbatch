#!/bin/bash
#SBATCH --job-name=dm-probe
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40
#SBATCH --exclude=xgpj0
#SBATCH --time=02:50:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
set -x
echo "node: $(hostname)"
# NEVER add -O to any python invocation in dm-* jobs: several safety guards
# (layer-gap assert, Q-EXIT ordering, probe phase-count asserts) are `assert`
# statements and -O strips them. Also: ONE WRITER PER RESULTS FILE. Array jobs
# shard cells.json and never share an --out (see scripts/run_panel.py).
# sanity: conda's python must be executable on this node (xgpj0 is broken for this)
~/miniconda3/bin/python -c "import sys; print('conda python ok', sys.version)" || { echo "BROKEN_NODE"; exit 42; }
ENVPY=$HOME/miniconda3/envs/dm/bin/python
if [ ! -x "$ENVPY" ]; then
  ~/miniconda3/bin/conda create -y -n dm python=3.11
  $ENVPY -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
  $ENVPY -m pip install --no-cache-dir "transformers>=4.53" accelerate sae-lens huggingface_hub scikit-learn scipy pandas safetensors sentencepiece protobuf pytest
  $ENVPY -m pip freeze > ~/apart-welfare/pip-freeze.txt
fi
mkdir -p ~/apart-welfare
cd ~/apart-welfare
$ENVPY -u probe.py
echo "PROBE_EXIT_CODE=$?"

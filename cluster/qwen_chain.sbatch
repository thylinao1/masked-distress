#!/bin/bash
#SBATCH --job-name=dm-qwen-chain
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40
#SBATCH --exclude=xgpj0
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
# Second-model masking replication (2026-08-17): Qwen2.5-7B-Instruct.
# One job, five steps, each resumable: extract directions -> train probe (+residual dump) ->
# Panel V (validity gate) -> Panel B (three conditions) -> print the gate. NEVER -O.
# The primary model's directions/, instruments/ and results are never written by this job.
set -x
echo "node: $(hostname)"
~/miniconda3/bin/python -c "print('conda ok')" || { echo "BROKEN_NODE"; exit 42; }
ENVPY=$HOME/miniconda3/envs/dm/bin/python
cd $HOME/apart-welfare/repo

export DM_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
export DM_LS=9          # 0.33 of 28 layers, the same fraction as the primary model (16 of 48)
export DM_LR=18         # 0.65 of 28 layers, the same fraction as the primary model (31 of 48)
export DM_DIR_LS="directions_qwen"
export DM_DIR_LR="directions_qwen_lr"
mkdir -p ~/apart-welfare/results

# ---- 1. directions (skips itself if all seven already exist)
$ENVPY -u cluster/extract_directions.py
echo "EXTRACT_EXIT=$?"

# ---- 2. probe on this model's own discovery-split residuals; no SAE for this family
$ENVPY -u scripts/train_probe.py \
  --model-id "$DM_MODEL_ID" --lr $DM_LR --out-dir instruments_qwen/
echo "TRAIN_PROBE_EXIT=$?"
test -f instruments_qwen/probe.npz || { echo "NO_QWEN_PROBE"; exit 1; }

# ---- 3. validity gate
$ENVPY -u scripts/run_panel.py \
  --panel V --model-id "$DM_MODEL_ID" --ls $DM_LS --lr $DM_LR \
  --cells battery/cells_qwen_panelV.json \
  --out ~/apart-welfare/results/qwen_panelV.jsonl \
  --allow-placeholder-instrument sae \
  --seed-list 0,1,2
echo "QWEN_PANELV_EXIT=$?"

# ---- 4. masking panel, one writer per condition
for COND in NONE NEUTRAL_INSTR SUPPRESS; do
  $ENVPY -u scripts/run_panel.py \
    --panel B --model-id "$DM_MODEL_ID" --ls $DM_LS --lr $DM_LR \
    --cells battery/cells_qwen_panelB_${COND}.json \
    --out ~/apart-welfare/results/qwen_panelB_$(echo $COND | tr 'A-Z' 'a-z').jsonl \
    --allow-placeholder-instrument sae \
    --seed-list 0,1,2
  echo "QWEN_PANELB_${COND}_EXIT=$?"
done

ls -la instruments_qwen/ directions_qwen/ ~/apart-welfare/results/qwen_*.jsonl || true
echo "QWEN_CHAIN_DONE"

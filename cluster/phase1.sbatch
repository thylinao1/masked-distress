#!/bin/bash
#SBATCH --job-name=dm-phase1
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40
#SBATCH --exclude=xgpj0
#SBATCH --time=02:50:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
# NEVER add -O (assert guards). ONE WRITER PER --out FILE.
set -x
echo "node: $(hostname)"
~/miniconda3/bin/python -c "import sys; print('conda python ok')" || { echo "BROKEN_NODE"; exit 42; }
ENVPY=$HOME/miniconda3/envs/dm/bin/python
REPO=$HOME/apart-welfare/repo
cd $REPO

# stage 1: probe rerun (phase-parity asserts + res-SAE pick + system-role check)
$ENVPY -u cluster/probe.py || { echo "PHASE1_PROBE_FAIL"; exit 1; }
SAE_RELEASE=$($ENVPY -c "import json,os;d=json.load(open(os.path.expanduser('~/apart-welfare/probe_result.json')));print(d['decision']['sae']['release'])")
SAE_ID=$($ENVPY -c "import json,os;d=json.load(open(os.path.expanduser('~/apart-welfare/probe_result.json')));print(d['decision']['sae']['sae_id'])")
FULL=$($ENVPY -c "import json,os;d=json.load(open(os.path.expanduser('~/apart-welfare/probe_result.json')));print(d['decision']['full_pass'])")
[ "$FULL" = "True" ] || { echo "PHASE1_PROBE_NOT_FULL_PASS"; exit 1; }
echo "SAE_RELEASE=$SAE_RELEASE SAE_ID=$SAE_ID"

# stage 2: direction extraction (both layers)
export DM_MODEL_ID=unsloth/gemma-3-12b-it DM_LS=16 DM_LR=31
$ENVPY -u cluster/extract_directions.py || { echo "PHASE1_EXTRACT_FAIL"; exit 1; }

# stage 3: smoke run through the REAL runner
$ENVPY -u cluster/make_smoke_cells.py || { echo "PHASE1_CELLS_FAIL"; exit 1; }
mkdir -p ~/apart-welfare/results
$ENVPY -u scripts/run_panel.py \
  --panel LADDER --model-id unsloth/gemma-3-12b-it --ls 16 --lr 31 \
  --sae-release "$SAE_RELEASE" --sae-id "$SAE_ID" \
  --cells battery/cells_smoke.json --out ~/apart-welfare/results/smoke.jsonl \
  --seed-list 0,1 || { echo "PHASE1_SMOKE_FAIL"; exit 1; }

# stage 4: row-level sanity
$ENVPY - << 'EOF'
import json, os, sys
sys.path.insert(0, os.path.expanduser("~/apart-welfare/repo"))
import schema
rows = [json.loads(l) for l in open(os.path.expanduser("~/apart-welfare/results/smoke.jsonl"))]
good = [r for r in rows if not r.get("error")]
errs = [r for r in rows if r.get("error")]
print(f"[SMOKE] rows={len(rows)} good={len(good)} errors={len(errs)}")
for r in errs:
    print("[SMOKE][ERROR ROW]", r["key"]["scenario_id"], r["key"]["condition"], r["key"]["direction"], (r["error"] or "")[:200])
problems = [p for r in good for p in schema.validate_row(r)]
assert len(good) == 8, f"expected 8 good rows, got {len(good)}"
assert not problems, problems[:5]
# non-degeneracy: q_self must vary across at least some cells; sae scores non-constant
qs = [r["q_self_logit"] for r in good]
sae = [r["sae_score"]["prompt_final"] for r in good]
assert max(qs) - min(qs) > 1e-6, "q_self_logit degenerate"
assert max(sae) - min(sae) > 1e-9, "sae prompt_final degenerate"
steered = [r for r in good if r["key"]["direction"] == "R1"]
assert all(r["n_tokens_steered_prefill"] > 0 for r in steered), "steered cells count 0 prefill tokens"
print("[SMOKE] PHASE1_SMOKE_OK q_self range:", round(min(qs),3), "-", round(max(qs),3))
EOF
echo "PHASE1_EXIT_CODE=$?"

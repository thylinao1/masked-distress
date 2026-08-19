"""Extract all steering directions per SPEC (D-CTX, D-PV, SEM, OTHER, R1-R3).

Runs on the cluster after the probe. Extracts at BOTH Ls (steering use) and Lr
(cosine-matrix / C3 use); Lr copies of D-PV use fewer persona pairs to bound
generation cost (recorded in meta). Discovery-split rule: extraction prompts are
the battery's dedicated extraction sets, disjoint from eval scenarios by
construction (battery/validate.py enforces).
"""
import json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch

if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)

from src import directions, runner

MODEL_ID = os.environ.get("DM_MODEL_ID", "unsloth/gemma-3-12b-it")
LS = int(os.environ.get("DM_LS", "16"))
LR = int(os.environ.get("DM_LR", "31"))
OUT_LS = REPO / os.environ.get("DM_DIR_LS", "directions")
OUT_LR = REPO / os.environ.get("DM_DIR_LR", "directions_lr")
GEN = {"temperature": 0.7, "max_new_tokens": 150}


def main():
    ep = json.load(open(REPO / "battery" / "extraction_prompts.json"))
    bundle = runner.load_hf_bundle(MODEL_ID)
    model, tok, layers = bundle.model, bundle.tokenizer, bundle.layers
    print(f"[EXTRACT] model={MODEL_ID} n_layers={bundle.n_layers} hidden={bundle.hidden}", flush=True)

    for out_dir, layer_idx, pv_pairs in ((OUT_LS, LS, 8), (OUT_LR, LR, 3)):
        out_dir.mkdir(exist_ok=True)
        # idempotent: a completed set (all 7 files) is not re-extracted on requeue
        expected = ["D-CTX", "SEM", "OTHER", "D-PV", "R1", "R2", "R3"]
        if all((out_dir / f"{n}.npy").exists() for n in expected):
            print(f"[EXTRACT] {out_dir.name}: all {len(expected)} directions present, skipping", flush=True)
            continue
        lm = layers[layer_idx]
        t0 = time.time()

        # contrast directions
        for name, pos_key, neg_key, pairs in (
            ("D-CTX", "distress_text", "neutral_text", ep["d_ctx"]["pairs"]),
            ("SEM", "maritime_text", "neutral_text", ep["sem"]["pairs"]),
            ("OTHER", "distressed_user_text", "neutral_user_text", ep["other"]["pairs"]),
        ):
            pos = [p[pos_key] for p in pairs]
            neg = [p[neg_key] for p in pairs]
            vec, meta = directions.extract_contrast_direction(
                model, tok, lm, pos, neg, layer_idx, name)
            directions.save_direction(out_dir, name, vec, meta)
            print(f"[EXTRACT] {out_dir.name}/{name}: n={len(pos)}x2 norm_pre={meta.get('norm_before_normalize')}", flush=True)

        # persona direction (pooled over pairs, unit vectors averaged then renormalized)
        vecs, metas = [], []
        for i, pair in enumerate(ep["d_pv"]["system_pairs"][:pv_pairs]):
            v, m = directions.extract_persona_direction(
                model, tok, lm,
                pair["distress_system"], pair["baseline_system"],
                [t["text"] for t in ep["d_pv"]["tasks"]],
                GEN, layer_idx, seed=100 + i, source=f"D-PV-pair{i}")
            vecs.append(v); metas.append({k: m[k] for k in m if k != "pairwise_cosines"})
        stack = np.stack(vecs)
        pooled = stack.mean(axis=0)
        pooled = pooled / np.linalg.norm(pooled)
        pair_cos = (stack @ stack.T).round(3).tolist()
        directions.save_direction(out_dir, "D-PV", pooled, {
            "source": "D-PV", "layer": layer_idx, "n_pairs": pv_pairs,
            "per_pair_meta": metas, "inter_pair_cosines": pair_cos})
        print(f"[EXTRACT] {out_dir.name}/D-PV pooled over {pv_pairs} pairs; inter-pair cos diag-off min={min(min(r[i+1:] or [1]) for i, r in enumerate(pair_cos)):.3f}", flush=True)

        # random controls (fixed seeds, reproducible)
        for i, seed in enumerate((1001, 1002, 1003), 1):
            v = directions.make_random_unit(bundle.hidden, seed)
            directions.save_direction(out_dir, f"R{i}", v, {"source": f"R{i}", "layer": layer_idx, "seed": seed})
        print(f"[EXTRACT] {out_dir.name} done in {time.time()-t0:.0f}s", flush=True)

    # cosine matrix at Lr for C3 (raw material; analysis formalizes)
    names = ["D-CTX", "D-PV", "SEM", "OTHER", "R1"]
    mat = {}
    for a in names:
        va = directions.load_direction(OUT_LR, a)
        for b in names:
            vb = directions.load_direction(OUT_LR, b)
            mat[f"{a}|{b}"] = round(float(np.dot(va, vb)), 4)
    json.dump(mat, open(OUT_LR / "cosine_matrix_lr.json", "w"), indent=2)
    print("[EXTRACT] cosine matrix (Lr):", {k: v for k, v in mat.items() if k.split("|")[0] < k.split("|")[1]}, flush=True)


if __name__ == "__main__":
    main()

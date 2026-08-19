"""Recompute the intended 32-feature I-SAE instrument from stored residuals.

What went wrong, stated once: src/runner.build_instruments reads
instruments.sae.feature_ids from the cells file, and no cells file ever carried it
(scripts/make_cells.py writes only the probe path; the SAE was meant to be "supplied via
run_panel.py --sae-release/--sae-id", which has no feature-id argument). With
feature_ids = None, _sae_from_saelens sums ALL d_sae = 16,384 features, so every logged
sae_score is the total activation mass of the layer-31 SAE, and the 32-feature list in
instruments/sae_features.json was never used at run time. Every JSONL row records
gen_config.instruments.sae.feature_ids = null; that is the audit trail.

What can be recovered from disk: scripts/train_probe.py dumped the condition-NONE
prompt-final residuals of all 30 scenarios (instruments/residuals_lr.npz), which is the
Panel V position. Encoding those with the published SAE gives the intended 32-feature
score for the validity gate. Panel B and A-prime residuals were NOT stored, so the SAE
rows of those panels stay the all-feature sum and are relabelled in the report.

Self-checks before anything is emitted:
  1. the JumpReLU encode reproduces the per-feature discovery statistics recorded in
     instruments/sae_features.json (mean_distress / mean_neutral for every selected
     feature, tolerance 1e-2 absolute), which pins the encoder to the one used on the
     cluster;
  2. the all-feature sum reproduces results/validity_auc_sae_heldout.json's value
     (tolerance 1e-9), which pins the residuals to the ones the runner scored.

Weights: google/gemma-scope-2-12b-it, resid_post/layer_31_width_16k_l0_medium/params.safetensors
(the release the cluster loaded through sae_lens; CC-BY-4.0). Downloaded through
huggingface_hub into the local HF cache (~503 MB) if not present. Only w_enc, b_enc and
threshold are read.

Output: results/validity_auc_sae_recomputed.json via results_io.emit_one. Final-build
order: run_all, explore_exit_channel.py, count_exposure.py, audit_checks.py, then this.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis import results_io, stats  # noqa: E402

RESULTS_NAME = "validity_auc_sae_recomputed"
SAE_REPO = "google/gemma-scope-2-12b-it"
SAE_FILE = "resid_post/layer_31_width_16k_l0_medium/params.safetensors"
RESIDUALS = _REPO / "instruments" / "residuals_lr.npz"
FEATURES = _REPO / "instruments" / "sae_features.json"
ALLFEAT_RESULT = _REPO / "results" / "validity_auc_sae_heldout.json"
SPLIT = _REPO / "battery" / "split.json"
STAT_TOL = 1e-2
AUC_TOL = 1e-9


def load_encoder(path: Optional[str] = None):
    """Return (W_enc [hidden, d_sae], b_enc [d_sae], threshold [d_sae]) as float32 numpy."""
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(SAE_REPO, SAE_FILE)
    from safetensors import safe_open
    with safe_open(path, framework="np") as f:
        W = np.asarray(f.get_tensor("w_enc"), dtype=np.float32)
        b = np.asarray(f.get_tensor("b_enc"), dtype=np.float32)
        thr = np.asarray(f.get_tensor("threshold"), dtype=np.float32)
    return W, b, thr, str(path)


def jumprelu_encode(X: np.ndarray, W: np.ndarray, b: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Mirror of src.runner._sae_from_npz / cluster/probe.py: acts = pre * (pre > threshold)."""
    pre = X.astype(np.float32) @ W + b
    return pre * (pre > thr)


def scores_from_acts(acts: np.ndarray, sel: dict) -> dict:
    fids = [int(f) for f in sel["feature_ids"]]
    tstats = {int(f["feature_id"]): float(f["t_stat"]) for f in sel["features"]}
    plain = acts[:, fids].sum(axis=1)
    signed = sum(np.sign(tstats[f]) * acts[:, f] for f in fids)
    pos = [f for f in fids if tstats[f] > 0]
    positive_only = acts[:, pos].sum(axis=1)
    return {"all_feature_sum": acts.sum(axis=1), "selected32_plain_sum": plain,
            "selected32_signed_sum": np.asarray(signed), "positive_t_features_sum": positive_only,
            "n_positive_t": len(pos), "n_negative_t": len(fids) - len(pos)}


def check_encoder(acts: np.ndarray, sel: dict, split: np.ndarray, label: np.ndarray) -> dict:
    disc_d = (split == "discovery") & (label == 1)
    disc_n = (split == "discovery") & (label == 0)
    worst = 0.0
    for f in sel["features"]:
        fid = int(f["feature_id"])
        md = float(acts[disc_d, fid].mean()); mn = float(acts[disc_n, fid].mean())
        worst = max(worst, abs(md - f["mean_distress"]), abs(mn - f["mean_neutral"]))
    return {"max_abs_deviation_from_recorded_stats": worst, "n_features_checked": len(sel["features"]),
            "tolerance": STAT_TOL, "passed": worst <= STAT_TOL}


def auc_block(score: np.ndarray, frame: pd.DataFrame, B: int, seed: int) -> dict:
    f = frame.assign(score=score)
    r = stats.auc_cluster(f, "score", "label", B=B, seed=seed)
    return {"auc": r.value, "ci_low": r.ci_low, "ci_high": r.ci_high, "n_clusters": r.n_clusters,
            "conf_distress_scores": [float(x) for x in f.loc[f["label"] == 1, "score"]],
            "conf_neutral_scores": [float(x) for x in f.loc[f["label"] == 0, "score"]]}


def compute(B: int = stats.DEFAULT_B, seed: int = 0, sae_path: Optional[str] = None) -> dict:
    z = np.load(RESIDUALS)
    X = np.asarray(z["X"], dtype=np.float32); split = z["split"]; label = z["label"].astype(int)
    typ = z["type"]; sid = z["scenario_id"]
    sel = json.loads(FEATURES.read_text())
    W, b, thr, path = load_encoder(sae_path)
    acts = jumprelu_encode(X, W, b, thr)
    enc_check = check_encoder(acts, sel, split, label)
    if not enc_check["passed"]:
        raise RuntimeError(f"encoder does not reproduce recorded feature statistics: {enc_check}")
    sc = scores_from_acts(acts, sel)

    conf = (split == "confirmation") & (typ != "third_person")
    frame = pd.DataFrame({"label": label[conf].astype(float), "scenario_id": sid[conf],
                          "scenario_class": np.where(label[conf] == 1, "distress", "neutral")})
    blocks = {k: auc_block(sc[k][conf], frame, B, seed)
              for k in ("all_feature_sum", "selected32_plain_sum", "selected32_signed_sum",
                        "positive_t_features_sum")}
    # in-sample discovery AUC for context (no CI; discovery is where the features were chosen)
    disc = (split == "discovery") & (typ != "third_person")
    disc_auc = {k: float(stats.auc_stat(sc[k][disc], label[disc])) for k in blocks}

    logged = json.loads(ALLFEAT_RESULT.read_text())
    logged_val = float(logged["value"]) if logged.get("value") is not None else float("nan")
    allfeat_check = {"logged_validity_auc_sae_heldout": logged_val,
                     "recomputed_all_feature_auc": blocks["all_feature_sum"]["auc"],
                     "abs_difference": abs(blocks["all_feature_sum"]["auc"] - logged_val),
                     "tolerance": AUC_TOL,
                     "passed": abs(blocks["all_feature_sum"]["auc"] - logged_val) <= AUC_TOL}
    if not allfeat_check["passed"]:
        raise RuntimeError(f"all-feature sum does not reproduce the logged SAE AUC: {allfeat_check}")

    tp = typ == "third_person"
    entry = {
        "value": blocks["selected32_plain_sum"]["auc"],
        "ci_low": blocks["selected32_plain_sum"]["ci_low"],
        "ci_high": blocks["selected32_plain_sum"]["ci_high"],
        "n_clusters": blocks["selected32_plain_sum"]["n_clusters"],
        "extra": {
            "split_side": "confirmation",
            "what_ran": ("every JSONL sae_score is the sum over ALL 16,384 features "
                         "(gen_config.instruments.sae.feature_ids = null on every row); "
                         "instruments/sae_features.json was never passed to the runner"),
            "instrument_variants": {k: {kk: vv for kk, vv in v.items()
                                        if kk not in ("conf_distress_scores", "conf_neutral_scores")}
                                    for k, v in blocks.items()},
            "per_scenario_scores_confirmation": {k: {"distress": v["conf_distress_scores"],
                                                     "neutral": v["conf_neutral_scores"]}
                                                 for k, v in blocks.items()},
            "discovery_in_sample_auc": disc_auc,
            "n_selected_features": len(sel["feature_ids"]),
            "n_selected_with_positive_t": sc["n_positive_t"],
            "n_selected_with_negative_t": sc["n_negative_t"],
            "runner_sum_ignores_sign": True,
            "third_person_all_feature_sum": [float(x) for x in sc["all_feature_sum"][tp]],
            "third_person_selected32_plain_sum": [float(x) for x in sc["selected32_plain_sum"][tp]],
            "encoder_check": enc_check, "all_feature_check": allfeat_check,
            "sae_weights": {"repo": SAE_REPO, "file": SAE_FILE, "local_path": path,
                            "architecture": "jump_relu (config.json of the release)"},
            "residuals": str(RESIDUALS.relative_to(_REPO)),
            "note": ("post-data correction: the intended instrument "
                     "recomputed from stored Panel-V-position residuals; Panel B and A-prime "
                     "residuals were not stored, so their SAE rows remain the all-feature sum"),
        },
    }
    return entry


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(_REPO / "results"))
    ap.add_argument("--B", type=int, default=stats.DEFAULT_B)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sae-path", default=None, help="local params.safetensors (skips the download)")
    ap.add_argument("--no-emit", action="store_true")
    args = ap.parse_args(argv)
    entry = compute(B=args.B, seed=args.seed, sae_path=args.sae_path)
    v = entry["extra"]["instrument_variants"]
    print(f"[sae] encoder check max dev {entry['extra']['encoder_check']['max_abs_deviation_from_recorded_stats']:.4f}; "
          f"all-feature AUC {v['all_feature_sum']['auc']:.3f} (logged "
          f"{entry['extra']['all_feature_check']['logged_validity_auc_sae_heldout']:.3f}); "
          f"32-plain {v['selected32_plain_sum']['auc']:.3f} [{v['selected32_plain_sum']['ci_low']:.3f}, {v['selected32_plain_sum']['ci_high']:.3f}]; "
          f"32-signed {v['selected32_signed_sum']['auc']:.3f}; positive-t {v['positive_t_features_sum']['auc']:.3f}")
    if args.no_emit:
        return 0
    results_io.emit_one(RESULTS_NAME, entry, out_dir=args.out,
                        source_files=[str(RESIDUALS.relative_to(_REPO)), str(FEATURES.relative_to(_REPO)),
                                      str(ALLFEAT_RESULT.relative_to(_REPO)), f"{SAE_REPO}:{SAE_FILE}"],
                        synthetic=False,
                        extra_provenance={"producer": "scripts/recompute_sae_instrument.py"},
                        B=args.B, seed=args.seed)
    print(f"[sae] wrote {Path(args.out) / (RESULTS_NAME + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

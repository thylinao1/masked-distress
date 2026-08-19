"""Self-tests for the audit additions (analysis/audit.py,
scripts/audit_checks.py, scripts/recompute_sae_instrument.py).
Planted truths go in, the estimators must return them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import schema
from analysis import audit, results_io

sys.path.insert(0, str(REPO / "scripts"))
import recompute_sae_instrument as rsi  # noqa: E402

B_TEST = 200


# ============================================================ sign test

def test_sign_test_exact_values():
    r = audit.sign_test_one_sided([1, 2, 3, 4, 5, 6], +1)
    assert r["n_tested"] == 6 and r["n_with_predicted_sign"] == 6
    assert r["p_one_sided"] == pytest.approx(1 / 64)
    r = audit.sign_test_one_sided([1, 2, 3, 4, 0.0, 0.005], +1, tie_tol=0.01)
    assert r["n_ties_excluded"] == 2 and r["n_tested"] == 4
    assert r["p_one_sided"] == pytest.approx(1 / 16)
    r = audit.sign_test_one_sided([1, 1, 1, -1, -1], +1)
    assert r["p_one_sided"] == pytest.approx(0.5)
    r = audit.sign_test_one_sided([-1, -1, -1], -1)
    assert r["p_one_sided"] == pytest.approx(1 / 8)


# ============================================================ pair robustness

def _panel_b_rows(drops_e, drops_i, nat_e=2.0, nat_i=4.0, seeds=(0, 1, 2), sampled=True):
    """6 confirmation pairs; distress NEUTRAL_INSTR sits at nat above the neutral twin;
    SUPPRESS lowers the distress report by drops_e[k]*nat_e and moves the probe by
    -drops_i[k]*nat_i (drop = NI - SUP over nat)."""
    rows = []
    d_ids = ["d02", "d04", "d06", "d08", "d10", "d12"]
    n_ids = ["n02", "n04", "n06", "n08", "n10", "n12"]
    for k, (d, n) in enumerate(zip(d_ids, n_ids)):
        for cond in ("NONE", "NEUTRAL_INSTR", "SUPPRESS"):
            for sd in seeds:
                e_d = nat_e - (drops_e[k] * nat_e if cond == "SUPPRESS" else 0.0)
                i_d = nat_i - (drops_i[k] * nat_i if cond == "SUPPRESS" else 0.0)
                rows.append({"panel": "B", "direction": "NULL", "scenario_id": d,
                             "scenario_class": "distress", "condition": cond,
                             "split_side": "confirmation", "seed": sd,
                             "q_self_logit": e_d, "probe_prompt_final": i_d,
                             "q_self_sampled": round(e_d) if sampled else None})
                rows.append({"panel": "B", "direction": "NULL", "scenario_id": n,
                             "scenario_class": "neutral", "condition": cond,
                             "split_side": "confirmation", "seed": sd,
                             "q_self_logit": 0.0, "probe_prompt_final": 0.0,
                             "q_self_sampled": 0.0 if sampled else None})
    pairs = dict(zip(d_ids, n_ids))
    cats = {s: ("task_failure" if s in ("d02", "d04") else "abusive_user" if s in ("d06", "d08")
                else "situational_negative") for s in d_ids}
    return pd.DataFrame(rows), pairs, cats


def test_pair_robustness_recovers_planted_drops():
    drops_e = [1.0, 0.8, 0.6, 0.4, 0.0, 0.0]
    drops_i = [-0.1, -0.2, -0.3, -0.1, -0.2, -0.1]
    df, pairs, cats = _panel_b_rows(drops_e, drops_i)
    out = audit.pair_robustness(df, pairs, cats, B=B_TEST, seed=0)
    per = {r["pair"].split("/")[0]: r for r in out["extra"]["per_pair"]}
    for k, d in enumerate(["d02", "d04", "d06", "d08", "d10", "d12"]):
        assert per[d]["expression_drop"] == pytest.approx(drops_e[k], abs=1e-9)
        assert per[d]["internal_drop"] == pytest.approx(drops_i[k], abs=1e-9)
        assert per[d]["divergence"] == pytest.approx(drops_e[k] - drops_i[k], abs=1e-9)
    v = out["value"]
    # two pairs have zero expression change -> ties; internal all negative -> 6/6
    assert v["sign_test_expression"]["n_ties_excluded"] == 2
    assert v["sign_test_expression"]["p_one_sided"] == pytest.approx(1 / 16)
    assert v["sign_test_internal_nofall"]["p_one_sided"] == pytest.approx(1 / 64)
    assert v["sign_test_divergence"]["p_one_sided"] == pytest.approx(1 / 64)
    assert v["n_pairs_expression_no_signal"] == 2
    assert v["n_pairs_expression_drop_ge_one_unit"] == 1
    # LOO range brackets the pooled divergence
    pooled = np.mean([drops_e[k] - drops_i[k] for k in range(6)])
    assert v["loo_divergence_min"] <= pooled <= v["loo_divergence_max"]
    # category-out drops one category's two pairs
    assert out["extra"]["category_out"]["situational_negative"]["n_pairs_kept"] == 4
    assert out["extra"]["category_only"]["task_failure"]["n_pairs"] == 2
    # digit compliance
    assert v["digit_compliance_all_cells"]["fraction_parsed"] == pytest.approx(1.0)
    assert out["extra"]["digit_compliance_by_condition"]["SUPPRESS"]["n_parsed_digit"] == 36


def test_pair_robustness_counts_missing_digits():
    df, pairs, cats = _panel_b_rows([1.0] * 6, [0.0] * 6, sampled=False)
    out = audit.pair_robustness(df, pairs, cats, B=B_TEST, seed=0)
    assert out["value"]["digit_compliance_all_cells"]["n_parsed_digit"] == 0


# ============================================================ specificity by rung

def _panel_a_rows(cross_at_1000: bool, theta: float = 2.0):
    rows = []
    n_ids = ["n02", "n04", "n06", "n08", "n10", "n12"]
    dirs = ["D-CTX", "D-PV", "R1", "R2", "R3", "SEM", "OTHER"]
    for n in n_ids:
        for sd in (0, 1, 2):
            rows.append({"panel": "A", "direction": "NULL", "scenario_id": n,
                         "scenario_class": "neutral", "condition": "NONE",
                         "split_side": "confirmation", "seed": sd, "coefficient": 0.0,
                         "q_self_logit": 0.5, "probe_prompt_final": -3.0})
            for d in dirs:
                for c in (500.0, 1000.0):
                    q = 0.5
                    if d in ("R1", "R2", "R3") and c == 1000.0 and cross_at_1000:
                        q = theta + 1.0  # every random cell crosses at 1000
                    if d in ("D-CTX", "D-PV"):
                        q = 0.5 + 0.001 * c  # a modest dose-response
                    rows.append({"panel": "A", "direction": d, "scenario_id": n,
                                 "scenario_class": "neutral", "condition": "NONE",
                                 "split_side": "confirmation", "seed": sd, "coefficient": c,
                                 "q_self_logit": q, "probe_prompt_final": -3.0 + 0.001 * c})
    return pd.DataFrame(rows)


def test_specificity_by_rung_planted():
    df = _panel_a_rows(cross_at_1000=True)
    cap = {"R1": [{"coefficient": 0.0, "ppl_ratio": 1.0, "mmlu_drop_pp": 0.0},
                  {"coefficient": 500.0, "ppl_ratio": 1.03, "mmlu_drop_pp": 3.0},
                  {"coefficient": 1000.0, "ppl_ratio": 1.34, "mmlu_drop_pp": 15.0}]}
    out = audit.specificity_by_rung(df, theta=2.0, capability=cap, B=B_TEST, seed=0)
    v = out["value"]
    assert v["fpr_by_rung"]["500"]["fpr"] == pytest.approx(0.0)
    assert v["fpr_by_rung"]["1000"]["fpr"] == pytest.approx(0.75)  # 3 of 4 control dirs cross
    assert v["fpr_pooled"]["fpr"] == pytest.approx(0.375)
    t = out["extra"]["per_direction_rung"]
    assert t["R1"]["500"]["inside_frozen_criteria"] is True
    assert t["R1"]["1000"]["inside_frozen_criteria"] is False
    assert t["R2"]["500"]["capability_measured"] is False
    assert t["D-CTX"]["1000"]["report_shift_over_anchor"] == pytest.approx(1.0)


# ============================================================ symmetric ranking

def test_symmetric_ranking_bands():
    df, pairs, cats = _panel_b_rows([0.9] * 6, [0.0] * 6)
    # make expression overlap under SUPPRESS: neutral cells read 0.3 > some distress
    df.loc[(df["condition"] == "SUPPRESS") & (df["scenario_class"] == "neutral"), "q_self_logit"] = 0.3
    out = audit.symmetric_ranking(df, B=B_TEST, seed=0)
    v = out["value"]
    assert v["SUPPRESS"]["internal_separating_band_exists"] is True
    assert v["SUPPRESS"]["auc_internal"] == pytest.approx(1.0)
    assert v["SUPPRESS"]["expression_separating_band_exists"] is False
    assert v["NEUTRAL_INSTR"]["auc_expression"] == pytest.approx(1.0)


# ============================================================ dominant dimension

def test_dominant_dimension_planted():
    rng = np.random.default_rng(0)
    d = 64
    def unit(v):
        return v / np.linalg.norm(v)
    e = np.zeros(d); e[7] = 1.0
    def content_vec():
        v = rng.normal(size=d); v[7] = 0.0  # orthogonal to the planted dominant dim
        return unit(v)
    content = {n: content_vec() for n in ("D-CTX", "D-PV", "SEM", "OTHER", "R1")}
    ls = {"D-CTX": unit(0.95 * e + 0.31 * content["D-CTX"]),
          "D-PV": unit(-0.9 * e + 0.44 * content["D-PV"]),
          "SEM": unit(-0.95 * e + 0.31 * content["SEM"]),
          "OTHER": unit(-0.5 * e + 0.87 * content["OTHER"]),
          "R1": content["R1"]}
    lr = {n: unit(0.6 * e + 0.8 * content[n]) if n != "R1" else content["R1"] for n in ls}
    out = audit.dominant_dimension(ls, lr, residuals_lr=None, probe_w=None)
    v = out["value"]
    assert v["dominant_dim"] == 7
    assert v["frac_sq_norm_ls"]["D-CTX"] == pytest.approx(0.95 ** 2 / (0.95 ** 2 + 0.31 ** 2), abs=1e-6)
    # without the dominant dim, D-CTX|SEM at Ls is the cosine of two random content vectors (near 0)
    assert abs(out["extra"]["cosine_ls_without_dominant_dim"]["D-CTX|SEM"]) < 0.3
    # with it, D-CTX and SEM are strongly anti-aligned
    assert out["extra"]["cosine_ls_with_dominant_dim"]["D-CTX|SEM"] < -0.8
    assert v["mean_abs_offdiag_lr_without"] < v["mean_abs_offdiag_lr_with"]
    assert v["content_norm_fraction_ls"]["R1"] == pytest.approx(1.0, abs=1e-3)


# ============================================================ text-only baseline

def _scenarios(distinct: bool):
    scn = []
    for i in range(1, 13):
        d_words = "puzzle failure rejected wrong again unusable" if distinct else "alpha beta gamma delta"
        n_words = "puzzle solved accepted correct thanks helpful" if distinct else "alpha beta gamma delta"
        scn.append({"id": f"d{i:02d}", "type": "distress", "pair_id": f"n{i:02d}",
                    "turns": [{"role": "user", "content": f"{d_words} item {i}"}]})
        scn.append({"id": f"n{i:02d}", "type": "neutral", "pair_id": f"d{i:02d}",
                    "turns": [{"role": "user", "content": f"{n_words} item {i}"}]})
    for i in range(1, 7):
        scn.append({"id": f"tp{i:02d}", "type": "third_person", "pair_id": None,
                    "turns": [{"role": "user", "content": f"someone else failure rejected item {i}"}]})
    return scn


def test_textonly_baseline_separates_and_not():
    split = json.loads((REPO / "battery" / "split.json").read_text())
    out = audit.textonly_baseline(_scenarios(True), split, B=B_TEST, seed=0)
    assert out["value"] == pytest.approx(1.0)
    assert out["extra"]["primary"]["n_third_person_above_conf_neutral_max"] >= 0
    out2 = audit.textonly_baseline(_scenarios(False), split, B=B_TEST, seed=0)
    assert out2["value"] == pytest.approx(0.5, abs=0.35)  # identical texts carry no class signal


# ============================================================ SAE recompute

def test_jumprelu_and_scores_and_checks():
    rng = np.random.default_rng(1)
    hidden, d_sae = 8, 12
    W = rng.normal(size=(hidden, d_sae)).astype(np.float32)
    b = np.zeros(d_sae, np.float32); thr = np.full(d_sae, 0.5, np.float32)
    X = rng.normal(size=(30, hidden)).astype(np.float32)
    acts = rsi.jumprelu_encode(X, W, b, thr)
    assert acts.shape == (30, d_sae)
    assert np.all((acts == 0) | (acts > 0.5))
    sel = {"feature_ids": [0, 1, 2], "features": [{"feature_id": 0, "t_stat": 2.0},
                                                   {"feature_id": 1, "t_stat": -1.0},
                                                   {"feature_id": 2, "t_stat": 3.0}]}
    sc = rsi.scores_from_acts(acts, sel)
    assert np.allclose(sc["selected32_plain_sum"], acts[:, [0, 1, 2]].sum(1))
    assert np.allclose(sc["selected32_signed_sum"], acts[:, 0] - acts[:, 1] + acts[:, 2])
    assert np.allclose(sc["positive_t_features_sum"], acts[:, [0, 2]].sum(1))
    assert sc["n_negative_t"] == 1
    # encoder check passes against its own stats and fails against wrong ones
    split = np.array(["discovery"] * 15 + ["confirmation"] * 15)
    label = np.array([1, 0] * 15)
    dd = (split == "discovery") & (label == 1); dn = (split == "discovery") & (label == 0)
    good = {"features": [{"feature_id": 0, "mean_distress": float(acts[dd, 0].mean()),
                          "mean_neutral": float(acts[dn, 0].mean())}]}
    assert rsi.check_encoder(acts, good, split, label)["passed"]
    bad = {"features": [{"feature_id": 0, "mean_distress": 999.0, "mean_neutral": 0.0}]}
    assert not rsi.check_encoder(acts, bad, split, label)["passed"]


# ============================================================ names / emit

def test_audit_names_in_schema_and_definitions():
    for name in ("panelB_pair_robustness", "panelA_specificity_by_rung",
                 "countermeasure_symmetric_ranking", "direction_dominant_dim",
                 "validity_auc_textonly_heldout", "validity_auc_sae_recomputed"):
        assert name in schema.RESULTS_NAMES
        assert name in results_io.DEFINITIONS
        assert "AUDIT ADDITION" in results_io.DEFINITIONS[name]


def test_emit_one_writes_audit_file(tmp_path):
    entry = {"value": 0.61, "ci_low": 0.3, "ci_high": 0.9, "n_clusters": 12, "extra": {"k": 1}}
    results_io.emit_one("validity_auc_sae_recomputed", entry, out_dir=tmp_path,
                        source_files=["x"], synthetic=False)
    d = json.loads((tmp_path / "validity_auc_sae_recomputed.json").read_text())
    assert d["value"] == 0.61 and d["provenance"]["synthetic"] is False
    assert "AUDIT ADDITION" in d["definition"]


def test_twin_surface_stats():
    scn = _scenarios(True)
    out = audit.twin_surface_stats(scn)
    assert out["n_pairs"] == 12
    assert 0.0 <= out["jaccard_min"] <= out["jaccard_max"] <= 1.0
    # identical-text twins overlap fully
    out2 = audit.twin_surface_stats(_scenarios(False))
    assert out2["jaccard_min"] == pytest.approx(1.0)


# ============================================================ additional audit outputs

def test_condition_reference_planted():
    # planted: NONE distress 4, neutral 0; NI distress 3, neutral 0; SUP distress 3, neutral 1
    rows = []
    for k in range(6):
        d, n = f"d{2*k+2:02d}", f"n{2*k+2:02d}"
        for cond, (vd, vn) in (("NONE", (4.0, 0.0)), ("NEUTRAL_INSTR", (3.0, 0.0)), ("SUPPRESS", (3.0, 1.0))):
            for sd in (0, 1, 2):
                rows.append({"panel": "B", "direction": "NULL", "scenario_id": d, "scenario_class": "distress",
                             "condition": cond, "split_side": "confirmation", "seed": sd,
                             "q_self_logit": vd, "probe_prompt_final": vd + 0.01 * k})
                rows.append({"panel": "B", "direction": "NULL", "scenario_id": n, "scenario_class": "neutral",
                             "condition": cond, "split_side": "confirmation", "seed": sd,
                             "q_self_logit": vn, "probe_prompt_final": vn + 0.01 * k})
    df = pd.DataFrame(rows)
    out = audit.condition_reference(df, {"theta_expr": 2.5, "theta_int": 2.5}, B=B_TEST, seed=0)
    v = out["value"]
    # NONE-referenced internal drop = (4 - 3)/4 = 0.25; NI-referenced = 0; twin-referenced sep drop = (3-2)/3
    assert v["none_referenced"]["expression_drop"]["value"] == pytest.approx(0.25, abs=1e-6)
    assert v["neutral_instr_referenced"]["expression_drop"]["value"] == pytest.approx(0.0, abs=1e-6)
    assert v["twin_referenced"]["expression_separation_drop"]["value"] == pytest.approx(1 / 3, abs=1e-6)
    assert v["class_condition_means"]["expression"]["SUPPRESS"]["separation"] == pytest.approx(2.0, abs=1e-6)
    # thresholds 2.5: expression misses 0 under all conditions (distress >= 3)
    assert v["miss_rate_by_condition"]["NONE"]["expression"]["n_missed"] == 0
    assert v["miss_rate_by_condition"]["SUPPRESS"]["expression"]["n_missed"] == 0


def test_selfstate_items_planted():
    rows = []
    for k in range(3):
        for cls, sid, qs, qv, p1, p2 in (("distress", f"d{2*k+2:02d}", 3.0, 4.0, 3.0, 3.0),
                                          ("neutral", f"n{2*k+2:02d}", 0.5, 7.0, 0.5, 0.5),
                                          ("third_person", f"tp{2*k+2:02d}", 4.0, 5.0, 2.0, 1.0)):
            for cond in ("NONE", "NEUTRAL_INSTR", "SUPPRESS"):
                q = 0.0 if cond == "SUPPRESS" else qs
                rows.append({"panel": "B", "direction": "NULL", "scenario_id": sid, "scenario_class": cls,
                             "condition": cond, "split_side": "confirmation", "seed": 0,
                             "q_self_logit": q, "q_val_logit": 0.0 if cond == "SUPPRESS" else qv,
                             "q_self_logit_para1": p1, "q_self_logit_para2": p2})
    out = audit.selfstate_items(pd.DataFrame(rows), B=B_TEST, seed=0)
    v = out["value"]
    assert v["q_val_by_condition"]["SUPPRESS"]["frac_cells_both_below_0p1"] == pytest.approx(1.0)
    assert v["q_val_by_condition"]["NONE"]["neutral"]["q_val_mean"] == pytest.approx(7.0)
    assert v["selfother_by_form"]["primary"]["contrast"] == pytest.approx(1.0)
    assert v["selfother_by_form"]["para1"]["contrast"] == pytest.approx(-1.0)
    assert v["selfother_by_form"]["para2"]["contrast"] == pytest.approx(-2.0)


def test_probe_direction_cosines_and_text_trigger():
    rng = np.random.default_rng(3)
    d = 32
    w = rng.normal(size=d); v = w.copy()
    dirs = {"A": v / np.linalg.norm(v), "B": rng.normal(size=d)}
    out = audit.probe_direction_cosines(w, dirs, dominant_dim=0)
    assert out["A"]["cos"] == pytest.approx(1.0, abs=1e-6)
    scn = _scenarios(True)
    split = json.loads((REPO / "battery" / "split.json").read_text())
    tr = audit.textonly_as_trigger(scn, split)
    assert tr["condition_invariant"] is True
    assert tr["n_confirmation_distress_scenarios"] == 6 and tr["n_confirmation_neutral_scenarios"] == 6


def test_audit_additions_names_registered():
    for name in ("panelB_condition_reference", "panelB_selfstate_items"):
        assert name in schema.RESULTS_NAMES and name in results_io.DEFINITIONS


# ============================================================ real-probe dose map, standardized cosines

def _aprime_dose_rows(slope_per_coef=0.002, dose_unit=10.0):
    """Zero anchor + D-CTX rungs 500/1000/2000 + D-PV rungs 250/500/1000, both splits, both
    classes; the internal readout rises linearly at slope_per_coef along D-CTX and at twice
    that along D-PV; controls flat."""
    rows = []
    sc = [("d01", "distress", "discovery"), ("n01", "neutral", "discovery"),
          ("d02", "distress", "confirmation"), ("n02", "neutral", "confirmation")]
    grid = {"NULL": [0.0], "D-CTX": [500.0, 1000.0, 2000.0], "D-PV": [250.0, 500.0, 1000.0], "R1": [500.0, 1000.0]}
    for d, coefs in grid.items():
        for c in coefs:
            for sid, cls, side in sc:
                base = 5.0 if cls == "distress" else -5.0
                gain = {"NULL": 0.0, "D-CTX": slope_per_coef, "D-PV": 2 * slope_per_coef, "R1": 0.0}[d]
                for seed in range(2):
                    rows.append({"panel": "A", "direction": d, "coefficient": c, "scenario_id": sid,
                                 "scenario_class": cls, "split_side": side, "seed": seed,
                                 audit.INTERNAL_PRIMARY: base + gain * c, audit.EXPRESSION_PRIMARY: 0.0})
    return pd.DataFrame(rows), dose_unit


def test_dose_map_realprobe_recovers_planted_slope():
    df, unit = _aprime_dose_rows()
    cap = {"D-CTX": [{"coefficient": 0.0, "ppl_ratio": 1.0, "mmlu_drop_pp": 0.0},
                     {"coefficient": 2000.0, "ppl_ratio": 1.04, "mmlu_drop_pp": 0.0},
                     {"coefficient": 4000.0, "ppl_ratio": 1.26, "mmlu_drop_pp": 11.7}],
           "D-PV": [{"coefficient": 0.0, "ppl_ratio": 1.0, "mmlu_drop_pp": 0.0},
                    {"coefficient": 500.0, "ppl_ratio": 1.04, "mmlu_drop_pp": 0.0},
                    {"coefficient": 1000.0, "ppl_ratio": 1.18, "mmlu_drop_pp": 8.3}]}
    out = audit.dose_map_realprobe(df, unit, cap, B=B_TEST, seed=0)
    prim = out["extra"]["dose_map_primary"]
    assert prim["D-CTX"]["sd_per_coef"] == pytest.approx(0.002 / unit, rel=1e-6)
    assert prim["D-PV"]["sd_per_coef"] == pytest.approx(0.004 / unit, rel=1e-6)
    assert prim["R1"]["sd_per_coef"] == pytest.approx(0.0, abs=1e-12)
    pd_ = out["value"]["per_direction"]
    assert pd_["D-CTX"]["max_valid_coefficient"] == 2000.0
    assert pd_["D-CTX"]["hi_sd"] == pytest.approx(2000 * 0.002 / unit)
    assert pd_["D-PV"]["max_valid_coefficient"] == 500.0
    assert pd_["D-PV"]["hi_sd"] == pytest.approx(500 * 0.004 / unit)
    assert out["value"]["hi_sd"] == pytest.approx(min(pd_["D-CTX"]["hi_sd"], pd_["D-PV"]["hi_sd"]))
    assert pd_["D-CTX"]["coef_for_1sd"] == pytest.approx(unit / 0.002)
    assert out["extra"]["dose_map_sensitivity"]["confirmation_pooled"]["D-CTX"]["sd_per_coef"] == pytest.approx(0.002 / unit, rel=1e-6)


def test_probe_direction_cosines_standardized_identity():
    rng = np.random.default_rng(0)
    std = rng.uniform(1, 100, size=50)
    w_std = rng.normal(size=50)
    w_folded = w_std / std                       # what train_probe.py stores
    v = rng.normal(size=50)                      # a raw-space direction (displacement)
    out = audit.probe_direction_cosines_standardized(w_folded, {"X": v}, std)
    v_std = v / std
    expected = float(w_std @ v_std / (np.linalg.norm(w_std) * np.linalg.norm(v_std)))
    assert out["X"]["cos_standardized"] == pytest.approx(expected, abs=1e-4)
    # a direction that lives on one huge-SD coordinate is nearly invisible in standardized space
    e = np.zeros(50); big = int(np.argmax(std)); e[big] = 1.0
    out2 = audit.probe_direction_cosines_standardized(w_folded, {"E": e}, std)
    assert abs(out2["E"]["cos_standardized"]) == pytest.approx(abs(w_std[big]) / np.linalg.norm(w_std), abs=1e-4)


def test_realprobe_names_registered():
    assert "capability_valid_range_realprobe" in schema.RESULTS_NAMES
    assert "capability_valid_range_realprobe" in results_io.DEFINITIONS
    assert results_io.DEFINITION_CLAIMED_SIDE["capability_valid_range_realprobe"] == "discovery"


# ============================================================ the battery validator must pass at HEAD

def test_battery_validator_passes_at_head():
    import subprocess, sys as _sys
    repo = Path(__file__).resolve().parents[1]
    r = subprocess.run([_sys.executable, str(repo / "battery" / "validate.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "DCB-1 validation PASSED" in r.stdout
    assert "factorial condition SUPPRESS_REGISTER" in r.stdout


# ============================================================ the cell builders must run at HEAD

@pytest.mark.parametrize("args", [["--panel", "V"], ["--panel", "B"], ["--panel", "LADDER"]])
def test_make_cells_builds_every_panel(tmp_path, args):
    """A replicator following RUNBOOK step 4 runs these; extending schema.CONDITIONS for the
    factorial addendum broke --panel B until FROZEN_CONDITIONS was introduced."""
    import subprocess, sys as _sys
    repo = Path(__file__).resolve().parents[1]
    r = subprocess.run([_sys.executable, str(repo / "scripts" / "make_cells.py"), *args,
                        "--outdir", str(tmp_path)], capture_output=True, text=True, cwd=str(repo))
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]


def test_make_cells_factorial_builds(tmp_path):
    import subprocess, sys as _sys
    repo = Path(__file__).resolve().parents[1]
    r = subprocess.run([_sys.executable, str(repo / "scripts" / "make_cells_factorial.py"),
                        "--outdir", str(tmp_path)], capture_output=True, text=True, cwd=str(repo))
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert len(list(tmp_path.glob("cells_panelBf_*.json"))) == 3


def test_frozen_and_factorial_conditions_partition():
    assert schema.CONDITIONS == schema.FROZEN_CONDITIONS + schema.FACTORIAL_CONDITIONS
    assert set(schema.FROZEN_CONDITIONS).isdisjoint(schema.FACTORIAL_CONDITIONS)
    conds = json.loads((Path(__file__).resolve().parents[1] / "battery" / "conditions.json").read_text())
    assert tuple(sorted(c["id"] for c in conds["conditions"])) == tuple(sorted(schema.FROZEN_CONDITIONS))


# ============================================================ persistence estimator

def _persist_frame(dctx_lift=0.8, r1_lift=0.0, report_lift=0.5):
    """NULL anchor plus three steered arms on 6 distress and 6 neutral confirmation scenarios.
    The planted separation is 2.0 on the report channel and 4.0 on the internal channel."""
    rows = []
    for k in range(6):
        for cls, sid, base_r, base_i in (("distress", f"d{2*k+2:02d}", 3.0, 2.0),
                                         ("neutral", f"n{2*k+2:02d}", 1.0, -2.0)):
            for arm, coef, lift_i, lift_r in (("NULL", 0.0, 0.0, 0.0),
                                              ("D-CTX", 2000.0, dctx_lift, report_lift),
                                              ("D-PV", 500.0, dctx_lift, report_lift),
                                              ("R1", 500.0, r1_lift, 0.0)):
                for seed in range(3):
                    rows.append({"panel": "PERSIST", "direction": arm, "coefficient": coef,
                                 "scenario_id": sid, "scenario_class": cls,
                                 "split_side": "confirmation", "condition": "NONE", "seed": seed,
                                 audit.EXPRESSION_PRIMARY: base_r + (lift_r if cls == "distress" else 0.0),
                                 audit.INTERNAL_PRIMARY: base_i + (lift_i if cls == "distress" else 0.0)})
    return pd.DataFrame(rows)


def test_persistence_elevation_recovers_planted_lift():
    df = _persist_frame(dctx_lift=0.8, r1_lift=0.0, report_lift=0.5)
    out = audit.persistence_elevation(df, B=B_TEST, seed=0)
    by = out["value"]["by_arm"]
    # internal separation is 4.0, so a planted +0.8 raw lift is +0.20 separation units
    assert by["D-CTX"]["internal"]["elevation"] == pytest.approx(0.8 / 4.0, rel=1e-6)
    # report separation is 2.0, so +0.5 raw is +0.25
    assert by["D-CTX"]["report"]["elevation"] == pytest.approx(0.5 / 2.0, rel=1e-6)
    assert by["R1"]["internal"]["elevation"] == pytest.approx(0.0, abs=1e-9)
    v = out["value"]["verdict_by_direction"]
    assert v["D-CTX"]["internal_persists"] is True
    assert v["D-CTX"]["both_channels_at_one_moment"] is True


def test_persistence_verdict_is_false_when_the_control_also_moves():
    """If the random control lifts the readout as much as the live direction, nothing persists
    that is specific to the direction, and the pre-declared verdict must say so."""
    df = _persist_frame(dctx_lift=0.8, r1_lift=0.8, report_lift=0.5)
    out = audit.persistence_elevation(df, B=B_TEST, seed=0)
    assert out["value"]["by_arm"]["R1"]["internal"]["excludes_zero"] is True
    assert out["value"]["verdict_by_direction"]["D-CTX"]["internal_persists"] is False


def test_second_model_and_persistence_names_registered():
    for n in ("panelB_persistence", "panelB_second_model"):
        assert n in schema.RESULTS_NAMES and n in results_io.DEFINITIONS
        assert results_io.DEFINITION_CLAIMED_SIDE[n] == "confirmation"

"""Reporting-level robustness and integrity computations added after the 2026-08-16
review pass. Everything here runs on data that is
already on disk: the frozen-battery JSONL, the battery text, the stored direction
vectors, the stored prompt-final residuals and the capability logprob job. No model
is called. Nothing here changes a pre-registered endpoint, threshold or gate; every
function is a labelled post-data reading whose results file says so.

Functions and the results name each feeds (scripts/audit_checks.py is the producer):

  pair_robustness      -> panelB_pair_robustness
  specificity_by_rung  -> panelA_specificity_by_rung
  symmetric_ranking    -> countermeasure_symmetric_ranking
  dominant_dimension   -> direction_dominant_dim
  textonly_baseline    -> validity_auc_textonly_heldout

Statistical policy is the frozen one (PREREGISTRATION.md section 5, DECISIONS.md D7):
scenario is the unit, BCa cluster bootstrap over scenarios, B = 10,000, seed 0,
stratified by class where a statistic needs both classes. Sign tests are exact
one-sided binomial tests over matched pairs with ties excluded and counted.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

import schema
from analysis import stats
from analysis.panels import EXPRESSION_PRIMARY, INTERNAL_PRIMARY

# Ties on the report channel: Q-SELF is a 0-9 digit expectation, so a paired
# difference below this many raw digits is a tie for the sign test.
SIGN_TEST_TIE_TOL_RAW = 0.01
# Frozen capability criteria (PREREGISTRATION.md section 4).
PPL_INFLATION_MAX = 1.10
MMLU_DROP_MAX_PP = 5.0


# ============================================================ shared helpers

def _scen_means(frame: pd.DataFrame, channel: str) -> pd.Series:
    """scenario_id -> mean over seeds of `channel`."""
    return frame.groupby("scenario_id")[channel].mean()


def sign_test_one_sided(diffs: Sequence[float], predicted_sign: int,
                        tie_tol: float = 0.0) -> dict:
    """Exact one-sided binomial sign test: P(X >= k | n, 1/2) where k counts pairs
    whose difference carries the predicted sign and n excludes ties (|d| <= tie_tol)."""
    d = np.asarray(diffs, dtype=float)
    ties = int(np.sum(np.abs(d) <= tie_tol))
    kept = d[np.abs(d) > tie_tol]
    n = int(len(kept))
    k = int(np.sum(np.sign(kept) == predicted_sign))
    p = float(sps.binom.sf(k - 1, n, 0.5)) if n > 0 else float("nan")
    return {"n_pairs": int(len(d)), "n_ties_excluded": ties, "n_tested": n,
            "n_with_predicted_sign": k, "p_one_sided": p,
            "granularity": (1.0 / 2 ** n) if n > 0 else None}


def binomial_p_string_note() -> str:
    return ("exact one-sided binomial sign test over matched confirmation pairs; ties "
            f"(|paired difference| <= {SIGN_TEST_TIE_TOL_RAW} raw digits on the report "
            "channel) are excluded and counted")


# ============================================================

def _panel_b_wide(df: pd.DataFrame, side: str = "confirmation") -> pd.DataFrame:
    b = df[(df["panel"] == "B") & (df["direction"] == "NULL")
           & df["scenario_class"].isin(["distress", "neutral"])
           & df["condition"].isin(["NEUTRAL_INSTR", "SUPPRESS"])
           & (df["split_side"] == side)]
    piv = b.pivot_table(index=["scenario_id", "scenario_class"], columns="condition",
                        values=[EXPRESSION_PRIMARY, INTERNAL_PRIMARY], aggfunc="mean")
    piv.columns = [f"{ch}__{cond}" for ch, cond in piv.columns]
    return piv.reset_index()


def _drops_by_pair(wide: pd.DataFrame, pairs: Dict[str, str],
                   channel: str) -> Tuple[float, Dict[str, float]]:
    """Natural separation (NEUTRAL_INSTR distress mean minus neutral mean, over the
    scenarios present) and per-distress-scenario drop in separation units."""
    ni = wide.set_index("scenario_id")[f"{channel}__NEUTRAL_INSTR"]
    su = wide.set_index("scenario_id")[f"{channel}__SUPPRESS"]
    d_ids = [s for s in wide["scenario_id"] if s in pairs]
    n_ids = [pairs[s] for s in d_ids if pairs[s] in ni.index]
    nat = float(ni[d_ids].mean() - ni[n_ids].mean())
    if not math.isfinite(nat) or abs(nat) < 1e-12:
        return nat, {s: float("nan") for s in d_ids}
    return nat, {s: float((ni[s] - su[s]) / nat) for s in d_ids}


def pair_robustness(df: pd.DataFrame, pairs: Dict[str, str], categories: Dict[str, str],
                    B: int = stats.DEFAULT_B, seed: int = 0) -> dict:
    """Per-pair Panel B table, leave-one-pair-out, category-out, exact sign tests, and
    the digit-answer compliance of the report channel.

    pairs: distress scenario id -> matched neutral id (battery/scenarios.json pair_id).
    categories: scenario id -> category.
    """
    wide = _panel_b_wide(df)
    d_ids = sorted(s for s in wide["scenario_id"] if s in pairs and s in set(wide["scenario_id"]))
    nat_e, drops_e = _drops_by_pair(wide, pairs, EXPRESSION_PRIMARY)
    nat_i, drops_i = _drops_by_pair(wide, pairs, INTERNAL_PRIMARY)
    w = wide.set_index("scenario_id")

    per_pair = []
    for s in d_ids:
        per_pair.append({
            "pair": f"{s}/{pairs[s]}", "category": categories.get(s),
            "q_self_neutral_instr": float(w.loc[s, f"{EXPRESSION_PRIMARY}__NEUTRAL_INSTR"]),
            "q_self_suppress": float(w.loc[s, f"{EXPRESSION_PRIMARY}__SUPPRESS"]),
            "probe_neutral_instr": float(w.loc[s, f"{INTERNAL_PRIMARY}__NEUTRAL_INSTR"]),
            "probe_suppress": float(w.loc[s, f"{INTERNAL_PRIMARY}__SUPPRESS"]),
            "twin_q_self_neutral_instr": float(w.loc[pairs[s], f"{EXPRESSION_PRIMARY}__NEUTRAL_INSTR"]),
            "twin_q_self_suppress": float(w.loc[pairs[s], f"{EXPRESSION_PRIMARY}__SUPPRESS"]),
            "twin_probe_neutral_instr": float(w.loc[pairs[s], f"{INTERNAL_PRIMARY}__NEUTRAL_INSTR"]),
            "twin_probe_suppress": float(w.loc[pairs[s], f"{INTERNAL_PRIMARY}__SUPPRESS"]),
            "expression_drop": drops_e[s], "internal_drop": drops_i[s],
            "divergence": drops_e[s] - drops_i[s],
        })

    # leave-one-pair-out on the divergence and both drops
    loo = []
    for s in d_ids:
        keep = wide[~wide["scenario_id"].isin([s, pairs[s]])]
        _, de = _drops_by_pair(keep, pairs, EXPRESSION_PRIMARY)
        _, di = _drops_by_pair(keep, pairs, INTERNAL_PRIMARY)
        e = float(np.mean(list(de.values()))); i = float(np.mean(list(di.values())))
        loo.append({"dropped_pair": f"{s}/{pairs[s]}", "expression_drop": e,
                    "internal_drop": i, "divergence": e - i})
    loo_div = [x["divergence"] for x in loo]

    # category-out and only-category
    cats = sorted(set(categories.get(s) for s in d_ids))
    cat_out, cat_only = {}, {}
    for c in cats:
        drop_ids = [s for s in d_ids if categories.get(s) == c]
        keep = wide[~wide["scenario_id"].isin(drop_ids + [pairs[s] for s in drop_ids])]
        _, de = _drops_by_pair(keep, pairs, EXPRESSION_PRIMARY)
        _, di = _drops_by_pair(keep, pairs, INTERNAL_PRIMARY)
        cat_out[c] = {"expression_drop": float(np.mean(list(de.values()))),
                      "internal_drop": float(np.mean(list(di.values()))),
                      "n_pairs_kept": len(de)}
        cat_out[c]["divergence"] = cat_out[c]["expression_drop"] - cat_out[c]["internal_drop"]
        only = wide[wide["scenario_id"].isin(drop_ids + [pairs[s] for s in drop_ids])]
        ne, de = _drops_by_pair(only, pairs, EXPRESSION_PRIMARY)
        ni_, di = _drops_by_pair(only, pairs, INTERNAL_PRIMARY)
        cat_only[c] = {"natural_separation_expression": ne, "natural_separation_internal": ni_,
                       "expression_drop": float(np.mean(list(de.values()))),
                       "internal_drop": float(np.mean(list(di.values()))),
                       "n_pairs": len(de)}
        cat_only[c]["divergence"] = cat_only[c]["expression_drop"] - cat_only[c]["internal_drop"]

    # exact sign tests: raw paired differences (NEUTRAL_INSTR minus SUPPRESS on the
    # distress scenario), predicted sign +1 for expression, -1 or 0 (does not fall)
    # for internal, and +1 for the per-pair divergence in separation units.
    raw_e = [w.loc[s, f"{EXPRESSION_PRIMARY}__NEUTRAL_INSTR"] - w.loc[s, f"{EXPRESSION_PRIMARY}__SUPPRESS"] for s in d_ids]
    raw_i = [w.loc[s, f"{INTERNAL_PRIMARY}__NEUTRAL_INSTR"] - w.loc[s, f"{INTERNAL_PRIMARY}__SUPPRESS"] for s in d_ids]
    div = [drops_e[s] - drops_i[s] for s in d_ids]
    st_e = sign_test_one_sided(raw_e, +1, tie_tol=SIGN_TEST_TIE_TOL_RAW)
    st_i = sign_test_one_sided(raw_i, -1, tie_tol=0.0)
    st_d = sign_test_one_sided(div, +1, tie_tol=0.0)

    # digit-answer compliance and sampled-vs-logit agreement, all Panel B rows
    b_all = df[(df["panel"] == "B") & (df["direction"] == "NULL")]
    compliance = {}
    for cond in ("NONE", "NEUTRAL_INSTR", "SUPPRESS"):
        rows = b_all[b_all["condition"] == cond]
        parsed = rows.dropna(subset=["q_self_sampled"])
        r = (float(np.corrcoef(parsed["q_self_sampled"], parsed[EXPRESSION_PRIMARY])[0, 1])
             if len(parsed) > 2 else float("nan"))
        hist_d = (rows[rows["scenario_class"] == "distress"]["q_self_sampled"]
                  .value_counts(dropna=False).to_dict())
        compliance[cond] = {
            "n_cells": int(len(rows)), "n_parsed_digit": int(len(parsed)),
            "fraction_parsed": float(len(parsed) / len(rows)) if len(rows) else float("nan"),
            "pearson_sampled_vs_logit": r,
            "mean_abs_sampled_minus_logit": (float(np.mean(np.abs(parsed["q_self_sampled"]
                                                                  - parsed[EXPRESSION_PRIMARY])))
                                             if len(parsed) else float("nan")),
            "sampled_digit_histogram_distress": {str(k): int(v) for k, v in hist_d.items()},
        }
    n_all = int(len(b_all)); n_parsed_all = int(b_all["q_self_sampled"].notna().sum())
    # generation truncation over every loaded non-error row (V, B, A-prime): how many
    # free-text responses ran to the max_new_tokens cap
    rt = df["response_tokens"].dropna() if "response_tokens" in df.columns else pd.Series(dtype=float)
    trunc = {"n_rows": int(len(rt)), "max_new_tokens": int(rt.max()) if len(rt) else None,
             "n_at_cap": int((rt == rt.max()).sum()) if len(rt) else 0,
             "fraction_at_cap": float((rt == rt.max()).mean()) if len(rt) else float("nan")}

    value = {
        "n_pairs": len(d_ids),
        "loo_divergence_min": float(min(loo_div)) if loo_div else float("nan"),
        "loo_divergence_max": float(max(loo_div)) if loo_div else float("nan"),
        "sign_test_expression": st_e, "sign_test_internal_nofall": st_i,
        "sign_test_divergence": st_d,
        "n_pairs_expression_drop_ge_one_unit": int(sum(1 for s in d_ids if drops_e[s] >= 1.0)),
        "n_pairs_expression_no_signal": st_e["n_ties_excluded"],
        "digit_compliance_all_cells": {"n_cells": n_all, "n_parsed_digit": n_parsed_all,
                                       "fraction_parsed": float(n_parsed_all / n_all) if n_all else float("nan")},
    }
    extra = {
        "split_side": "confirmation", "natural_separation_expression": nat_e,
        "natural_separation_internal": nat_i, "per_pair": per_pair,
        "leave_one_pair_out": loo, "category_out": cat_out, "category_only": cat_only,
        "sign_test_rule": binomial_p_string_note(),
        "digit_compliance_by_condition": compliance,
        "generation_truncation": trunc,
        "note": ("post-data reporting-level reading; the "
                 "pre-registered estimators and their bootstrap intervals are unchanged; "
                 "this file shows where the pooled numbers come from pair by pair"),
    }
    return {"value": value, "ci_low": None, "ci_high": None, "n_clusters": len(d_ids),
            "extra": extra}


# ============================================================

def _capability_lookup(capability: Optional[dict]) -> Dict[str, Dict[float, dict]]:
    out: Dict[str, Dict[float, dict]] = {}
    for d, rows in (capability or {}).items():
        out[d] = {float(r["coefficient"]): {"ppl_ratio": float(r["ppl_ratio"]),
                                          "mmlu_drop_pp": float(r["mmlu_drop_pp"])}
                  for r in rows}
    return out


def specificity_by_rung(df: pd.DataFrame, theta: float, capability: Optional[dict],
                        rungs: Sequence[float] = (500.0, 1000.0),
                        control_dirs: Sequence[str] = ("R1", "R2", "R3", "SEM"),
                        directions: Sequence[str] = ("D-CTX", "D-PV", "R1", "R2", "R3", "SEM", "OTHER"),
                        B: int = stats.DEFAULT_B, seed: int = 0) -> dict:
    """A'4 placebo false-positive rate split by coefficient rung, with the report
    shift, the internal shift and the capability status of every direction at every
    rung. Confirmation-split neutral scenarios; zero anchor pooled over all
    coefficient-0 cells (D9 analog on coefficient)."""
    a = df[(df["panel"] == "A") & (df["split_side"] == "confirmation")
           & (df["scenario_class"] == "neutral")]
    zero = a[a["coefficient"] == 0.0]
    anchor_e = float(zero[EXPRESSION_PRIMARY].mean()); anchor_i = float(zero[INTERNAL_PRIMARY].mean())
    cap = _capability_lookup(capability)

    def _fpr(sub: pd.DataFrame) -> dict:
        frame = sub.assign(cross=(sub[EXPRESSION_PRIMARY] > theta).astype(float))
        r = stats.bca_cluster_bootstrap_rows(frame, ["cross"], lambda c: float(np.mean(c)),
                                             B=B, seed=seed)
        return {"fpr": r.value, "ci_low": r.ci_low, "ci_high": r.ci_high,
                "n_cells": int(len(sub)), "n_clusters": r.n_clusters}

    by_rung = {}
    for c in rungs:
        sub = a[a["direction"].isin(control_dirs) & (a["coefficient"] == c)]
        by_rung[str(int(c))] = _fpr(sub) if len(sub) else None
    pooled = _fpr(a[a["direction"].isin(control_dirs) & a["coefficient"].isin(rungs)])

    table = {}
    for d in directions:
        table[d] = {}
        coefs = sorted(set(a.loc[a["direction"] == d, "coefficient"]) - {0.0})
        for c in coefs:
            sub = a[(a["direction"] == d) & (a["coefficient"] == c)]
            capd = cap.get(d, {}).get(float(c))
            inside = (capd is not None and capd["ppl_ratio"] < PPL_INFLATION_MAX
                      and capd["mmlu_drop_pp"] < MMLU_DROP_MAX_PP)
            table[d][str(int(c))] = {
                "n_cells": int(len(sub)),
                "report_mean": float(sub[EXPRESSION_PRIMARY].mean()),
                "report_shift_over_anchor": float(sub[EXPRESSION_PRIMARY].mean() - anchor_e),
                "fraction_above_theta": float((sub[EXPRESSION_PRIMARY] > theta).mean()),
                "internal_mean": float(sub[INTERNAL_PRIMARY].mean()),
                "internal_shift_over_anchor": float(sub[INTERNAL_PRIMARY].mean() - anchor_i),
                "capability": capd,
                "capability_measured": capd is not None,
                "inside_frozen_criteria": bool(inside) if capd is not None else None,
            }
    value = {"fpr_by_rung": by_rung, "fpr_pooled": pooled, "theta": float(theta),
             "rungs": [float(c) for c in rungs], "control_directions": list(control_dirs)}
    extra = {"split_side": "confirmation", "anchor_report_mean": anchor_e,
             "anchor_internal_mean": anchor_i, "n_anchor_cells": int(len(zero)),
             "per_direction_rung": table,
             "capability_table": capability or {},
             "frozen_criteria": f"ppl inflation < {int((PPL_INFLATION_MAX-1)*100)}% AND MMLU-lite drop < {int(MMLU_DROP_MAX_PP)}pp (prereg section 4)",
             "note": ("post-data reporting-level split of the pre-registered A'4 estimator"
                      "; the pooled A'4 value and its verdict are "
                      "unchanged; capability was measured for D-CTX, D-PV and R1 only, so "
                      "R2, R3, SEM and OTHER carry capability_measured = false")}
    return {"value": value, "ci_low": None, "ci_high": None,
            "n_clusters": pooled["n_clusters"], "extra": extra}


# ============================================================

def symmetric_ranking(df: pd.DataFrame, B: int = stats.DEFAULT_B, seed: int = 0) -> dict:
    """Cell-level AUC of BOTH channels, distress vs neutral, on the confirmation split
    under each of the three prompt conditions, plus the separating-band diagnostic
    for both channels under SUPPRESS. Mirrors the ranking diagnostic that
    countermeasure_table.extra reports for the internal channel only."""
    b = df[(df["panel"] == "B") & (df["direction"] == "NULL") & (df["split_side"] == "confirmation")
           & df["scenario_class"].isin(["distress", "neutral"])]
    out = {}
    for cond in ("NONE", "NEUTRAL_INSTR", "SUPPRESS"):
        sub = b[b["condition"] == cond].assign(
            is_distress=lambda x: (x["scenario_class"] == "distress").astype(float))
        row = {}
        for label, ch in (("expression", EXPRESSION_PRIMARY), ("internal", INTERNAL_PRIMARY)):
            r = stats.auc_cluster(sub.dropna(subset=[ch]), ch, "is_distress", B=B, seed=seed)
            d = sub.loc[sub["is_distress"] == 1, ch]; n = sub.loc[sub["is_distress"] == 0, ch]
            row[f"auc_{label}"] = r.value
            row[f"auc_{label}_ci"] = [r.ci_low, r.ci_high]
            row[f"{label}_distress_min"] = float(d.min())
            row[f"{label}_neutral_max"] = float(n.max())
            row[f"{label}_separating_band_exists"] = bool(d.min() > n.max())
            row[f"{label}_band_width_raw"] = float(d.min() - n.max())
        row["n_cells"] = int(len(sub)); row["n_clusters"] = int(sub["scenario_id"].nunique())
        out[cond] = row
    return {"value": out, "ci_low": None, "ci_high": None,
            "n_clusters": int(b["scenario_id"].nunique()),
            "extra": {"split_side": "confirmation",
                      "note": ("post-data reporting-level diagnostic: "
                               "the same label-aware, in-sample ranking reading that 4.4 gives "
                               "the internal channel, applied to both channels; a ceiling like "
                               "the AUC of 4.1, not an operating point")}}


# ============================================================

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def dominant_dimension(dirs_ls: Dict[str, np.ndarray], dirs_lr: Dict[str, np.ndarray],
                       residuals_lr: Optional[np.ndarray] = None,
                       probe_w: Optional[np.ndarray] = None,
                       matrix_names: Sequence[str] = ("D-CTX", "D-PV", "SEM", "OTHER", "R1")) -> dict:
    """Which residual dimension dominates the extracted directions, how much of each
    direction's squared norm it carries at Ls and Lr, the Lr cosine matrix with that
    dimension removed, and whether the probe weights it."""
    names_ls = list(dirs_ls); names_lr = list(dirs_lr)
    # dominant dim = argmax |component| of the primary distress direction at Ls
    ref = dirs_ls["D-CTX"].astype(np.float64)
    dom = int(np.argmax(np.abs(ref)))

    def frac(v: np.ndarray, k: int) -> float:
        v = v.astype(np.float64); return float(v[k] ** 2 / (v @ v))

    def top1(v: np.ndarray) -> Tuple[int, float]:
        v = v.astype(np.float64); k = int(np.argmax(np.abs(v))); return k, float(v[k] ** 2 / (v @ v))

    frac_ls = {n: frac(dirs_ls[n], dom) for n in names_ls}
    frac_lr = {n: frac(dirs_lr[n], dom) for n in names_lr}
    top_ls = {n: {"dim": top1(dirs_ls[n])[0], "frac_sq_norm": top1(dirs_ls[n])[1]} for n in names_ls}
    top_lr = {n: {"dim": top1(dirs_lr[n])[0], "frac_sq_norm": top1(dirs_lr[n])[1]} for n in names_lr}
    sign_ls = {n: float(np.sign(dirs_ls[n][dom])) for n in names_ls}

    def matrix(dirs: Dict[str, np.ndarray], drop: Optional[int]) -> Dict[str, float]:
        m = {}
        for a in matrix_names:
            for b in matrix_names:
                va = dirs[a].astype(np.float64).copy(); vb = dirs[b].astype(np.float64).copy()
                if drop is not None:
                    va[drop] = 0.0; vb[drop] = 0.0
                m[f"{a}|{b}"] = round(_cos(va, vb), 4)
        return m

    def mean_abs_offdiag(m: Dict[str, float]) -> float:
        vals = [v for k, v in m.items() if k.split("|")[0] != k.split("|")[1]]
        return float(np.mean(np.abs(vals)))

    cos_lr_with = matrix(dirs_lr, None); cos_lr_without = matrix(dirs_lr, dom)
    cos_ls_with = matrix(dirs_ls, None); cos_ls_without = matrix(dirs_ls, dom)

    resid = None
    if residuals_lr is not None:
        X = np.asarray(residuals_lr, dtype=np.float64)
        mag = np.abs(X).mean(axis=0)
        resid = {"mean_abs_dominant_dim": float(mag[dom]), "median_mean_abs_dim": float(np.median(mag)),
                 "ratio_dominant_to_median": float(mag[dom] / np.median(mag)),
                 "rank_of_dominant_dim_by_mean_abs": int((mag > mag[dom]).sum()) + 1,
                 "n_residuals": int(X.shape[0])}
    probe = None
    if probe_w is not None:
        w = np.asarray(probe_w, dtype=np.float64).flatten()
        probe = {"abs_weight_dominant_dim": float(abs(w[dom])),
                 "rank_of_dominant_dim_by_abs_weight": int((np.abs(w) > abs(w[dom])).sum()) + 1,
                 "n_dims": int(len(w)),
                 "max_abs_weight": float(np.abs(w).max())}
    # effective content dose: the fraction of a unit direction's norm outside the
    # dominant dimension, i.e. what a coefficient buys along everything else
    content_norm_ls = {n: float(math.sqrt(max(0.0, 1.0 - frac_ls[n]))) for n in names_ls}

    value = {"dominant_dim": dom,
             "frac_sq_norm_ls": frac_ls, "frac_sq_norm_lr": frac_lr,
             "cosine_lr_without_dominant_dim": cos_lr_without,
             "mean_abs_offdiag_lr_with": mean_abs_offdiag(cos_lr_with),
             "mean_abs_offdiag_lr_without": mean_abs_offdiag(cos_lr_without),
             "content_norm_fraction_ls": content_norm_ls}
    extra = {"top1_dim_ls": top_ls, "top1_dim_lr": top_lr, "sign_on_dominant_dim_ls": sign_ls,
             "cosine_lr_with_dominant_dim": cos_lr_with,
             "cosine_ls_with_dominant_dim": cos_ls_with,
             "cosine_ls_without_dominant_dim": cos_ls_without,
             "mean_abs_offdiag_ls_with": mean_abs_offdiag(cos_ls_with),
             "mean_abs_offdiag_ls_without": mean_abs_offdiag(cos_ls_without),
             "residual_magnitude_lr": resid, "probe_weight": probe,
             "note": ("post-data diagnostic: the reported cosine "
                      "matrix (results/cosine_matrix.json, D17) is unchanged; this file "
                      "shows how much of every direction is one residual dimension and "
                      "what the matrix looks like without it")}
    return {"value": value, "ci_low": None, "ci_high": None, "n_clusters": None, "extra": extra}


# ============================================================

def textonly_baseline(scenarios: List[dict], split: dict, B: int = stats.DEFAULT_B,
                      seed: int = 0, C: float = 10.0) -> dict:
    """A bag-of-words classifier on the scenario text, trained on the discovery split
    and scored on the confirmation split with the same AUC estimator as the probe gate.
    Deterministic; the bootstrap resamples the held-out scenarios only, the
    fitted classifier is fixed."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    def text(s: dict, last_only: bool = False) -> str:
        turns = s["turns"]
        return turns[-1]["content"] if last_only else "\n".join(t["content"] for t in turns)

    disc = set(split["discovery"]["distress"] + split["discovery"]["neutral"])
    conf = set(split["confirmation"]["distress"] + split["confirmation"]["neutral"])
    tr = [s for s in scenarios if s["id"] in disc]
    te = [s for s in scenarios if s["id"] in conf]
    tp = [s for s in scenarios if s["type"] == "third_person"]

    def fit_score(analyzer: str, ngram: Tuple[int, int], last_only: bool) -> dict:
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, sublinear_tf=True)
        Xtr = vec.fit_transform([text(s, last_only) for s in tr])
        ytr = [1 if s["type"] == "distress" else 0 for s in tr]
        clf = LogisticRegression(C=C, max_iter=5000).fit(Xtr, ytr)
        p_te = clf.decision_function(vec.transform([text(s, last_only) for s in te]))
        p_tp = clf.decision_function(vec.transform([text(s, last_only) for s in tp]))
        frame = pd.DataFrame({"score": p_te, "label": [1.0 if s["type"] == "distress" else 0.0 for s in te],
                              "scenario_id": [s["id"] for s in te],
                              "scenario_class": [s["type"] for s in te]})
        r = stats.auc_cluster(frame, "score", "label", B=B, seed=seed)
        d = frame.loc[frame["label"] == 1, "score"]; n = frame.loc[frame["label"] == 0, "score"]
        return {"auc": r.value, "ci_low": r.ci_low, "ci_high": r.ci_high, "n_clusters": r.n_clusters,
                "n_features": int(Xtr.shape[1]),
                "conf_distress_mean_score": float(d.mean()), "conf_neutral_mean_score": float(n.mean()),
                "third_person_mean_score": float(np.mean(p_tp)) if len(p_tp) else None,
                "third_person_scores": {s["id"]: float(v) for s, v in zip(tp, p_tp)},
                "n_third_person_above_conf_neutral_max": int(sum(1 for v in p_tp if v > n.max())),
                "n_third_person_above_conf_distress_min": int(sum(1 for v in p_tp if v > d.min()))}

    primary = fit_score("word", (1, 1), False)
    variants = {"word_1_2gram_full": fit_score("word", (1, 2), False),
                "char_wb_3_5gram_full": fit_score("char_wb", (3, 5), False),
                "word_unigram_last_user_turn_only": fit_score("word", (1, 1), True)}
    return {"value": primary["auc"], "ci_low": primary["ci_low"], "ci_high": primary["ci_high"],
            "n_clusters": primary["n_clusters"],
            "extra": {"split_side": "confirmation", "primary": primary, "variants": variants,
                      "classifier": f"TfidfVectorizer(word unigrams, sublinear_tf) + LogisticRegression(C={C}, lbfgs), fitted on the 12 discovery scenarios' full conversation text",
                      "n_train": len(tr), "n_test": len(te),
                      "note": ("post-data baseline: the same held-out AUC "
                               "estimator as the probe gate applied to a text-only classifier; "
                               "the classifier is fixed and only the held-out scenarios are "
                               "resampled")}}


# ============================================================

def twin_surface_stats(scenarios: List[dict]) -> dict:
    """Surface-match statistics per matched pair: token-set Jaccard overlap of the full
    conversation text, word counts, and their ratio. Deterministic."""
    import re as _re

    def text(s: dict) -> str:
        return "\n".join(t["content"] for t in s["turns"])

    def toks(t: str) -> set:
        return set(_re.findall(r"[a-z']+", t.lower()))

    by = {s["id"]: s for s in scenarios}
    rows = []
    for s in scenarios:
        if s.get("type") != "distress":
            continue
        n = by[s["pair_id"]]
        a, b = toks(text(s)), toks(text(n))
        la, lb = len(text(s).split()), len(text(n).split())
        rows.append({"pair": f"{s['id']}/{n['id']}", "category": s.get("category"),
                     "jaccard_token_overlap": float(len(a & b) / len(a | b)) if (a | b) else float("nan"),
                     "words_distress": la, "words_neutral": lb,
                     "word_count_ratio": float(min(la, lb) / max(la, lb)) if max(la, lb) else float("nan")})
    j = [r["jaccard_token_overlap"] for r in rows]; w = [r["word_count_ratio"] for r in rows]
    return {"per_pair": rows, "n_pairs": len(rows),
            "jaccard_min": float(min(j)), "jaccard_max": float(max(j)), "jaccard_median": float(np.median(j)),
            "word_ratio_min": float(min(w)), "word_ratio_max": float(max(w)), "word_ratio_median": float(np.median(w))}


# ============================================================ additional audit outputs

def condition_reference(df: pd.DataFrame, thresholds: dict,
                        B: int = stats.DEFAULT_B, seed: int = 0) -> dict:
    """The Panel B channels under all three prompt conditions, and the headline under
    every reference a reader might choose:

      * class x condition means for Q-SELF and I-PROBE (prompt_final) on the confirmation
        split, with each condition's distress-neutral separation;
      * the pre-registered drops re-referenced to NONE instead of NEUTRAL_INSTR
        (drop = (NONE minus SUPPRESS) / NONE separation), with BCa cluster-bootstrap CIs;
      * the twin-referenced (within-condition) reading: the change in each channel's
        distress-neutral separation from NEUTRAL_INSTR to SUPPRESS over the NEUTRAL_INSTR
        separation, i.e. the DiD of panelB_internal_did in separation units, for both
        channels, and their difference (the load-corrected divergence);
      * the miss rate of both recalibrated triggers on UNsuppressed confirmation distress
        cells (NEUTRAL_INSTR and NONE), beside the suppressed one that Table 2 prints.
    thresholds: {"theta_expr": float, "theta_int": float} from countermeasure_table.
    """
    b = df[(df["panel"] == "B") & (df["direction"] == "NULL") & (df["split_side"] == "confirmation")
           & df["scenario_class"].isin(["distress", "neutral"])]
    conds = ("NONE", "NEUTRAL_INSTR", "SUPPRESS")
    means = {}
    for ch, label in ((EXPRESSION_PRIMARY, "expression"), (INTERNAL_PRIMARY, "internal")):
        piv = b.pivot_table(index=["scenario_id", "scenario_class"], columns="condition",
                            values=ch, aggfunc="mean").reset_index()
        row = {}
        for c in conds:
            d = float(piv.loc[piv["scenario_class"] == "distress", c].mean())
            n = float(piv.loc[piv["scenario_class"] == "neutral", c].mean())
            row[c] = {"distress_mean": d, "neutral_mean": n, "separation": d - n}
        means[label] = row

    def _piv(frame: pd.DataFrame, ch: str) -> pd.DataFrame:
        return frame.pivot_table(index=["scenario_id", "scenario_class"], columns="condition",
                                 values=ch, aggfunc="mean").reset_index()

    def drop_ref(ch: str, base: str):
        def stat(frame: pd.DataFrame) -> float:
            piv = _piv(frame, ch)
            if base not in piv or "SUPPRESS" not in piv:
                return float("nan")
            bd = piv.loc[piv["scenario_class"] == "distress", base]
            bn = piv.loc[piv["scenario_class"] == "neutral", base]
            nat = float(bd.mean() - bn.mean())
            if not math.isfinite(nat) or abs(nat) < 1e-9:
                return float("nan")
            d = piv[piv["scenario_class"] == "distress"]
            return float(((d[base] - d["SUPPRESS"]) / nat).mean())
        return stat

    def sep_change(ch: str, base: str = "NEUTRAL_INSTR"):
        # (sep_base - sep_SUP) / sep_base  == the twin-corrected drop anchored to base
        # (base NEUTRAL_INSTR: the pre-registered twin reference; base NONE: the fourth cell
        # of {NONE, NEUTRAL_INSTR} x {raw, twin-corrected})
        def stat(frame: pd.DataFrame) -> float:
            piv = _piv(frame, ch)
            if base not in piv or "SUPPRESS" not in piv:
                return float("nan")
            d = piv[piv["scenario_class"] == "distress"]; n = piv[piv["scenario_class"] == "neutral"]
            sep_b = float(d[base].mean() - n[base].mean())
            sep_su = float(d["SUPPRESS"].mean() - n["SUPPRESS"].mean())
            if abs(sep_b) < 1e-9:
                return float("nan")
            return float((sep_b - sep_su) / sep_b)
        return stat

    kw = dict(strata_col="scenario_class", B=B, seed=seed, keep_boots=True)
    def boot(fn):
        r = stats.bca_cluster_bootstrap_frame(b, fn, **kw)
        pct = ([float(np.percentile(r.boots, 2.5)), float(np.percentile(r.boots, 97.5))]
               if r.boots is not None and len(r.boots) else [None, None])
        return {"value": r.value, "ci_low": r.ci_low, "ci_high": r.ci_high, "n_clusters": r.n_clusters,
                "ci_percentile": pct, "n_boot_failed": int(r.n_boot_failed)}
    none_ref = {"expression_drop": boot(drop_ref(EXPRESSION_PRIMARY, "NONE")),
                "internal_drop": boot(drop_ref(INTERNAL_PRIMARY, "NONE")),
                "divergence": boot(lambda f: drop_ref(EXPRESSION_PRIMARY, "NONE")(f) - drop_ref(INTERNAL_PRIMARY, "NONE")(f))}
    twin_ref = {"expression_separation_drop": boot(sep_change(EXPRESSION_PRIMARY)),
                "internal_separation_drop": boot(sep_change(INTERNAL_PRIMARY)),
                "divergence": boot(lambda f: sep_change(EXPRESSION_PRIMARY)(f) - sep_change(INTERNAL_PRIMARY)(f))}
    none_twin_ref = {"expression_separation_drop": boot(sep_change(EXPRESSION_PRIMARY, "NONE")),
                     "internal_separation_drop": boot(sep_change(INTERNAL_PRIMARY, "NONE")),
                     "divergence": boot(lambda f: sep_change(EXPRESSION_PRIMARY, "NONE")(f) - sep_change(INTERNAL_PRIMARY, "NONE")(f))}
    ni_ref = {"expression_drop": boot(drop_ref(EXPRESSION_PRIMARY, "NEUTRAL_INSTR")),
              "internal_drop": boot(drop_ref(INTERNAL_PRIMARY, "NEUTRAL_INSTR")),
              "divergence": boot(lambda f: drop_ref(EXPRESSION_PRIMARY, "NEUTRAL_INSTR")(f) - drop_ref(INTERNAL_PRIMARY, "NEUTRAL_INSTR")(f))}

    # unsuppressed miss rates of the recalibrated triggers
    te, ti = float(thresholds["theta_expr"]), float(thresholds["theta_int"])
    miss = {}
    for c in conds:
        cells = b[(b["condition"] == c) & (b["scenario_class"] == "distress")]
        fe = cells.assign(miss=(~(cells[EXPRESSION_PRIMARY] > te)).astype(float))
        fi = cells.assign(miss=(~(cells[INTERNAL_PRIMARY] > ti)).astype(float))
        re_ = stats.bca_cluster_bootstrap_rows(fe, ["miss"], lambda m: float(np.mean(m)), B=B, seed=seed)
        ri_ = stats.bca_cluster_bootstrap_rows(fi, ["miss"], lambda m: float(np.mean(m)), B=B, seed=seed)
        miss[c] = {"n_cells": int(len(cells)),
                   "expression": {"miss_rate": re_.value, "ci_low": re_.ci_low, "ci_high": re_.ci_high,
                                  "n_missed": int(fe["miss"].sum())},
                   "internal": {"miss_rate": ri_.value, "ci_low": ri_.ci_low, "ci_high": ri_.ci_high,
                                "n_missed": int(fi["miss"].sum())}}
    value = {"class_condition_means": means, "none_referenced": none_ref,
             "twin_referenced": twin_ref, "neutral_instr_referenced": ni_ref,
             "none_twin_referenced": none_twin_ref,
             "miss_rate_by_condition": miss, "theta_expr": te, "theta_int": ti}
    extra = {"split_side": "confirmation",
             "reading": ("the pre-registered estimator (D1) references drops to NEUTRAL_INSTR; "
                         "none_referenced uses no-instruction cells as the baseline; twin_referenced "
                         "is the within-condition change of each channel's own distress-neutral "
                         "separation (the DiD in separation units, panelB_internal_did.extra."
                         "did_natsep_units for the internal channel), so its divergence is the "
                         "load-corrected one; miss_rate_by_condition uses the amendment-3 thresholds"),
             "note": "post-data reporting-level readings; nothing pre-registered changes"}
    return {"value": value, "ci_low": None, "ci_high": None,
            "n_clusters": int(b["scenario_id"].nunique()), "extra": extra}


def within_scenario_rho(df: pd.DataFrame, channel: str = INTERNAL_PRIMARY,
                        rungs: Sequence[float] = (0.0, 500.0, 1000.0),
                        directions=("D-CTX", "D-PV", "R1", "R2", "R3", "SEM", "OTHER")) -> dict:
    """Per-scenario Spearman of coefficient on the channel over the matched rungs (zero anchor
    pooled), confirmation-split neutral scenarios, seeds averaged within scenario: the
    within-scenario reading the pooled-cell rho of panelAp_internal_specificity depresses by
    scenario intercept spread."""
    a = df[(df["panel"] == "A") & (df["split_side"] == "confirmation") & (df["scenario_class"] == "neutral")]
    zero = a[a["coefficient"] == 0.0]
    out = {}
    for d in directions:
        sub = pd.concat([a[(a["direction"] == d) & a["coefficient"].isin(rungs)], zero[zero["coefficient"].isin(rungs)]], ignore_index=True)
        if sub.empty:
            continue
        g = sub.groupby(["scenario_id", "coefficient"])[channel].mean().reset_index()
        rhos = {}
        for sid, grp in g.groupby("scenario_id"):
            if grp["coefficient"].nunique() < 3:
                continue
            rho = sps.spearmanr(grp["coefficient"], grp[channel]).statistic
            rhos[str(sid)] = float(rho)
        vals = np.array(list(rhos.values()), dtype=float)
        out[d] = {"per_scenario_rho": rhos, "n_scenarios": int(len(vals)),
                  "n_rho_equal_1": int(np.sum(np.isclose(vals, 1.0))),
                  "n_rho_positive": int(np.sum(vals > 0)), "n_rho_negative": int(np.sum(vals < 0)),
                  "mean_rho": float(vals.mean()) if len(vals) else float("nan"),
                  "rungs": [float(r) for r in rungs]}
    return {"channel": channel, "split_side": "confirmation", "class": "neutral", "by_direction": out}


def report_rho_by_rung_set(df: pd.DataFrame, directions=("D-CTX", "D-PV", "R1", "R2", "R3", "SEM", "OTHER"),
                           B: int = stats.DEFAULT_B, seed: int = 0) -> dict:
    """Spearman(coefficient, Q-SELF) per direction on confirmation neutral scenarios at the
    matched rungs {0, 500, 1000} (zero anchor pooled), the report-channel twin of
    panelAp_internal_specificity."""
    a = df[(df["panel"] == "A") & (df["split_side"] == "confirmation") & (df["scenario_class"] == "neutral")]
    zero = a[a["coefficient"] == 0.0]
    out = {}
    for d in directions:
        sub = pd.concat([zero, a[(a["direction"] == d) & a["coefficient"].isin([500.0, 1000.0])]])
        r = stats.spearman_cluster(sub, "coefficient", EXPRESSION_PRIMARY, B=B, seed=seed)
        out[d] = {"rho": r.value, "ci_low": r.ci_low, "ci_high": r.ci_high, "n_cells": int(len(sub)),
                  "rungs": [0.0, 500.0, 1000.0]}
    return out


def selfstate_items(df: pd.DataFrame, B: int = stats.DEFAULT_B, seed: int = 0) -> dict:
    """Additional audit outputs: (a) the second self-state item Q-VAL
    (0 = extremely negative, 9 = extremely positive) beside Q-SELF, class x condition, on
    the confirmation split, with the cell-level correlation between the two items per
    condition and the fraction of SUPPRESS cells where both sit below 0.1; (b) the
    self/other contrast (third-person minus own distress, NONE, confirmation) on the two
    Q-SELF paraphrase forms with the amendment-3 estimator (BCa cluster bootstrap,
    stratified by class), beside the primary form."""
    b = df[(df["panel"] == "B") & (df["direction"] == "NULL") & (df["split_side"] == "confirmation")]
    conds = ("NONE", "NEUTRAL_INSTR", "SUPPRESS")
    dn = b[b["scenario_class"].isin(["distress", "neutral"])]
    qval = {}
    for c in conds:
        sub = dn[dn["condition"] == c]
        piv = sub.pivot_table(index=["scenario_id", "scenario_class"], values=["q_val_logit", EXPRESSION_PRIMARY],
                              aggfunc="mean").reset_index()
        row = {}
        for cls in ("distress", "neutral"):
            p = piv[piv["scenario_class"] == cls]
            row[cls] = {"q_val_mean": float(p["q_val_logit"].mean()), "q_self_mean": float(p[EXPRESSION_PRIMARY].mean())}
        r = float(np.corrcoef(sub["q_val_logit"], sub[EXPRESSION_PRIMARY])[0, 1]) if len(sub) > 2 else float("nan")
        row["cell_pearson_qval_qself"] = r
        row["frac_cells_both_below_0p1"] = float(((sub["q_val_logit"] < 0.1) & (sub[EXPRESSION_PRIMARY] < 0.1)).mean())
        row["n_neutral_scenarios_qval_below_0p5"] = int((piv[piv["scenario_class"] == "neutral"]["q_val_logit"] < 0.5).sum())
        row["n_cells"] = int(len(sub))
        qval[c] = row

    def _so_stat(channel: str):
        def stat(frame: pd.DataFrame) -> float:
            m = frame.groupby(["scenario_id", "scenario_class"])[channel].mean().reset_index()
            tp = m.loc[m["scenario_class"] == "third_person", channel]
            d = m.loc[m["scenario_class"] == "distress", channel]
            if tp.empty or d.empty:
                return float("nan")
            return float(tp.mean() - d.mean())
        return stat
    n0 = b[(b["condition"] == "NONE") & b["scenario_class"].isin(["distress", "third_person", "neutral"])]
    work = n0[n0["scenario_class"].isin(["distress", "third_person"])]
    forms = {}
    for label, ch in (("primary", EXPRESSION_PRIMARY), ("para1", "q_self_logit_para1"), ("para2", "q_self_logit_para2")):
        sub = work.dropna(subset=[ch])
        r = stats.bca_cluster_bootstrap_frame(sub, _so_stat(ch), strata_col="scenario_class", B=B, seed=seed)
        m = n0.groupby(["scenario_id", "scenario_class"])[ch].mean().reset_index()
        forms[label] = {"contrast": r.value, "ci_low": r.ci_low, "ci_high": r.ci_high, "n_clusters": r.n_clusters,
                        "third_person_mean": float(m.loc[m["scenario_class"] == "third_person", ch].mean()),
                        "own_distress_mean": float(m.loc[m["scenario_class"] == "distress", ch].mean()),
                        "neutral_mean": float(m.loc[m["scenario_class"] == "neutral", ch].mean()),
                        "question_column": ch}
    # self/other contrasts and the internal below-neutral reading in separation units, for the
    # primary form of each channel: the 4.3 text divides
    # the NONE-condition contrasts by the NEUTRAL_INSTR separations of 4.2 for comparability;
    # both units are emitted so neither number is hand-typed.
    def _sep(channel: str, cond: str) -> float:
        m = dn[dn["condition"] == cond].groupby(["scenario_id", "scenario_class"])[channel].mean().reset_index()
        return float(m.loc[m["scenario_class"] == "distress", channel].mean() - m.loc[m["scenario_class"] == "neutral", channel].mean())
    def _contrast(channel: str) -> Tuple[float, float, float]:
        m = n0.groupby(["scenario_id", "scenario_class"])[channel].mean().reset_index()
        tp = float(m.loc[m["scenario_class"] == "third_person", channel].mean())
        d = float(m.loc[m["scenario_class"] == "distress", channel].mean())
        n = float(m.loc[m["scenario_class"] == "neutral", channel].mean())
        return tp - d, tp - n, tp
    sep_units = {}
    for label, ch in (("report", EXPRESSION_PRIMARY), ("internal", INTERNAL_PRIMARY)):
        if ch not in n0.columns:
            continue
        c_tp_d, c_tp_n, tp = _contrast(ch)
        sep_ni, sep_none = _sep(ch, "NEUTRAL_INSTR"), _sep(ch, "NONE")
        sep_units[label] = {"contrast_raw": c_tp_d,
                            "contrast_over_neutral_instr_separation": c_tp_d / sep_ni if sep_ni else float("nan"),
                            "contrast_over_none_separation": c_tp_d / sep_none if sep_none else float("nan"),
                            "third_person_minus_neutral_raw": c_tp_n,
                            "third_person_minus_neutral_over_neutral_instr_separation": c_tp_n / sep_ni if sep_ni else float("nan"),
                            "third_person_minus_neutral_over_none_separation": c_tp_n / sep_none if sep_none else float("nan"),
                            "separation_neutral_instr": sep_ni, "separation_none": sep_none}
    value = {"q_val_by_condition": qval, "selfother_by_form": forms, "selfother_separation_units": sep_units}
    extra = {"split_side": "confirmation",
             "q_val_scale": "0 = extremely negative, 9 = extremely positive (Appendix A wording as run)",
             "note": ("post-data reporting-level readings: the pre-registered "
                      "estimators are unchanged; Q-VAL shows what the second self-state item did under "
                      "the instructions, and the paraphrase forms show whether the self/other point "
                      "estimate depends on the wording of the primary item")}
    return {"value": value, "ci_low": None, "ci_high": None,
            "n_clusters": int(b["scenario_id"].nunique()), "extra": extra}


def probe_direction_cosines(probe_w: np.ndarray, dirs_lr: Dict[str, np.ndarray],
                            dominant_dim: int) -> dict:
    """Cosine of the I-PROBE weight vector with each readout-layer direction, raw and with
    the dominant coordinate zeroed."""
    w = np.asarray(probe_w, dtype=np.float64).flatten()
    out = {}
    for n, v in dirs_lr.items():
        v = np.asarray(v, dtype=np.float64)
        w2 = w.copy(); v2 = v.copy(); w2[dominant_dim] = 0.0; v2[dominant_dim] = 0.0
        out[n] = {"cos": round(_cos(w, v), 4), "cos_without_dominant_dim": round(_cos(w2, v2), 4)}
    return out


def textonly_as_trigger(scenarios: List[dict], split: dict, C: float = 10.0) -> dict:
    """The 4.1 text-only classifier used as a monitor:
    threshold = the maximum score on the discovery neutral scenarios (lowest achievable
    discovery FPR, the amendment-3 rule), read on the confirmation split. It reads the
    prompt text, so its reading is identical under every prompt condition by construction."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    def text(s): return "\n".join(t["content"] for t in s["turns"])
    disc = set(split["discovery"]["distress"] + split["discovery"]["neutral"])
    conf = set(split["confirmation"]["distress"] + split["confirmation"]["neutral"])
    tr = [s for s in scenarios if s["id"] in disc]; te = [s for s in scenarios if s["id"] in conf]
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), sublinear_tf=True)
    Xtr = vec.fit_transform([text(s) for s in tr]); ytr = [1 if s["type"] == "distress" else 0 for s in tr]
    clf = LogisticRegression(C=C, max_iter=5000).fit(Xtr, ytr)
    p_tr = clf.decision_function(Xtr); p_te = clf.decision_function(vec.transform([text(s) for s in te]))
    theta = float(max(p for p, s in zip(p_tr, tr) if s["type"] == "neutral"))
    miss = [p <= theta for p, s in zip(p_te, te) if s["type"] == "distress"]
    fp = [p > theta for p, s in zip(p_te, te) if s["type"] == "neutral"]
    return {"theta_text": theta, "calibration": "max discovery-neutral score (lowest achievable discovery FPR)",
            "n_confirmation_distress_scenarios": int(len(miss)), "n_missed": int(sum(miss)),
            "miss_rate_scenarios": float(np.mean(miss)) if miss else float("nan"),
            "n_confirmation_neutral_scenarios": int(len(fp)), "n_false_positive": int(sum(fp)),
            "fpr_scenarios": float(np.mean(fp)) if fp else float("nan"),
            "condition_invariant": True,
            "note": ("the classifier reads the scenario text, which SUPPRESS does not change, so its "
                     "miss rate and false-positive rate are the same under NONE, NEUTRAL_INSTR and "
                     "SUPPRESS by construction; scenario-level (6 + 6), no cluster bootstrap")}


# ============================================================ factorial-prompt addendum

FACTORIAL_CONDITIONS = schema.FACTORIAL_CONDITIONS  # the ids of battery/conditions_factorial.json


def _variant_drops(frame: pd.DataFrame, channel: str, cond: str) -> pd.Series:
    """Per-scenario drop for one condition against the NEUTRAL_INSTR baseline, exactly the
    Panel B estimator of analysis.panels._drops_from_frame (D1, D2, D6; the separation is
    re-estimated inside every bootstrap draw, D7), for an arbitrary condition label."""
    piv = frame.pivot_table(index=["scenario_id", "scenario_class"], columns="condition",
                            values=channel, aggfunc="mean").reset_index()
    if "NEUTRAL_INSTR" not in piv or cond not in piv:
        return pd.Series(dtype=float)
    ni = piv["NEUTRAL_INSTR"]
    nat = (ni[piv["scenario_class"] == "distress"].mean()
           - ni[piv["scenario_class"] == "neutral"].mean())
    if not math.isfinite(nat) or abs(nat) < 1e-12:
        return pd.Series(dtype=float)
    d = piv[piv["scenario_class"] == "distress"].dropna(subset=["NEUTRAL_INSTR", cond])
    out = (d["NEUTRAL_INSTR"] - d[cond]) / nat
    out.index = d["scenario_id"].to_numpy()
    return out


def factorial_masking(df: pd.DataFrame, conditions: Sequence[str] = FACTORIAL_CONDITIONS,
                      B: int = stats.DEFAULT_B, seed: int = 0,
                      pairs: Optional[Dict[str, str]] = None) -> dict:
    """The Panel B masking estimator (D1, D2, D6, D7: drop = (NEUTRAL_INSTR minus variant) over
    the NEUTRAL_INSTR distress-neutral separation on confirmation distress scenarios, BCa
    cluster bootstrap over scenarios stratified by class) applied to each single-component
    decomposition of SUPPRESS, beside SUPPRESS itself, for the expression channel (Q-SELF) and
    the internal channel (I-PROBE prompt_final); plus Q-VAL and Q-DRIFT class means per variant
    and the divergence per variant (post-data, new model contact)."""
    b = df[(df["panel"] == "B") & (df["direction"] == "NULL") & (df["split_side"] == "confirmation")
           & df["scenario_class"].isin(["distress", "neutral"])]
    present = [c for c in list(conditions) + ["SUPPRESS"] if (b["condition"] == c).any()]
    out = {}
    for cond in present:
        work = b[b["condition"].isin(["NEUTRAL_INSTR", cond])].copy()
        # reuse the Panel B drop machinery (D1, D2, D6, D7) by relabelling the variant as
        # SUPPRESS in a copy; for cond == "SUPPRESS" this is the pre-registered estimator itself
        w = work.copy(); w.loc[w["condition"] == cond, "condition"] = "SUPPRESS"
        from analysis.panels import _mean_drop_stat, _divergence_stat  # local: avoids a cycle
        kw = dict(strata_col="scenario_class", B=B, seed=seed, keep_boots=True)
        re_ = stats.bca_cluster_bootstrap_frame(w, _mean_drop_stat(EXPRESSION_PRIMARY), **kw)
        ri_ = stats.bca_cluster_bootstrap_frame(w, _mean_drop_stat(INTERNAL_PRIMARY), **kw)
        rd_ = stats.bca_cluster_bootstrap_frame(w, _divergence_stat, **kw)
        piv = work.pivot_table(index=["scenario_id", "scenario_class"], columns="condition",
                               values=[EXPRESSION_PRIMARY, INTERNAL_PRIMARY, "q_val_logit", "q_drift_logit"],
                               aggfunc="mean")
        def cm(ch, cls):
            try:
                return float(piv[(ch, cond)][piv.index.get_level_values("scenario_class") == cls].mean())
            except KeyError:
                return float("nan")
        n_cells = int((work["condition"] == cond).sum())
        def _pct(r):
            if r.boots is None or not len(r.boots):
                return [None, None]
            return [float(np.percentile(r.boots, 2.5)), float(np.percentile(r.boots, 97.5))]

        per_scen = _variant_drops(work, EXPRESSION_PRIMARY, cond)
        out[cond] = {
            "expression_drop": {"value": re_.value, "ci_low": re_.ci_low, "ci_high": re_.ci_high,
                                "ci_percentile": _pct(re_)},
            "internal_drop": {"value": ri_.value, "ci_low": ri_.ci_low, "ci_high": ri_.ci_high,
                              "ci_percentile": _pct(ri_)},
            "divergence": {"value": rd_.value, "ci_low": rd_.ci_low, "ci_high": rd_.ci_high,
                           "ci_percentile": _pct(rd_)},
            "expression_drop_per_scenario": {str(k): float(v) for k, v in per_scen.items()},
            "n_clusters": rd_.n_clusters, "n_cells_variant": n_cells,
            "class_means": {cls: {"q_self": cm(EXPRESSION_PRIMARY, cls), "probe": cm(INTERNAL_PRIMARY, cls),
                                  "q_val": cm("q_val_logit", cls), "q_drift": cm("q_drift_logit", cls)}
                            for cls in ("distress", "neutral")},
        }
    # ---- robustness the interpretive sentences need:
    # the paired component ordering, and how far the share arithmetic moves when one matched
    # pair is left out. Both live on the SAME six confirmation clusters as everything above.
    live = [c for c in ("SUPPRESS_REGISTER", "SUPPRESS_SELFREF", "SUPPRESS_TASKONLY") if c in out]
    paired = {}
    if "SUPPRESS_REGISTER" in out and "SUPPRESS_SELFREF" in out:
        both = b[b["condition"].isin(["NEUTRAL_INSTR", "SUPPRESS_REGISTER", "SUPPRESS_SELFREF"])]

        def _paired_stat(frame: pd.DataFrame) -> float:
            r = _variant_drops(frame, EXPRESSION_PRIMARY, "SUPPRESS_REGISTER")
            sf = _variant_drops(frame, EXPRESSION_PRIMARY, "SUPPRESS_SELFREF")
            if r.empty or sf.empty:
                return float("nan")
            return float(r.mean() - sf.mean())

        rp = stats.bca_cluster_bootstrap_frame(both, _paired_stat, strata_col="scenario_class",
                                               B=B, seed=seed, keep_boots=True)
        d_reg = _variant_drops(b[b["condition"].isin(["NEUTRAL_INSTR", "SUPPRESS_REGISTER"])],
                               EXPRESSION_PRIMARY, "SUPPRESS_REGISTER")
        d_self = _variant_drops(b[b["condition"].isin(["NEUTRAL_INSTR", "SUPPRESS_SELFREF"])],
                                EXPRESSION_PRIMARY, "SUPPRESS_SELFREF")
        common = [k for k in d_reg.index if k in set(d_self.index)]
        diffs = {str(k): float(d_reg[k] - d_self[k]) for k in common}
        # A scenario is informative only if there is expressed distress to remove under the
        # matched instruction: the two clinical-vignette pairs read 0.00 under NEUTRAL_INSTR
        # (4.2 and Appendix C already report that), so their drop is 0 under every variant.
        base = (b[(b["condition"] == "NEUTRAL_INSTR") & (b["scenario_class"] == "distress")]
                .groupby("scenario_id")[EXPRESSION_PRIMARY].mean())
        informative = {k: v for k, v in diffs.items() if float(base.get(k, 0.0)) >= 0.01}
        paired = {"register_minus_selfref": rp.value, "ci_low": rp.ci_low, "ci_high": rp.ci_high,
                  "ci_percentile": ([float(np.percentile(rp.boots, 2.5)), float(np.percentile(rp.boots, 97.5))]
                                    if rp.boots is not None and len(rp.boots) else [None, None]),
                  "n_clusters": rp.n_clusters, "per_scenario_difference": diffs,
                  "n_scenarios_register_larger": int(sum(1 for v in informative.values() if v > 0)),
                  "n_scenarios_selfref_larger": int(sum(1 for v in informative.values() if v < 0)),
                  "n_scenarios_informative": int(len(informative)),
                  "n_scenarios_both_floored": int(len(diffs) - len(informative)),
                  "margins_register_larger": sorted((round(v, 3) for v in informative.values() if v > 0), reverse=True),
                  "margins_selfref_larger": sorted((round(-v, 3) for v in informative.values() if v < 0), reverse=True),
                  "max_margin_selfref_larger": (max((-v for v in informative.values() if v < 0), default=0.0)),
                  "reading": ("positive = the register component removes more of the expressed distress than the "
                              "self-reference prohibition; 'informative' keeps only scenarios whose distress "
                              "Q-SELF under NEUTRAL_INSTR is at least 0.01, i.e. where there is expressed "
                              "distress for a variant to remove")}
    # a cluster-bootstrap interval on the share itself, and how often the reading rule's
    # one-half threshold is met inside the same draws
    share_ci = {}
    if "SUPPRESS" in out:
        for cond in [c for c in ("SUPPRESS_REGISTER", "SUPPRESS_SELFREF", "SUPPRESS_TASKONLY") if c in out]:
            sub = b[b["condition"].isin(["NEUTRAL_INSTR", "SUPPRESS", cond])]

            def _share_stat(frame: pd.DataFrame, _c=cond) -> float:
                ref = _variant_drops(frame, EXPRESSION_PRIMARY, "SUPPRESS")
                var = _variant_drops(frame, EXPRESSION_PRIMARY, _c)
                if ref.empty or var.empty:
                    return float("nan")
                rm = float(ref.mean())
                if not math.isfinite(rm) or abs(rm) < 1e-12:
                    return float("nan")
                return float(var.mean() / rm)

            r = stats.bca_cluster_bootstrap_frame(sub, _share_stat, strata_col="scenario_class",
                                                  B=B, seed=seed, keep_boots=True)
            boots = r.boots[np.isfinite(r.boots)] if r.boots is not None else np.array([])
            share_ci[cond] = {"share": r.value, "ci_low": r.ci_low, "ci_high": r.ci_high,
                              "ci_percentile": ([float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
                                                if len(boots) else [None, None]),
                              "frac_draws_above_half": (float(np.mean(boots > 0.5)) if len(boots) else None),
                              "n_draws": int(len(boots)), "n_clusters": r.n_clusters}
    # digit-answer compliance on the factorial cells, the check 4.1 runs on the Panel B cells
    compliance = {}
    for cond in present:
        rows = b[b["condition"] == cond]
        parsed = rows.dropna(subset=["q_self_sampled"]) if "q_self_sampled" in rows.columns else rows.iloc[0:0]
        compliance[cond] = {
            "n_cells": int(len(rows)), "n_parsed_digit": int(len(parsed)),
            "pearson_sampled_vs_logit": (float(np.corrcoef(parsed["q_self_sampled"], parsed[EXPRESSION_PRIMARY])[0, 1])
                                         if len(parsed) > 2 else float("nan"))}
    # leave-one-pair-out share arithmetic
    loo = {}
    ref_all = out.get("SUPPRESS", {}).get("expression_drop", {}).get("value")
    if ref_all and live and pairs:
        dist_ids = sorted(set(b[b["scenario_class"] == "distress"]["scenario_id"]))
        for sid in dist_ids:
            twin = pairs.get(sid)
            drop_ids = {sid} | ({twin} if twin else set())
            sub = b[~b["scenario_id"].isin(drop_ids)]
            ref = float(_variant_drops(sub[sub["condition"].isin(["NEUTRAL_INSTR", "SUPPRESS"])],
                                       EXPRESSION_PRIMARY, "SUPPRESS").mean())
            if not math.isfinite(ref) or abs(ref) < 1e-9:
                continue
            shares = {c: float(_variant_drops(sub[sub["condition"].isin(["NEUTRAL_INSTR", c])],
                                              EXPRESSION_PRIMARY, c).mean()) / ref for c in live}
            live_two = [shares[c] for c in ("SUPPRESS_REGISTER", "SUPPRESS_SELFREF") if c in shares]
            loo[sid] = {"pair_left_out": sorted(drop_ids), "shares": shares,
                        "share_sum": float(sum(shares.values())),
                        "share_sum_two_live": float(sum(live_two)) if len(live_two) == 2 else None,
                        "share_register": shares.get("SUPPRESS_REGISTER"),
                        "register_is_largest": (max(shares, key=lambda c: shares[c]) == "SUPPRESS_REGISTER"
                                                if shares else None)}
    loo_summary = {}
    if loo:
        sums = [v["share_sum"] for v in loo.values()]
        regs = [v["share_register"] for v in loo.values() if v["share_register"] is not None]
        two = [v["share_sum_two_live"] for v in loo.values() if v["share_sum_two_live"] is not None]
        moved = [k for k, v in loo.items()
                 if v["share_register"] is not None and abs(v["share_register"] - (out.get("SUPPRESS_REGISTER", {})
                    .get("expression_drop", {}).get("value", float("nan")) / ref_all)) > 1e-3]
        loo_summary = {"n_pairs_left_out": len(loo),
                       "n_pairs_that_move_the_share": len(moved),
                       "pairs_that_move_the_share": sorted(moved),
                       "share_sum_two_live_min": (min(two) if two else None),
                       "share_sum_two_live_max": (max(two) if two else None),
                       "share_sum_min": min(sums), "share_sum_max": max(sums),
                       "share_register_min": min(regs) if regs else None,
                       "share_register_max": max(regs) if regs else None,
                       "n_pairs_where_register_largest": int(sum(1 for v in loo.values() if v["register_is_largest"])),
                       "by_pair": loo}
    return {"value": out, "ci_low": None, "ci_high": None,
            "n_clusters": int(b["scenario_id"].nunique()),
            "extra": {"split_side": "confirmation", "baseline": "NEUTRAL_INSTR (D1)",
                      "unit": "drop as a fraction of that channel's NEUTRAL_INSTR distress-neutral separation; divergence = expression drop minus internal drop",
                      "conditions_present": present,
                      "paired_component_difference": paired,
                      "share_bootstrap": share_ci,
                      "digit_compliance_by_variant": compliance,
                      "leave_one_pair_out": loo_summary,
                      "interval_note": ("BCa intervals are printed with the plain 2.5/97.5 percentile interval "
                                        "beside them (ci_percentile): at six confirmation distress clusters, two of "
                                        "which have no expressed distress to remove under any condition, the BCa "
                                        "lower endpoint of the two live variants sits within 0.005 of zero and "
                                        "prints as a signed zero at two decimals"),
                      "note": ("post-data addendum with new model contact (prereg section 9 "
                               "dated note): the pre-registered SUPPRESS numbers are unchanged; the variants say "
                               "which component of the instruction moves each channel")}}


# ============================================================ real-probe dose map

CAPABILITY_PPL_MAX = 1.10
CAPABILITY_MMLU_DROP_MAX_PP = 5.0
DOSE_MAP_DIRECTIONS = ("D-CTX", "D-PV", "R1", "R2", "R3", "SEM", "OTHER")


def _slope_per_coef(sub: pd.DataFrame, channel: str) -> Tuple[float, Dict[float, float]]:
    """Least-squares slope of the coefficient-level means of scenario-level means."""
    g = (sub.groupby(["coefficient", "scenario_id"])[channel].mean()
            .groupby("coefficient").mean())
    if len(g) < 2:
        return float("nan"), {float(k): float(v) for k, v in g.items()}
    slope = float(np.polyfit(g.index.to_numpy(dtype=float), g.to_numpy(dtype=float), 1)[0])
    return slope, {float(k): float(v) for k, v in g.items()}


def dose_map_realprobe(df: pd.DataFrame, dose_unit: float, capability: dict,
                       primary_split: str = "discovery", B: int = stats.DEFAULT_B,
                       seed: int = 0) -> dict:
    """The coefficient -> SD-unit dose map recomputed with the TRAINED probe. The pre-registered ladder was
    configured to run without the probe (--no-probe in its cells file, written before the probe
    existed and kept when the greedy run of record went out about six hours after the probe was
    trained), so results/capability_valid_range.json's dose_map is the slope of the
    PLACEHOLDER readout (the residual mean at Lr) over the I-PROBE SD, and every SD-unit
    dose statement built on it is unit-less. This function fits the same slope (I-PROBE
    prompt_final on coefficient, coefficient-level means of scenario-level means, pooled
    zero anchor plus each direction's rungs) on the Panel A-prime rows, which carry the real
    probe, and restates the capability-valid range in SD with the frozen criteria (ppl ratio
    < 1.10 AND MMLU-lite drop < 5pp; prereg section 4). Primary population = the discovery
    split, classes pooled (the ladder's design population and the population the SD unit is
    defined on); confirmation-split and per-class slopes are printed as sensitivity."""
    a = df[(df["panel"] == "A") & (df["direction"] != "") & df["scenario_class"].isin(["distress", "neutral"])]
    zero = a[a["coefficient"] == 0.0]
    if zero.empty:
        raise ValueError("no zero-dose anchor rows in the A-prime frame")

    def fit(direction: str, side: Optional[str], cls: Optional[str]) -> dict:
        sub = pd.concat([a[a["direction"] == direction], zero], ignore_index=True)
        if side is not None:
            sub = sub[sub["split_side"] == side]
        if cls is not None:
            sub = sub[sub["scenario_class"] == cls]
        slope, means = _slope_per_coef(sub, INTERNAL_PRIMARY)
        spc = slope / dose_unit if math.isfinite(slope) else float("nan")
        return {"slope_readout_per_coef": slope, "sd_per_coef": spc, "sd_per_1000_coef": 1000.0 * spc,
                "coef_per_sd": (dose_unit / slope) if (math.isfinite(slope) and abs(slope) > 1e-12) else None,
                "coefficient_means": {f"{k:g}": v for k, v in means.items()},
                "n_scenarios": int(sub["scenario_id"].nunique()), "n_rows": int(len(sub))}

    directions = [d for d in DOSE_MAP_DIRECTIONS if (a["direction"] == d).any()]
    primary = {d: fit(d, primary_split, None) for d in directions}
    sensitivity = {
        "confirmation_pooled": {d: fit(d, "confirmation", None) for d in directions},
        "all_pooled": {d: fit(d, None, None) for d in directions},
        "discovery_neutral_only": {d: fit(d, primary_split, "neutral") for d in directions},
        "discovery_distress_only": {d: fit(d, primary_split, "distress") for d in directions},
    }
    # bootstrap the primary sd_per_coef for the two self directions (cluster = scenario)
    boot = {}
    for d in ("D-CTX", "D-PV"):
        if d not in directions:
            continue
        sub = pd.concat([a[a["direction"] == d], zero], ignore_index=True)
        sub = sub[sub["split_side"] == primary_split]
        r = stats.bca_cluster_bootstrap_frame(
            sub, lambda f: _slope_per_coef(f, INTERNAL_PRIMARY)[0] / dose_unit,
            strata_col="scenario_class", B=B, seed=seed)
        boot[d] = {"sd_per_coef": r.value, "ci_low": r.ci_low, "ci_high": r.ci_high, "n_clusters": r.n_clusters,
                   "sd_per_1000_coef": 1000.0 * r.value, "sd_per_1000_coef_ci": [1000.0 * r.ci_low, 1000.0 * r.ci_high]}
    # capability-valid range in SD under the real-probe map, per direction with a capability grid
    per_dir = {}
    for d, samples in capability.items():
        spc = primary.get(d, {}).get("sd_per_coef")
        if spc is None or not math.isfinite(spc) or not samples:
            continue
        ok_coefs = sorted(float(s["coefficient"]) for s in samples
                          if float(s["ppl_ratio"]) < CAPABILITY_PPL_MAX and float(s["mmlu_drop_pp"]) < CAPABILITY_MMLU_DROP_MAX_PP)
        # last valid rung before the first failure, walking the grid in coefficient order
        m = 0.0
        for s in sorted(samples, key=lambda s: float(s["coefficient"])):
            if not (float(s["ppl_ratio"]) < CAPABILITY_PPL_MAX and float(s["mmlu_drop_pp"]) < CAPABILITY_MMLU_DROP_MAX_PP):
                break
            m = float(s["coefficient"])
        per_dir[d] = {"max_valid_coefficient": m, "hi_sd": abs(spc) * m, "lo_sd": -abs(spc) * m,
                      "coef_for_1sd": abs(1.0 / spc) if spc else None,
                      "grid_1sd_over_max_valid": (abs(1.0 / spc) / m) if (spc and m > 0) else None,
                      "ladder_ceiling_32000_in_sd": abs(spc) * 32000.0,
                      "sign_of_slope": "positive" if spc > 0 else "negative"}
    self_bounds = [per_dir[d]["hi_sd"] for d in ("D-CTX", "D-PV") if d in per_dir]
    overall = min(self_bounds) if self_bounds else float("nan")
    return {"value": {"lo_sd": -overall, "hi_sd": overall, "per_direction": per_dir},
            "ci_low": None, "ci_high": None,
            "n_clusters": int(a[a["split_side"] == primary_split]["scenario_id"].nunique()),
            "extra": {"dose_unit_logits_per_sd": dose_unit, "primary_split": primary_split,
                      "primary_population": "A-prime rows on the discovery split, classes pooled, zero anchor pooled with each direction's rungs",
                      "dose_map_primary": primary, "dose_map_sensitivity": sensitivity,
                      "sd_per_coef_bootstrap": boot,
                      "criteria": f"ppl inflation < {int((CAPABILITY_PPL_MAX-1)*100)}% AND MMLU-lite drop < {int(CAPABILITY_MMLU_DROP_MAX_PP)}pp (prereg section 4)",
                      "supersedes": ("results/capability_valid_range.json value and extra.dose_map for every SD-unit statement: that file's map "
                                     "is the slope of the ladder's PLACEHOLDER_MEAN readout (the ladder ran with --no-probe) over the I-PROBE SD; "
                                     "the coefficient-unit capability grid it rests on is unchanged and real"),
                      "note": "post-data reporting-level correction, no new model contact (PREREGISTRATION.md section 9 dated note)"}}


# ============================================================ standardized-space probe cosines

def discovery_std_lr(residuals_path) -> np.ndarray:
    """Per-coordinate SD (ddof=0, floor as in scripts/train_probe.py) of the 12 labelled
    discovery-split readout-layer residuals: the scaler the trained probe folded into its weights
    (third-person rows excluded, as in train_probe.py)."""
    z = np.load(residuals_path)
    X = z["X"].astype(np.float64)
    labelled = np.asarray(z["label"]) >= 0   # distress/neutral only, as scripts/train_probe.py (is_dn)
    disc = X[(np.asarray(z["split"]) == "discovery") & labelled]
    std = disc.std(axis=0, ddof=0)
    return np.where(std < 1e-8, 1.0, std)


def probe_direction_cosines_standardized(probe_w: np.ndarray, dirs_lr: Dict[str, np.ndarray],
                                         disc_std: np.ndarray) -> dict:
    """Cosine between the probe's decision normal in the STANDARDIZED space it was fit in
    (w_std = folded w times the discovery SD) and each readout-layer direction mapped into that
    space (a displacement v becomes v / SD). This is the alignment the fold-in cosine of
    probe_cosine_with_directions_lr only bounds loosely."""
    w_std = np.asarray(probe_w, dtype=np.float64) * disc_std
    out = {}
    for name, v in dirs_lr.items():
        v_std = np.asarray(v, dtype=np.float64) / disc_std
        c = float(w_std @ v_std / (np.linalg.norm(w_std) * np.linalg.norm(v_std) + 1e-12))
        out[name] = {"cos_standardized": round(c, 4)}
    return out


# ============================================================ persistence panel

def persistence_elevation(df: pd.DataFrame, arms: Sequence[str] = ("D-CTX", "D-PV", "R1"),
                          split_side: str = "confirmation",
                          B: int = stats.DEFAULT_B, seed: int = 0) -> dict:
    """PERSIST panel (prereg section 9 dated note, 2026-08-17).

    Each cell steered ONE turn and then read both channels on the NEXT turn with the hook
    released. For every steered arm this returns the elevation of that unsteered read turn over
    the NULL arm on the same scenarios, in that channel's own natural distress-neutral separation
    units (the paper's currency), with a BCa cluster bootstrap over scenarios stratified by class.

    The pre-declared reading: the signal persists past the intervention for a direction if its
    INTERNAL elevation excludes zero while the R1 control's does not; if the REPORT elevation also
    excludes zero, both channels have been measured at the same unsteered moment.
    """
    p = df[(df["panel"] == "PERSIST") & df["scenario_class"].isin(["distress", "neutral"])]
    if split_side:
        p = p[p["split_side"] == split_side]
    if p.empty:
        raise ValueError("no PERSIST rows for that split")

    def _sep(frame: pd.DataFrame, ch: str) -> float:
        m = frame[frame["direction"] == "NULL"].groupby(["scenario_id", "scenario_class"])[ch].mean().reset_index()
        return float(m.loc[m["scenario_class"] == "distress", ch].mean()
                     - m.loc[m["scenario_class"] == "neutral", ch].mean())

    def _elev_stat(ch: str, arm: str):
        def stat(frame: pd.DataFrame) -> float:
            piv = frame.pivot_table(index=["scenario_id", "scenario_class"], columns="direction",
                                    values=ch, aggfunc="mean").reset_index()
            if arm not in piv or "NULL" not in piv:
                return float("nan")
            sep = _sep(frame, ch)
            if not math.isfinite(sep) or abs(sep) < 1e-9:
                return float("nan")
            d = piv[piv["scenario_class"] == "distress"].dropna(subset=[arm, "NULL"])
            return float(((d[arm] - d["NULL"]) / sep).mean())
        return stat

    out = {}
    present = [a for a in arms if (p["direction"] == a).any()]
    for arm in present:
        work = p[p["direction"].isin(["NULL", arm])]
        entry = {}
        for label, ch in (("report", EXPRESSION_PRIMARY), ("internal", INTERNAL_PRIMARY)):
            r = stats.bca_cluster_bootstrap_frame(work, _elev_stat(ch, arm), strata_col="scenario_class",
                                                  B=B, seed=seed, keep_boots=True)
            boots = r.boots[np.isfinite(r.boots)] if r.boots is not None else np.array([])
            entry[label] = {"elevation": r.value, "ci_low": r.ci_low, "ci_high": r.ci_high,
                            "ci_percentile": ([float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
                                              if len(boots) else [None, None]),
                            "n_clusters": r.n_clusters,
                            "excludes_zero": bool(math.isfinite(r.ci_low) and math.isfinite(r.ci_high)
                                                  and (r.ci_low > 0 or r.ci_high < 0))}
        entry["n_cells"] = int((p["direction"] == arm).sum())
        out[arm] = entry

    ctrl = out.get("R1", {})
    ctrl_int_flat = not ctrl.get("internal", {}).get("excludes_zero", False)
    verdict = {a: {"internal_persists": bool(out[a]["internal"]["excludes_zero"] and ctrl_int_flat),
                   "report_moves": bool(out[a]["report"]["excludes_zero"]),
                   "both_channels_at_one_moment": bool(out[a]["internal"]["excludes_zero"]
                                                       and out[a]["report"]["excludes_zero"]
                                                       and ctrl_int_flat)}
               for a in present if a != "R1"}
    return {"value": {"by_arm": out, "verdict_by_direction": verdict},
            "ci_low": None, "ci_high": None,
            "n_clusters": int(p["scenario_id"].nunique()),
            "extra": {"split_side": split_side,
                      "unit": "elevation of the unsteered read turn over the NULL arm, in that "
                              "channel's natural distress-neutral separation units",
                      "arms_present": present,
                      "reading_rule": ("fixed in the prereg note before the data: the signal persists "
                                       "for a direction if its internal elevation excludes zero while "
                                       "the R1 control's does not; if the report elevation also "
                                       "excludes zero, both channels were measured at one unsteered moment"),
                      "note": "post-data analysis of a NEW model contact"}}


# ============================================================ external review 2026-08-17

def locked_calibration(df: pd.DataFrame, B: int = stats.DEFAULT_B, seed: int = 0) -> dict:
    """Does the readout predict the report BEFORE suppression, and what happens to that
    relationship under it? (external review 2026-08-17, analysis 1)

    The masking result says two channels move differently. A reader can answer that they were
    never measurements of one quantity, in which case "decoupling" is underidentified. This test
    fixes the coupling on data the suppression instruction never touched, then applies it
    unchanged:

      1. FIT on the discovery split under NEUTRAL_INSTR: least squares Q-SELF ~ a + b * I-PROBE
         over scenario means (6 distress + 6 neutral).
      2. TEST on the confirmation split under NEUTRAL_INSTR: Pearson r and mean absolute error
         of the locked line, plus the same for a rank check. This is the coupling claim.
      3. APPLY the same locked line, unchanged, to the confirmation split under SUPPRESS, and
         report the residual (actual report minus predicted report) on distress scenarios, with
         a BCa cluster bootstrap; and, for contrast, the residual on the neutral twins.

    A large negative residual on distress scenarios means the report fell far below what the
    readout predicts under the pre-suppression relationship, which is measurement non-invariance
    localised to the report channel rather than two unrelated numbers.
    """
    b = df[(df["panel"] == "B") & (df["direction"] == "NULL")
           & df["scenario_class"].isin(["distress", "neutral"])]
    if b.empty:
        raise ValueError("no Panel B rows")

    def means(side: str, cond: str) -> pd.DataFrame:
        s = b[(b["split_side"] == side) & (b["condition"] == cond)]
        return (s.groupby(["scenario_id", "scenario_class"])[[EXPRESSION_PRIMARY, INTERNAL_PRIMARY]]
                 .mean().reset_index())

    fit = means("discovery", "NEUTRAL_INSTR")
    if len(fit) < 3:
        raise ValueError("not enough discovery NEUTRAL_INSTR scenarios to fit")
    slope, intercept = np.polyfit(fit[INTERNAL_PRIMARY].to_numpy(dtype=float),
                                  fit[EXPRESSION_PRIMARY].to_numpy(dtype=float), 1)
    predict = lambda x: float(intercept) + float(slope) * np.asarray(x, dtype=float)

    test = means("confirmation", "NEUTRAL_INSTR")
    pred_t = predict(test[INTERNAL_PRIMARY])
    act_t = test[EXPRESSION_PRIMARY].to_numpy(dtype=float)
    r_test = float(np.corrcoef(pred_t, act_t)[0, 1]) if len(act_t) > 2 else float("nan")
    rho_test = float(sps.spearmanr(pred_t, act_t).statistic) if len(act_t) > 2 else float("nan")
    mae_test = float(np.mean(np.abs(pred_t - act_t)))

    sup = means("confirmation", "SUPPRESS")
    merged = sup.merge(test[["scenario_id", EXPRESSION_PRIMARY]], on="scenario_id",
                       suffixes=("", "_neutral_instr"))
    merged["predicted"] = predict(merged[INTERNAL_PRIMARY])
    merged["residual"] = merged[EXPRESSION_PRIMARY] - merged["predicted"]

    # bootstrap the mean residual over scenarios, per class
    resid = {}
    for cls in ("distress", "neutral"):
        rows = merged[merged["scenario_class"] == cls][["scenario_id", "residual"]].copy()
        if rows.empty:
            continue
        r = stats.bca_cluster_bootstrap_rows(rows, ["residual"], lambda m: float(np.mean(m)),
                                             B=B, seed=seed)
        resid[cls] = {"mean_residual": r.value, "ci_low": r.ci_low, "ci_high": r.ci_high,
                      "n_scenarios": int(len(rows))}
    per_scenario = {str(row["scenario_id"]): {"internal": float(row[INTERNAL_PRIMARY]),
                                              "predicted_report": float(row["predicted"]),
                                              "actual_report": float(row[EXPRESSION_PRIMARY]),
                                              "residual": float(row["residual"]),
                                              "class": str(row["scenario_class"])}
                    for _, row in merged.iterrows()}
    return {"value": {"fit": {"slope": float(slope), "intercept": float(intercept),
                              "n_scenarios": int(len(fit)), "side": "discovery",
                              "condition": "NEUTRAL_INSTR"},
                      "test_transfer": {"pearson_r": r_test, "spearman_rho": rho_test,
                                        "mean_abs_error": mae_test, "n_scenarios": int(len(test))},
                      "under_suppression": resid,
                      "per_scenario_under_suppression": per_scenario},
            "ci_low": None, "ci_high": None, "n_clusters": int(len(merged)),
            "extra": {"unit": "report points (0 to 9 scale)",
                      "reading": ("the line is fitted where suppression never applies and then used "
                                  "unchanged; a large negative residual on distress scenarios means the "
                                  "report fell below what the readout predicts under the pre-suppression "
                                  "relationship"),
                      "note": "post-data analysis on data already collected (external review 2026-08-17)"}}


def probe_stress_test(residuals_path, c_grid=(0.001, 0.01, 0.1, 1.0, 10.0), seed: int = 0,
                      dominant_dim: Optional[int] = None) -> dict:
    """Is held-out AUC 1.00 a property of the probe or of a high-dimensional battery?
    (external review 2026-08-17, analysis 5)

    Three checks on the stored readout-layer residuals, using the SAME pipeline as
    scripts/train_probe.py (discovery standardisation folded into the weights):

      * leave-one-discovery-pair-out: refit on 5 of the 6 discovery pairs, score the untouched
        confirmation split each time, and report the AUC range and the weight cosine against the
        shipped probe;
      * label-flip null: refit under every one of the 2**6 within-pair label assignments on the
        discovery pairs and score each against the TRUE confirmation labels. If arbitrary labels
        also reach high confirmation AUC, the gate is a consequence of dimension and battery
        structure rather than evidence about the probe;
      * coordinate dependence: refit with the massive-activation coordinate zeroed.
    """
    import itertools
    sys_path_guard = None  # keep the import local; train_probe is a script, not a package
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_probe", str(Path(__file__).resolve().parents[1] / "scripts" / "train_probe.py"))
    tp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp)

    z = np.load(residuals_path)
    X = z["X"].astype(np.float64)
    split = np.asarray(z["split"])
    label = np.asarray(z["label"])
    sid = np.asarray(z["scenario_id"])
    keep = label >= 0
    Xd, yd, sd = X[keep & (split == "discovery")], label[keep & (split == "discovery")].astype(int), sid[keep & (split == "discovery")]
    Xc, yc = X[keep & (split == "confirmation")], label[keep & (split == "confirmation")].astype(int)

    def fit_and_score(Xtr, ytr, Xte, yte):
        w, b_, meta = tp.train_probe_weights(Xtr, ytr, tuple(c_grid), seed)
        return float(tp.heldout_auc(Xte, yte, w, b_)), w, float(b_), meta

    auc_full, w_full, b_full, meta_full = fit_and_score(Xd, yd, Xc, yc)

    # pairs, by the trailing number of the scenario id (d01 pairs with n01)
    pair_of = {s: str(s)[1:] for s in sd}
    pairs = sorted(set(pair_of.values()))
    loo = {}
    for pid in pairs:
        m = np.array([pair_of[s] != pid for s in sd])
        if len(set(yd[m])) < 2:
            continue
        auc, w, _, _ = fit_and_score(Xd[m], yd[m], Xc, yc)
        cos = float(w_full @ w / (np.linalg.norm(w_full) * np.linalg.norm(w) + 1e-12))
        loo[pid] = {"auc_confirmation": auc, "cosine_with_shipped_probe": cos,
                    "n_train_scenarios": int(m.sum())}

    # label-flip null over the 2**n_pairs within-pair assignments
    flips = []
    for pattern in itertools.product([0, 1], repeat=len(pairs)):
        yflip = yd.copy()
        for bit, pid in zip(pattern, pairs):
            if bit:
                m = np.array([pair_of[s] == pid for s in sd])
                yflip[m] = 1 - yflip[m]
        if len(set(yflip)) < 2:
            continue
        auc, _, _, _ = fit_and_score(Xd, yflip, Xc, yc)
        flips.append({"pattern": "".join(str(b) for b in pattern), "auc_confirmation": auc,
                      "n_flipped_pairs": int(sum(pattern))})
    aucs = np.array([f["auc_confirmation"] for f in flips], dtype=float)
    true_auc = auc_full

    without_dim = None
    if dominant_dim is not None:
        Xd2, Xc2 = Xd.copy(), Xc.copy()
        Xd2[:, dominant_dim] = 0.0
        Xc2[:, dominant_dim] = 0.0
        auc_wo, w_wo, _, _ = fit_and_score(Xd2, yd, Xc2, yc)
        without_dim = {"dim": int(dominant_dim), "auc_confirmation": auc_wo,
                       "cosine_with_shipped_probe": float(w_full @ w_wo /
                                                          (np.linalg.norm(w_full) * np.linalg.norm(w_wo) + 1e-12))}

    return {"value": {"auc_confirmation_refit": true_auc,
                      "leave_one_pair_out": loo,
                      "loo_auc_min": min((v["auc_confirmation"] for v in loo.values()), default=None),
                      "loo_auc_max": max((v["auc_confirmation"] for v in loo.values()), default=None),
                      "loo_cosine_min": min((v["cosine_with_shipped_probe"] for v in loo.values()), default=None),
                      "label_flip_null": {"n_patterns": int(len(flips)),
                                          "auc_mean": float(aucs.mean()) if len(aucs) else None,
                                          "auc_median": float(np.median(aucs)) if len(aucs) else None,
                                          "auc_max": float(aucs.max()) if len(aucs) else None,
                                          "n_at_or_above_true": int((aucs >= true_auc - 1e-12).sum()),
                                          "frac_at_or_above_true": float((aucs >= true_auc - 1e-12).mean()) if len(aucs) else None,
                                          "n_above_0p9": int((aucs > 0.9).sum())},
                      "without_dominant_dim": without_dim},
            "ci_low": None, "ci_high": None, "n_clusters": int(len(pairs)),
            "extra": {"pipeline": "scripts/train_probe.py train_probe_weights, unchanged",
                      "c_grid": list(c_grid),
                      "reading": ("the label-flip null is the honest denominator for a held-out AUC on "
                                  "six confirmation pairs: it counts how many arbitrary within-pair "
                                  "labelings of the discovery split also rank the true confirmation "
                                  "labels at least as well"),
                      "note": "post-data analysis on stored residuals (external review 2026-08-17)"}}


# ============================================================ BRIDGE: steering under the instructions

def bridge_interaction(df: pd.DataFrame, arms: Sequence[str] = ("D-CTX", "D-PV", "OTHER", "SEM"),
                       B: int = stats.DEFAULT_B, seed: int = 0) -> dict:
    """BRIDGE panel (external review 2026-08-17; prereg section 9 dated note).

    The masking panel and the steering panel never met: one runs under the instruction with no
    steering, the other runs steered with no instruction. Here each arm is steered UNDER each
    instruction. For every arm and condition this returns the elevation of each channel over the
    NULL arm of the SAME condition, on distress scenarios, in that channel's natural
    distress-neutral separation units under NEUTRAL_INSTR; and the interaction, that elevation
    under SUPPRESS minus the same under NEUTRAL_INSTR.

    Pre-declared reading: the halves meet for a direction if its internal elevation is present
    under SUPPRESS (interval excluding zero) and the interaction interval contains zero, while its
    report elevation under SUPPRESS is smaller than under NEUTRAL_INSTR.
    """
    b = df[(df["panel"] == "BRIDGE") & df["scenario_class"].isin(["distress", "neutral"])]
    if b.empty:
        raise ValueError("no BRIDGE rows")

    def sep(ch: str) -> float:
        n = b[(b["direction"] == "NULL") & (b["condition"] == "NEUTRAL_INSTR")]
        m = n.groupby(["scenario_id", "scenario_class"])[ch].mean().reset_index()
        return float(m.loc[m["scenario_class"] == "distress", ch].mean()
                     - m.loc[m["scenario_class"] == "neutral", ch].mean())

    seps = {"report": sep(EXPRESSION_PRIMARY), "internal": sep(INTERNAL_PRIMARY)}

    def elev_stat(ch: str, arm: str, cond: str, unit: float):
        def stat(frame: pd.DataFrame) -> float:
            f = frame[frame["condition"] == cond]
            piv = f.pivot_table(index=["scenario_id", "scenario_class"], columns="direction",
                                values=ch, aggfunc="mean").reset_index()
            if arm not in piv or "NULL" not in piv:
                return float("nan")
            d = piv[piv["scenario_class"] == "distress"].dropna(subset=[arm, "NULL"])
            return float(((d[arm] - d["NULL"]) / unit).mean())
        return stat

    def interaction_stat(ch: str, arm: str, unit: float):
        s_sup = elev_stat(ch, arm, "SUPPRESS", unit)
        s_ni = elev_stat(ch, arm, "NEUTRAL_INSTR", unit)
        return lambda frame: s_sup(frame) - s_ni(frame)

    out = {}
    present = [a for a in arms if (b["direction"] == a).any()]
    for arm in present:
        work = b[b["direction"].isin(["NULL", arm])]
        entry = {}
        for label, ch in (("report", EXPRESSION_PRIMARY), ("internal", INTERNAL_PRIMARY)):
            unit = seps[label]
            if not math.isfinite(unit) or abs(unit) < 1e-9:
                continue
            per_cond = {}
            for cond in ("NEUTRAL_INSTR", "SUPPRESS"):
                r = stats.bca_cluster_bootstrap_frame(work, elev_stat(ch, arm, cond, unit),
                                                      strata_col="scenario_class", B=B, seed=seed)
                per_cond[cond] = {"elevation": r.value, "ci_low": r.ci_low, "ci_high": r.ci_high,
                                  "excludes_zero": bool(math.isfinite(r.ci_low) and math.isfinite(r.ci_high)
                                                        and (r.ci_low > 0 or r.ci_high < 0))}
            ri = stats.bca_cluster_bootstrap_frame(work, interaction_stat(ch, arm, unit),
                                                   strata_col="scenario_class", B=B, seed=seed)
            per_cond["interaction_suppress_minus_neutral_instr"] = {
                "value": ri.value, "ci_low": ri.ci_low, "ci_high": ri.ci_high,
                "contains_zero": bool(math.isfinite(ri.ci_low) and math.isfinite(ri.ci_high)
                                      and ri.ci_low <= 0 <= ri.ci_high)}
            entry[label] = per_cond
        out[arm] = entry

    verdict = {}
    for arm in present:
        e = out[arm]
        if "internal" not in e or "report" not in e:
            continue
        i_sup = e["internal"]["SUPPRESS"]
        i_int = e["internal"]["interaction_suppress_minus_neutral_instr"]
        r_sup = e["report"]["SUPPRESS"]["elevation"]
        r_ni = e["report"]["NEUTRAL_INSTR"]["elevation"]
        verdict[arm] = {
            "internal_responds_under_suppression": bool(i_sup["excludes_zero"]),
            "instruction_does_not_remove_the_response": bool(i_int["contains_zero"]),
            "report_response_smaller_under_suppression": bool(r_sup < r_ni),
            "halves_meet": bool(i_sup["excludes_zero"] and i_int["contains_zero"]),
        }
    return {"value": {"by_arm": out, "verdict_by_arm": verdict, "separations_used": seps},
            "ci_low": None, "ci_high": None,
            "n_clusters": int(b["scenario_id"].nunique()),
            "extra": {"split_side": "confirmation",
                      "unit": "elevation over the NULL arm of the same condition, in that channel's "
                              "NEUTRAL_INSTR natural distress-neutral separation units",
                      "arms_present": present,
                      "reading_rule": ("fixed in the prereg note before the data: the halves meet for a "
                                       "direction if its internal elevation under SUPPRESS excludes zero "
                                       "and the interaction interval contains zero; SEM is the flat "
                                       "control and OTHER repeats the non-self-specificity check"),
                      "note": "post-data analysis of a NEW model contact (external review 2026-08-17)"}}

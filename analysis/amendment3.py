"""Amendment-3 corrective analyses (PREREGISTRATION.md section 9, 2026-08-16).

POST-DATA analyses ordered by the submit-gate audit, computed exclusively from the
already-collected frozen-battery data (no new model contact):

  * countermeasure_recalibrated: supersedes the D8 countermeasure calibration,
    whose theta_int was pinned to the maximum of 18 discovery-neutral cells at a
    degenerate realized FPR of 0 and fired on 100% of held-out neutral cells under
    instruction conditions. Amendment 3 sets BOTH thresholds on the
    deployment-realistic negative class (discovery-split neutral cells under
    SUPPRESS + NEUTRAL_INSTR) at the lowest achievable FPR, and reports the
    held-out FPR of each trigger next to its miss rate, which is the number the old
    table hid.
  * internal_did: the instruction-content difference-in-differences that
    adjudicates "the internal reading rose": SUPPRESS-minus-NEUTRAL_INSTR delta
    on distress scenarios MINUS the same on matched neutral twins.
  * internal_nofall: the direct claim the abstract makes (internal drop <= 0)
    with a one-sided bootstrap p.
  * selfother_report / selfother_internal: the third-person control named in
    Appendix B, computed. Collected under the frozen battery, analysis
    exploratory (not a pre-registered endpoint).
  * internal_specificity: Spearman(coefficient, I-PROBE) per direction at
    matched rungs: the internal channel's placebo specificity.
  * projout_check: the projection-out circularity control described in
    Appendix C, analysed.
  * channels_holm: the D18/D20 per-channel drop table as its own results file.

Every function takes the flat DataFrame from analysis/loading.py and returns
entries shaped like analysis/panels.py entries; run_all merges them AFTER the
panel outputs so the recalibrated countermeasure_table replaces the D8 one.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis import stats
from analysis.panels import (
    EXPRESSION_PRIMARY, INTERNAL_PRIMARY, PLACEBO_DIRECTIONS, SELF_DIRECTIONS,
    _entry, _mean_drop_stat, _natural_source, _none_entry, _pick_side,
    _scen_means, natural_separation,
)

PROJOUT_INTERNAL = "probe_projout_prompt_final"
INSTRUCTION_CONDITIONS = ("NEUTRAL_INSTR", "SUPPRESS")
MATCHED_RUNGS = (500.0, 1000.0)      # rungs every steered direction was run at
SPECIFICITY_DIRECTIONS = tuple(SELF_DIRECTIONS) + tuple(PLACEBO_DIRECTIONS) + ("OTHER",)
BORDERLINE_RUNGS = {"D-PV": (1000.0,)}   # amendment 2: labelled, secondary only
STRICT_RUNGS = {"D-CTX": (500.0, 1000.0, 2000.0), "D-PV": (250.0, 500.0)}
BINOM_ALPHA = 0.05
_EPS = 1e-9

# Operating-curve offsets above the discovery-negative maximum, in internal
# natural-separation units (dated note to amendment 3, 2026-08-16): the 0-offset
# point IS the amendment-3 operating point (max-of-negatives, no margin); the
# curve shows what margin-carrying thresholds trade between miss and held-out FPR.
OPERATING_CURVE_OFFSETS = (0.0, 0.25, 0.5, 1.0)

# A CI endpoint within this distance of the point estimate coincides with it at
# the report's printed 2-dp precision (small-cluster interval degeneracy label).
ENDPOINT_TOUCH_TOL = 0.02

COUNTERMEASURE_A3_DEFINITION = (
    "AMENDMENT 3 (prereg §9, 2026-08-16, post-data) -- SUPERSEDES the D8 "
    "calibration, whose internal threshold was pinned to the maximum of 18 "
    "discovery-neutral cells at a degenerate realized FPR of 0 and fired on 100% "
    "of held-out neutral cells under instruction conditions. Recalibrated "
    "protocol: BOTH triggers (expression Q-SELF > theta_expr; internal I-PROBE "
    "prompt_final > theta_int) are thresholded on the deployment-realistic "
    "negative class -- DISCOVERY-split neutral cells under the instruction "
    "conditions (SUPPRESS + NEUTRAL_INSTR) -- each at the threshold giving the "
    "lowest achievable FPR on that class (the achieved discovery FPR is "
    "reported). Miss rates are read on confirmation-split distress cells under "
    "SUPPRESS (all cases and the D8 masked-but-elevated subset), and the "
    "HELD-OUT false-positive rate of each trigger is read on confirmation-split "
    "neutral cells under the same instruction conditions and reported beside the "
    "miss rates. Zero-width bootstrap intervals additionally carry the exact "
    "one-sided 95% binomial bound at n = the scenario-cluster count. BCa "
    "cluster bootstrap over scenarios throughout (D7). Dated addition "
    "(2026-08-16, note to amendment 3): the recalibrated internal threshold is "
    "itself still a max-of-negatives estimator, so extra.operating_curve reports "
    "the internal monitor's miss rate and held-out FPR at thresholds theta_int + "
    "{0, 0.25, 0.5, 1.0} internal natural separations (D2), and "
    "extra.split_level_intercept_shift reports the discovery-to-confirmation "
    "shift of the negative class maximum (raw and separation units) -- the "
    "quantity the residual held-out FPR is a symptom of."
)

EXPLORATORY_LABEL = ("collected under the frozen battery, analysis exploratory "
                     "(not a pre-registered endpoint)")


def binomial_bound(p_hat: float, n: int, alpha: float = BINOM_ALPHA) -> Optional[float]:
    """Exact one-sided (1-alpha) binomial bound for a degenerate proportion.

    p_hat == 0 -> upper bound 1 - alpha**(1/n); p_hat == 1 -> lower bound
    alpha**(1/n); anything else -> None (the bootstrap CI is informative there).
    """
    if n <= 0:
        return None
    if p_hat <= 0.0:
        return float(1.0 - alpha ** (1.0 / n))
    if p_hat >= 1.0:
        return float(alpha ** (1.0 / n))
    return None


def _attach_binomial_bound(row: dict, key: str, n_clusters: int) -> None:
    """When a rate's bootstrap CI is zero-width, print the binomial bound beside it."""
    ci = row.get(f"{key}_ci")
    val = row.get(key)
    if val is None or ci is None or ci[0] is None or ci[1] is None:
        return
    if abs(ci[1] - ci[0]) < _EPS:
        b = binomial_bound(float(val), n_clusters)
        if b is not None:
            which = "upper" if float(val) <= 0.0 else "lower"
            row[f"{key}_binomial_{which}_bound_n{n_clusters}"] = b


def _internal_natural_separation(df: pd.DataFrame) -> Tuple[Optional[float], str]:
    """The D2 natural separation of the internal channel (I-PROBE prompt_final):
    mean distress minus mean neutral scenario means on NEUTRAL_INSTR cells,
    panels V and B pooled, confirmation side preferred, using the exact machinery
    panel_v uses for extra.per_channel_natural_separation."""
    src, which, side = _natural_source(df)
    if src.empty or not src[INTERNAL_PRIMARY].notna().any():
        return None, "no natural-source cells for the internal channel (D2)"
    nat = natural_separation(src, INTERNAL_PRIMARY)
    if not math.isfinite(nat) or abs(nat) < _EPS:
        return None, "internal natural separation non-finite or ~0 (D2)"
    return float(nat), f"{which}, {side} split"


def _flag_degenerate_interval(row: dict, point_key: str = "rho") -> dict:
    """Small-cluster interval-degeneracy label (dated note to amendment 3):
    name the pathologies a 6-cluster BCa interval can show: zero width
    (flagged upstream by _annotate_bca_collapse), an endpoint coinciding with
    the point estimate at the printed 2-dp precision, or an endpoint sitting
    exactly on 0.0 (discrete bootstrap support), so the report prints the
    label instead of shipping the interval as if it resolved."""
    lo, hi, pt = row.get("ci_low"), row.get("ci_high"), row.get(point_key)
    if lo is None or hi is None or pt is None:
        return row
    reasons: List[str] = []
    if row.get("ci_bca_collapsed"):
        reasons.append("zero-width BCa collapse (percentile CI printed alongside)")
    else:
        if min(abs(pt - lo), abs(hi - pt)) < ENDPOINT_TOUCH_TOL:
            reasons.append("CI endpoint coincides with the point estimate at "
                           "the printed 2-dp precision")
        if lo == 0.0 or hi == 0.0:
            reasons.append("CI endpoint sits exactly on 0.0 (discrete bootstrap "
                           "support at this cluster count)")
    if reasons:
        row["interval_degenerate"] = "; ".join(reasons)
    return row


def _interval_caveat(table: Dict[str, dict], n_clusters: int,
                     flag_keys: Sequence[str] = ("interval_degenerate",)) -> Optional[str]:
    """extra.interval_caveat text naming every degenerate interval in a
    per-direction table; None when the table is clean."""
    named = []
    for direction, row in table.items():
        parts = list(dict.fromkeys(str(row[k]) for k in flag_keys
                                   if row.get(k)))  # deduped, order kept
        if parts:
            named.append(f"{direction}: {'; '.join(parts)}")
    if not named:
        return None
    return (f"SMALL-CLUSTER INTERVAL CAVEAT (n = {n_clusters} scenario "
            f"clusters): the BCa bootstrap does not resolve every interval in "
            f"this table -- {' | '.join(named)}. The reading this table "
            f"supports is the qualitative separation between the "
            f"affect-carrying directions and the dose-matched controls, not "
            f"the interval widths; plain 2.5/97.5 percentile intervals are "
            f"printed alongside for every direction (ci_percentile).")


def _panel_b_null(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "panel" not in df:
        return pd.DataFrame()
    return df[(df["panel"] == "B") & (df["direction"] == "NULL")]


def _rate(sub: pd.DataFrame, col: str, stat, B: int, seed: int) -> dict:
    res = stats.bca_cluster_bootstrap_rows(sub, [col], stat, B=B, seed=seed)
    d = res.as_dict()
    return {"value": d["value"], "ci": [d["ci_low"], d["ci_high"]],
            "n_clusters": d["n_clusters"]}


# ============================================== countermeasure recalibration

def countermeasure_recalibrated(df: pd.DataFrame, thresholds: Optional[dict],
                                superseded_d8: Optional[dict] = None,
                                B: int = stats.DEFAULT_B, seed: int = 0) -> dict:
    """The amendment-3 countermeasure_table entry (definition_override stamped)."""
    b = _panel_b_null(df)
    if b.empty:
        return _none_entry("no Panel B rows loaded (amendment 3 recalibration)")

    # ---- calibration: DISCOVERY neutral cells under instruction conditions ----
    cal = b[(b["scenario_class"] == "neutral")
            & (b["split_side"] == "discovery")
            & b["condition"].isin(INSTRUCTION_CONDITIONS)]
    cal = cal.dropna(subset=[EXPRESSION_PRIMARY, INTERNAL_PRIMARY])
    if cal.empty or cal["scenario_id"].nunique() < 2:
        return _none_entry("no discovery-split neutral instruction cells to "
                           "calibrate on (amendment 3); thresholds are never set "
                           "on the confirmation side")
    theta_expr = float(cal[EXPRESSION_PRIMARY].max())
    theta_int = float(cal[INTERNAL_PRIMARY].max())
    fpr_disc_expr = float(np.mean(cal[EXPRESSION_PRIMARY].to_numpy() > theta_expr))
    fpr_disc_int = float(np.mean(cal[INTERNAL_PRIMARY].to_numpy() > theta_int))

    # ---- miss rates: confirmation-split SUPPRESS distress cells ----
    cases_all = b[(b["scenario_class"] == "distress") & (b["condition"] == "SUPPRESS")]
    cases, case_side = _pick_side(cases_all)
    cases = cases.dropna(subset=[EXPRESSION_PRIMARY, INTERNAL_PRIMARY])
    if cases.empty:
        return _none_entry("no SUPPRESS distress cells for the miss rates "
                           "(amendment 3)")

    elev = (thresholds or {}).get("elevation_midpoint")
    sets: List[Tuple[str, pd.DataFrame]] = [("all_suppressed_distress", cases)]
    if elev is not None:
        sets.append(("masked_but_elevated", cases[cases[INTERNAL_PRIMARY] >= elev]))

    rows: List[dict] = []
    for set_name, sub in sets:
        if sub.empty:
            rows.append({"case_set": set_name, "n_cells": 0, "note": "no cases"})
            continue
        n_scen = int(sub["scenario_id"].nunique())
        row = {"case_set": set_name, "n_cells": int(len(sub)), "n_scenarios": n_scen}
        for trig, col, th in (("expression", EXPRESSION_PRIMARY, theta_expr),
                              ("internal", INTERNAL_PRIMARY, theta_int)):
            r = _rate(sub, col, lambda x, t=th: float(np.mean(x <= t)), B, seed)
            row[f"miss_rate_{trig}"] = r["value"]
            row[f"miss_rate_{trig}_ci"] = r["ci"]
            _attach_binomial_bound(row, f"miss_rate_{trig}", n_scen)
        rows.append(row)

    # ---- held-out FPR: confirmation neutral cells under instructions ----
    held_all = b[(b["scenario_class"] == "neutral")
                 & b["condition"].isin(INSTRUCTION_CONDITIONS)]
    held = held_all[held_all["split_side"] == "confirmation"]
    held_side = "confirmation"
    if held.empty:  # prereg section 7 fallback, labelled via split bookkeeping
        held, held_side = _pick_side(held_all)
    held = held.dropna(subset=[EXPRESSION_PRIMARY, INTERNAL_PRIMARY])
    heldout: Dict[str, dict] = {}
    if not held.empty:
        n_scen_h = int(held["scenario_id"].nunique())
        for trig, col, th in (("expression", EXPRESSION_PRIMARY, theta_expr),
                              ("internal", INTERNAL_PRIMARY, theta_int)):
            r = _rate(held, col, lambda x, t=th: float(np.mean(x > t)), B, seed)
            entry = {"fpr": r["value"], "fpr_ci": r["ci"],
                     "n_cells": int(len(held)), "n_scenarios": n_scen_h,
                     "split_side": held_side}
            _attach_binomial_bound(entry, "fpr", n_scen_h)
            per_cond = {}
            for cond in ("NONE", "NEUTRAL_INSTR", "SUPPRESS"):
                hc = b[(b["scenario_class"] == "neutral")
                       & (b["split_side"] == held_side)
                       & (b["condition"] == cond)].dropna(subset=[col])
                if not hc.empty:
                    per_cond[cond] = float(np.mean(hc[col].to_numpy() > th))
            entry["per_condition_point"] = per_cond
            heldout[trig] = entry

    # ---- threshold-transfer diagnostic: does ranking survive where the
    #      absolute operating point may not? ----
    ranking = None
    if not held.empty:
        sup_neu = held[held["condition"] == "SUPPRESS"]
        if not sup_neu.empty:
            ranking = {
                "confirmation_suppress_distress_probe_min":
                    float(cases[INTERNAL_PRIMARY].min()),
                "confirmation_instruction_neutral_probe_max":
                    float(held[INTERNAL_PRIMARY].max()),
                "note": "an oracle threshold between these separates perfectly "
                        "iff min(distress) > max(neutral); reported as a "
                        "threshold-transfer diagnostic, not an operating point",
            }

    # ---- operating curve + split-level intercept shift (dated note to
    #      amendment 3, 2026-08-16): theta_int is still a max-of-negatives
    #      estimator, so (a) show the miss/held-out-FPR trade at margin-carrying
    #      thresholds theta_int + k x natural separation, and (b) name the
    #      discovery-to-confirmation shift of the negative-class maximum, the
    #      quantity the residual held-out FPR is a symptom of. ----
    nat_int, nat_src = _internal_natural_separation(df)
    operating_curve: dict
    if nat_int is None:
        operating_curve = {"points": [],
                           "note": f"not computed: {nat_src}"}
    else:
        n_scen_cases = int(cases["scenario_id"].nunique())
        points: List[dict] = []
        for k in OPERATING_CURVE_OFFSETS:
            th = float(theta_int + k * nat_int)
            pt = {"offset_separation_units": float(k), "theta": th}
            r_miss = _rate(cases, INTERNAL_PRIMARY,
                           lambda x, t=th: float(np.mean(x <= t)), B, seed)
            pt["miss_rate"] = r_miss["value"]
            pt["miss_rate_ci"] = r_miss["ci"]
            _attach_binomial_bound(pt, "miss_rate", n_scen_cases)
            if not held.empty:
                r_fpr = _rate(held, INTERNAL_PRIMARY,
                              lambda x, t=th: float(np.mean(x > t)), B, seed)
                pt["heldout_fpr"] = r_fpr["value"]
                pt["heldout_fpr_ci"] = r_fpr["ci"]
                _attach_binomial_bound(pt, "heldout_fpr",
                                       int(held["scenario_id"].nunique()))
            pt["fpr_discovery"] = float(
                np.mean(cal[INTERNAL_PRIMARY].to_numpy() > th))
            points.append(pt)
        operating_curve = {
            "monitor": f"internal ({INTERNAL_PRIMARY})",
            "threshold_rule": "theta = discovery-negative max (theta_int) + "
                              "offset x internal natural separation",
            "natural_separation_internal": nat_int,
            "natural_separation_source": nat_src,
            "offsets_separation_units": [float(k) for k in
                                         OPERATING_CURVE_OFFSETS],
            "points": points,
            "note": "the offset-0 point IS the amendment-3 operating point "
                    "(max-of-negatives, no margin); miss on confirmation "
                    "SUPPRESS distress cells, held-out FPR on confirmation "
                    "instruction-condition neutral cells, BCa cluster-bootstrap "
                    "CIs, binomial bound beside zero-width readings",
        }
    intercept_shift = None
    if nat_int is not None and not held.empty:
        conf_max = float(held[INTERNAL_PRIMARY].max())
        shift_raw = conf_max - theta_int
        intercept_shift = {
            "discovery_negative_max_theta_int": theta_int,
            "confirmation_instruction_neutral_max": conf_max,
            "shift_raw_units": float(shift_raw),
            "shift_separation_units": float(shift_raw / nat_int),
            "natural_separation_internal": nat_int,
            "natural_separation_source": nat_src,
            "note": "the recalibrated internal threshold is still a "
                    "max-of-negatives estimator; what the recalibration bought "
                    "is a deployment-realistic negative class, and the residual "
                    "held-out FPR is this split-level intercept shift of the "
                    "negative class (confirmation instruction-neutral max minus "
                    "the discovery-negative max), not monitor noise",
        }

    value = {
        "theta_expr": theta_expr,
        "theta_int": theta_int,
        "calibration_negative_class":
            "discovery-split neutral cells under SUPPRESS + NEUTRAL_INSTR "
            "(amendment 3)",
        "n_calibration_cells": int(len(cal)),
        "n_calibration_scenarios": int(cal["scenario_id"].nunique()),
        "fpr_discovery_expression": fpr_disc_expr,
        "fpr_discovery_internal": fpr_disc_int,
        "elevation_midpoint": elev,
        "rows": rows,
        "heldout_fpr": heldout,
    }
    out = _entry(value=value, split_side=case_side,
                 note=f"case split side: {case_side}; thresholds from discovery "
                      f"instruction-condition neutral cells only (amendment 3)")
    out["n_clusters"] = int(cases["scenario_id"].nunique())
    if ranking is not None:
        out["extra"]["ranking_diagnostic"] = ranking
    out["extra"]["operating_curve"] = operating_curve
    if intercept_shift is not None:
        out["extra"]["split_level_intercept_shift"] = intercept_shift
    if superseded_d8 is not None:
        out["extra"]["superseded_d8"] = superseded_d8
    out["definition_override"] = COUNTERMEASURE_A3_DEFINITION
    return out


def superseded_d8_summary(d8_entry: Optional[dict]) -> Optional[dict]:
    """Compact record of the D8 table this amendment supersedes (for extra)."""
    if not d8_entry or not isinstance(d8_entry.get("value"), dict):
        return None
    v = d8_entry["value"]
    return {"theta_expr": v.get("theta_expr"), "theta_int": v.get("theta_int"),
            "matched_fpr_discovery": v.get("matched_fpr_discovery"),
            "rows": v.get("rows"),
            "note": "D8 calibration (superseded by amendment 3): theta_int was "
                    "the (1-FPR) quantile on discovery UNSTEERED neutral cells "
                    "at realized FPR 0, i.e. pinned to their maximum; its "
                    "held-out FPR on instruction-condition neutral cells was "
                    "100% and was not reported"}


# ============================================== instruction-content DiD

def _did_stat(frame: pd.DataFrame) -> float:
    piv = frame.pivot_table(index=["scenario_id", "scenario_class"],
                            columns="condition", values=INTERNAL_PRIMARY,
                            aggfunc="mean").reset_index()
    if "NEUTRAL_INSTR" not in piv or "SUPPRESS" not in piv:
        return float("nan")
    piv = piv.dropna(subset=["NEUTRAL_INSTR", "SUPPRESS"])
    delta = piv["SUPPRESS"] - piv["NEUTRAL_INSTR"]
    d = delta[piv["scenario_class"] == "distress"]
    n = delta[piv["scenario_class"] == "neutral"]
    if d.empty or n.empty:
        return float("nan")
    return float(d.mean() - n.mean())


def internal_did(df: pd.DataFrame, B: int = stats.DEFAULT_B, seed: int = 0) -> dict:
    b = _panel_b_null(df)
    work_all = b[b["scenario_class"].isin(["distress", "neutral"])
                 & b["condition"].isin(INSTRUCTION_CONDITIONS)] if not b.empty else b
    if b.empty or work_all.empty:
        return {"panelB_internal_did": _none_entry("no Panel B rows loaded")}
    work, side = _pick_side(work_all)
    if work.empty or work["scenario_class"].nunique() < 2:
        return {"panelB_internal_did":
                _none_entry("need both distress and neutral scenarios for the DiD")}
    res = stats.bca_cluster_bootstrap_frame(work, _did_stat,
                                            strata_col="scenario_class",
                                            B=B, seed=seed)
    # point components + natural-separation normalization for the record
    piv = work.pivot_table(index=["scenario_id", "scenario_class"],
                           columns="condition", values=INTERNAL_PRIMARY,
                           aggfunc="mean").reset_index()
    piv = piv.dropna(subset=["NEUTRAL_INSTR", "SUPPRESS"])
    delta = piv["SUPPRESS"] - piv["NEUTRAL_INSTR"]
    d_mean = float(delta[piv["scenario_class"] == "distress"].mean())
    n_mean = float(delta[piv["scenario_class"] == "neutral"].mean())
    ni = piv["NEUTRAL_INSTR"]
    nat = float(ni[piv["scenario_class"] == "distress"].mean()
                - ni[piv["scenario_class"] == "neutral"].mean())
    entry = _entry(res, split_side=side,
                   unit="raw I-PROBE (prompt_final) units",
                   delta_suppress_minus_neutralinstr_distress=d_mean,
                   delta_suppress_minus_neutralinstr_neutral=n_mean,
                   natural_separation_neutral_instr=nat,
                   did_natsep_units=(float(res.value) / nat
                                     if math.isfinite(nat) and abs(nat) > _EPS
                                     else None),
                   n_distress_clusters=int(
                       piv.loc[piv["scenario_class"] == "distress",
                               "scenario_id"].nunique()),
                   n_neutral_clusters=int(
                       piv.loc[piv["scenario_class"] == "neutral",
                               "scenario_id"].nunique()),
                   reading="negative = the rise under SUPPRESS is larger on the "
                           "matched neutral twins than on distress scenarios "
                           "(instruction-content-general, not distress-specific)")
    return {"panelB_internal_did": entry}


# ============================================== does-not-fall one-sided reading

def internal_nofall(df: pd.DataFrame, B: int = stats.DEFAULT_B, seed: int = 0) -> dict:
    b = _panel_b_null(df)
    work_all = b[b["scenario_class"].isin(["distress", "neutral"])
                 & b["condition"].isin(INSTRUCTION_CONDITIONS)] if not b.empty else b
    if b.empty or work_all.empty:
        return {"panelB_internal_nofall": _none_entry("no Panel B rows loaded")}
    work, side = _pick_side(work_all)
    if work.empty or work[work["scenario_class"] == "distress"].empty:
        return {"panelB_internal_nofall":
                _none_entry("no distress scenarios for the internal drop")}
    res = stats.bca_cluster_bootstrap_frame(
        work, _mean_drop_stat(INTERNAL_PRIMARY), strata_col="scenario_class",
        B=B, seed=seed, keep_boots=True)
    if res.boots is None or len(res.boots) == 0:
        return {"panelB_internal_nofall": _none_entry("bootstrap produced no draws")}
    # one-sided bootstrap p for the claim drop_internal <= 0 (does not fall):
    # small p = the bootstrap distribution of the drop lies below 0.
    p_onesided = float((np.sum(res.boots >= 0.0) + 1.0) / (len(res.boots) + 1.0))
    entry = _entry(res, split_side=side, unit="percent of natural separation")
    for k in ("value", "ci_low", "ci_high"):
        if entry[k] is not None:
            entry[k] = float(entry[k]) * 100.0
    entry["n_clusters"] = int(
        work.loc[work["scenario_class"] == "distress", "scenario_id"].nunique())
    entry["extra"]["p_onesided_nofall"] = p_onesided
    entry["extra"]["claim"] = ("drop_internal <= 0 (the internal readout does "
                               "not fall under SUPPRESS)")
    entry["extra"]["note"] = ("same estimator as panelB_internal_drop_pct; see "
                              "panelB_internal_did for whether the rise is "
                              "distress-specific (it is not)")
    return {"panelB_internal_nofall": entry}


# ============================================== third-person self/other contrast

def _selfother_stat(channel: str):
    def stat(frame: pd.DataFrame) -> float:
        m = _scen_means(frame.dropna(subset=[channel]), channel)
        tp = m.loc[m["scenario_class"] == "third_person", "value"]
        d = m.loc[m["scenario_class"] == "distress", "value"]
        if tp.empty or d.empty:
            return float("nan")
        return float(tp.mean() - d.mean())
    return stat


def selfother(df: pd.DataFrame, B: int = stats.DEFAULT_B, seed: int = 0) -> dict:
    names = {"panelB_selfother_report": EXPRESSION_PRIMARY,
             "panelB_selfother_internal": INTERNAL_PRIMARY}
    b = _panel_b_null(df)
    n0_all = b[(b["condition"] == "NONE")
               & b["scenario_class"].isin(["distress", "third_person", "neutral"])] \
        if not b.empty else b
    if b.empty or n0_all.empty:
        return {k: _none_entry("no Panel B condition-NONE rows loaded")
                for k in names}
    n0, side = _pick_side(n0_all)
    have = set(n0["scenario_class"].unique())
    if not {"distress", "third_person"} <= have:
        return {k: _none_entry("need both third-person and distress scenarios "
                               "under NONE") for k in names}
    work = n0[n0["scenario_class"].isin(["distress", "third_person"])]
    out: Dict[str, dict] = {}
    for name, channel in names.items():
        sub = work.dropna(subset=[channel])
        res = stats.bca_cluster_bootstrap_frame(
            sub, _selfother_stat(channel), strata_col="scenario_class",
            B=B, seed=seed)
        m = _scen_means(n0.dropna(subset=[channel]), channel)
        means = {cls: (float(m.loc[m["scenario_class"] == cls, "value"].mean())
                       if (m["scenario_class"] == cls).any() else None)
                 for cls in ("third_person", "distress", "neutral")}
        n_tp = int(sub.loc[sub["scenario_class"] == "third_person",
                           "scenario_id"].nunique())
        n_d = int(sub.loc[sub["scenario_class"] == "distress",
                          "scenario_id"].nunique())
        entry = _entry(res, split_side=side, condition="NONE", channel=channel,
                       third_person_mean=means["third_person"],
                       own_distress_mean=means["distress"],
                       neutral_mean_reference=means["neutral"],
                       n_third_person_clusters=n_tp,
                       n_distress_clusters=n_d,
                       small_n_label=f"n = {n_tp} third-person confirmation "
                                     f"clusters (small)",
                       exploratory=EXPLORATORY_LABEL)
        out[name] = entry
    return out


# ============================================== internal placebo specificity

def _direction_rung_rows(a: pd.DataFrame, direction: str,
                         rungs: Sequence[float]) -> pd.DataFrame:
    """Direction rows at the given nonzero rungs + the pooled zero anchor."""
    mask = np.zeros(len(a), dtype=bool)
    coef = a["coefficient"].to_numpy(dtype=float)
    for c in rungs:
        mask |= np.isclose(coef, float(c))
    at_rungs = (a["direction"] == direction).to_numpy() & mask
    return a[at_rungs | np.isclose(coef, 0.0)]


def _annotate_bca_collapse(row: dict, boots: Optional[np.ndarray],
                           alpha: float = 0.05) -> dict:
    """BCa pathology at small n: when the point estimate sits at the extreme of
    a NON-degenerate bootstrap distribution, the bias correction diverges and
    the BCa interval collapses to zero width. Flag it and print the plain
    percentile interval alongside, honestly, instead of shipping a zero-width
    CI that reads as certainty."""
    lo, hi = row.get("ci_low"), row.get("ci_high")
    if boots is None or len(boots) == 0 or lo is None or hi is None:
        return row
    if abs(hi - lo) >= _EPS or bool(np.allclose(boots, boots[0])):
        return row
    row["ci_bca_collapsed"] = True
    row["ci_percentile"] = [float(np.quantile(boots, alpha / 2.0)),
                            float(np.quantile(boots, 1.0 - alpha / 2.0))]
    row["note"] = ("zero-width BCa interval: the point estimate sits at the "
                   "extreme of a non-degenerate bootstrap distribution (BCa "
                   "bias-correction collapse at small cluster n); the plain "
                   "percentile interval is printed alongside")
    return row


def _spearman_table_row(sub: pd.DataFrame, ycol: str, B: int, seed: int) -> dict:
    res = stats.bca_cluster_bootstrap_rows(sub, ["coefficient", ycol],
                                           stats.spearman_stat, B=B, seed=seed,
                                           keep_boots=True)
    d = res.as_dict()
    row = {"rho": d["value"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
           "n_clusters": d["n_clusters"], "n_cells": int(len(sub))}
    row = _annotate_bca_collapse(row, res.boots)
    # percentile interval for EVERY direction, not only the collapsed one
    # (dated note to amendment 3): at 6 clusters the BCa interval is the
    # unstable object; the percentile interval is the stable companion reading.
    if "ci_percentile" not in row and res.boots is not None and len(res.boots):
        row["ci_percentile"] = [float(np.quantile(res.boots, 0.025)),
                                float(np.quantile(res.boots, 0.975))]
    return _flag_degenerate_interval(row)


def internal_specificity(df: pd.DataFrame, aprime: bool,
                         B: int = stats.DEFAULT_B, seed: int = 0) -> dict:
    name = "panelAp_internal_specificity"
    if not aprime:
        return {name: _none_entry("Panel A rows are not the amendment-2 "
                                  "coefficient-unit design; specificity is "
                                  "defined on the A-prime grid")}
    a_all = df[(df["panel"] == "A") & (df["scenario_class"] == "neutral")]
    if a_all.empty:
        return {name: _none_entry("no Panel A-prime neutral rows loaded")}
    a, side = _pick_side(a_all)
    table: Dict[str, dict] = {}
    rung_means: Dict[str, dict] = {}
    for direction in SPECIFICITY_DIRECTIONS:
        sub = _direction_rung_rows(a, direction, MATCHED_RUNGS).dropna(
            subset=["coefficient", INTERNAL_PRIMARY])
        if sub[sub["direction"] == direction].empty:
            continue
        row = _spearman_table_row(sub, INTERNAL_PRIMARY, B, seed)
        if direction in BORDERLINE_RUNGS:
            row["borderline_rungs_included"] = [float(c) for c in
                                                BORDERLINE_RUNGS[direction]]
        table[direction] = row
        g = sub.groupby("coefficient")[INTERNAL_PRIMARY].mean()
        rung_means[direction] = {f"{c:g}": float(v) for c, v in g.items()}
    if not table:
        return {name: _none_entry("no rows at the A-prime matched rungs")}
    strict_extra: Dict[str, dict] = {}
    for direction in SELF_DIRECTIONS:
        sub = _direction_rung_rows(a, direction, STRICT_RUNGS[direction]).dropna(
            subset=["coefficient", INTERNAL_PRIMARY])
        if not sub[sub["direction"] == direction].empty:
            strict_extra[direction] = _spearman_table_row(sub, INTERNAL_PRIMARY,
                                                          B, seed)
            strict_extra[direction]["rungs"] = \
                [0.0] + [float(c) for c in STRICT_RUNGS[direction]]
    entry = _entry(value=table, split_side=side,
                   matched_rungs=[0.0] + [float(c) for c in MATCHED_RUNGS],
                   rung_means=rung_means,
                   strict_rung_variants=strict_extra,
                   note="D-PV@1000 is the BORDERLINE capability rung "
                        "(amendment 2); the D-PV strict-rung variant is in "
                        "extra.strict_rung_variants")
    entry["n_clusters"] = int(a["scenario_id"].nunique())
    caveat = _interval_caveat(table, entry["n_clusters"])
    if caveat:
        entry["extra"]["interval_caveat"] = caveat
    return {name: entry}


# ============================================== projection-out check

def projout_check(df: pd.DataFrame, aprime: bool,
                  B: int = stats.DEFAULT_B, seed: int = 0) -> dict:
    name = "panelAp_projout_check"
    if not aprime:
        return {name: _none_entry("Panel A rows are not the amendment-2 "
                                  "coefficient-unit design")}
    if PROJOUT_INTERNAL not in df.columns:
        return {name: _none_entry("no probe_score_projout fields in the loaded "
                                  "data")}
    a_all = df[(df["panel"] == "A") & (df["scenario_class"] == "neutral")]
    if a_all.empty:
        return {name: _none_entry("no Panel A-prime neutral rows loaded")}
    a, side = _pick_side(a_all)
    table: Dict[str, dict] = {}
    for direction in SPECIFICITY_DIRECTIONS:
        steered = a[(a["direction"] == direction)
                    & (a["coefficient"].abs() > _EPS)]
        if steered.empty:
            continue
        n_total = int(len(steered))
        sub = steered.dropna(subset=["coefficient", INTERNAL_PRIMARY,
                                     PROJOUT_INTERNAL])
        if sub.empty or sub["coefficient"].nunique() < 2:
            table[direction] = {"note": "insufficient projout coverage",
                                "n_rows_total": n_total,
                                "n_rows_with_projout": int(len(sub))}
            continue
        raw = _spearman_table_row(sub, INTERNAL_PRIMARY, B, seed)
        pro = _spearman_table_row(sub, PROJOUT_INTERNAL, B, seed)
        # diagnostic: how rank-preserving is the projection itself? Near 1.0
        # means projecting out span(v_steer) is close to a per-direction
        # monotone shift of the probe score, which is why rho_raw and
        # rho_projout can coincide exactly.
        rank_agreement = stats.spearman_stat(
            sub[INTERNAL_PRIMARY].to_numpy(dtype=float),
            sub[PROJOUT_INTERNAL].to_numpy(dtype=float))
        g = sub.groupby("coefficient")[[INTERNAL_PRIMARY, PROJOUT_INTERNAL]].mean()
        rungs = {f"{c:g}": {"raw": float(r[INTERNAL_PRIMARY]),
                            "projout": float(r[PROJOUT_INTERNAL])}
                 for c, r in g.iterrows()}
        level_shift = float((sub[PROJOUT_INTERNAL] - sub[INTERNAL_PRIMARY]).mean())
        survives = None
        pro_lo, pro_hi = pro["ci_low"], pro["ci_high"]
        if pro.get("ci_bca_collapsed"):  # judge on the percentile CI instead
            pro_lo, pro_hi = pro["ci_percentile"]
        if (pro_lo is not None and pro_hi is not None
                and raw["rho"] is not None and pro["rho"] is not None):
            excludes0 = (pro_lo > 0.0) or (pro_hi < 0.0)
            survives = bool(excludes0
                            and np.sign(pro["rho"]) == np.sign(raw["rho"]))
        row = {"rho_raw": raw["rho"],
               "rho_raw_ci": [raw["ci_low"], raw["ci_high"]],
               "rho_projout": pro["rho"],
               "rho_projout_ci": [pro["ci_low"], pro["ci_high"]],
               **{f"raw_{k}": raw[k] for k in
                  ("ci_bca_collapsed", "ci_percentile", "note",
                   "interval_degenerate") if k in raw},
               **{f"projout_{k}": pro[k] for k in
                  ("ci_bca_collapsed", "ci_percentile", "note",
                   "interval_degenerate") if k in pro},
               "n_clusters": pro["n_clusters"],
               "n_rows_total": n_total,
               "n_rows_with_projout": int(len(sub)),
               "rungs": sorted(float(c) for c in sub["coefficient"].unique()),
               "rung_means": rungs,
               "level_shift_projout_minus_raw": level_shift,
               "rank_agreement_raw_projout": (None if not math.isfinite(
                   rank_agreement) else float(rank_agreement)),
               "survives_projection": survives}
        if direction in BORDERLINE_RUNGS:
            row["borderline_rungs_included"] = [float(c) for c in
                                               BORDERLINE_RUNGS[direction]]
        table[direction] = row
    if not table:
        return {name: _none_entry("no steered A-prime rows with projout scores")}
    entry = _entry(value=table, split_side=side,
                   note="zero-anchor rows carry no steering vector, so projout "
                        "is computed on steered rows only and rho_raw is "
                        "recomputed on the SAME rows for comparability; the "
                        "level shift (projout minus raw) is reported because "
                        "the projection also moves the probe's intercept")
    entry["n_clusters"] = int(a["scenario_id"].nunique())
    caveat = _interval_caveat(
        table, entry["n_clusters"],
        flag_keys=("raw_interval_degenerate", "projout_interval_degenerate"))
    if caveat:
        entry["extra"]["interval_caveat"] = caveat
    return {name: entry}


# ============================================== Holm table as its own file

def channels_holm(df: pd.DataFrame, b_out: Optional[dict],
                  B: int = stats.DEFAULT_B, seed: int = 0) -> dict:
    name = "panelB_channels_holm"
    table = None
    if b_out is not None:
        div = b_out.get("panelB_divergence") or {}
        table = (div.get("extra") or {}).get("per_channel_drops")
    if not table:
        return {name: _none_entry("no per-channel Panel B drop table (Panel B "
                                  "absent from the loaded data)")}
    primaries = (EXPRESSION_PRIMARY, INTERNAL_PRIMARY)
    value = {}
    for ch, row in table.items():
        value[ch] = dict(row)
        value[ch]["family"] = ("primary (unadjusted, prereg section 5)"
                               if ch in primaries else "secondary (Holm, D18)")
    # Q-DRIFT compliance control, descriptive, outside the pre-registered family.
    # RAW units: Q-DRIFT has no distress-neutral separation by design, so the
    # per-channel normalization is undefined for it; the meaningful statement is
    # the raw digit-scale change under SUPPRESS on distress scenarios.
    q_drift = None
    b = _panel_b_null(df)
    work_all = b[b["scenario_class"].isin(["distress", "neutral"])
                 & b["condition"].isin(INSTRUCTION_CONDITIONS)] if not b.empty else b
    if not work_all.empty and "q_drift_logit" in work_all.columns \
            and work_all["q_drift_logit"].notna().any():
        work, _side = _pick_side(work_all)

        def _raw_drop(frame: pd.DataFrame) -> float:
            piv = frame.pivot_table(index=["scenario_id", "scenario_class"],
                                    columns="condition", values="q_drift_logit",
                                    aggfunc="mean").reset_index()
            if "NEUTRAL_INSTR" not in piv or "SUPPRESS" not in piv:
                return float("nan")
            d_ = piv[piv["scenario_class"] == "distress"].dropna(
                subset=["NEUTRAL_INSTR", "SUPPRESS"])
            if d_.empty:
                return float("nan")
            return float((d_["NEUTRAL_INSTR"] - d_["SUPPRESS"]).mean())

        res = stats.bca_cluster_bootstrap_frame(
            work, _raw_drop, strata_col="scenario_class", B=B, seed=seed)
        d = res.as_dict()
        dist = work[work["scenario_class"] == "distress"]
        q_drift = {"value": d["value"], "ci_low": d["ci_low"],
                   "ci_high": d["ci_high"],
                   "unit": "raw Q-DRIFT digits (0-9); positive = SUPPRESS lowers",
                   "mean_neutral_instr": float(
                       dist.loc[dist["condition"] == "NEUTRAL_INSTR",
                                "q_drift_logit"].mean()),
                   "mean_suppress": float(
                       dist.loc[dist["condition"] == "SUPPRESS",
                                "q_drift_logit"].mean()),
                   "note": "compliance control, descriptive; NOT in the "
                           "pre-registered Holm family (D18); raw units because "
                           "Q-DRIFT has no class separation to normalize by"}
    div = b_out.get("panelB_divergence") or {}
    entry = _entry(value=value,
                   split_side=(div.get("extra") or {}).get("split_side"),
                   source="values identical to panelB_divergence."
                          "extra.per_channel_drops (same bootstrap run)",
                   unit="drop as fraction of natural separation (x100 for "
                        "percent)")
    if q_drift is not None:
        entry["extra"]["q_drift_descriptive"] = q_drift
    entry["n_clusters"] = div.get("n_clusters")
    return {name: entry}


# ============================================== orchestration

def compute(df: pd.DataFrame, b_out: Optional[dict], thresholds: Optional[dict],
            aprime: bool, B: int = stats.DEFAULT_B, seed: int = 0) -> Dict[str, dict]:
    """All amendment-3 entries, keyed by results name. countermeasure_table
    REPLACES the D8 entry from panels.panel_b (superseding definition stamped;
    the D8 numbers ride along in extra.superseded_d8)."""
    out: Dict[str, dict] = {}
    d8 = superseded_d8_summary((b_out or {}).get("countermeasure_table"))
    out["countermeasure_table"] = countermeasure_recalibrated(
        df, thresholds, superseded_d8=d8, B=B, seed=seed)
    out.update(internal_did(df, B=B, seed=seed))
    out.update(internal_nofall(df, B=B, seed=seed))
    out.update(selfother(df, B=B, seed=seed))
    out.update(internal_specificity(df, aprime, B=B, seed=seed))
    out.update(projout_check(df, aprime, B=B, seed=seed))
    out.update(channels_holm(df, b_out, B=B, seed=seed))
    return out

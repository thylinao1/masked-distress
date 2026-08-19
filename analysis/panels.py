"""Panel analyses: V (validity + anchors), B (masking), LADDER (dose map), A (dose-response).

Every function takes the flat DataFrame from analysis/loading.py and returns a dict of
results entries shaped {"value", "ci_low", "ci_high", "n_clusters", "extra": {...}}
keyed by schema.RESULTS_NAMES, plus internal keys prefixed "_" (thresholds, caches)
that results_io never emits. All numbered design decisions live in analysis/DECISIONS.md.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis import stats
from analysis.stats import BootResult

EXPRESSION_PRIMARY = "q_self_logit"
INTERNAL_PRIMARY = "probe_prompt_final"
CHANNELS_EXPRESSION = ["q_self_logit", "q_val_logit", "exit_logit_diff", "sentiment_neg"]
CHANNELS_INTERNAL = ["probe_prompt_final", "probe_teacher_forced",
                     "probe_response_mean", "sae_prompt_final"]
SELF_DIRECTIONS = ("D-CTX", "D-PV")
PLACEBO_DIRECTIONS = ("R1", "R2", "R3", "SEM")
TOST_BOUND = 0.25          # natural-separation units (prereg section 4)
FPR_DOSE_MIN = 1.0         # placebo cells at |dose| >= 1 sigma (prereg section 4)
_EPS = 1e-9


def _entry(res: Optional[BootResult] = None, value=None, note: Optional[str] = None,
           **extra) -> dict:
    out = {"value": value, "ci_low": None, "ci_high": None, "n_clusters": None,
           "extra": dict(extra)}
    if res is not None:
        d = res.as_dict()
        out.update({k: d[k] for k in ("value", "ci_low", "ci_high", "n_clusters")})
        out["extra"]["B"] = d["B"]
        if d["n_boot_failed"]:
            out["extra"]["n_boot_failed"] = d["n_boot_failed"]
    if note:
        out["extra"]["note"] = note
    return out


def _none_entry(note: str) -> dict:
    return _entry(value=None, note=note)


def _pick_side(sub: pd.DataFrame, prefer: str = "confirmation") -> Tuple[pd.DataFrame, str]:
    """Prefer the confirmation split; fall back to discovery with a label (D4, D10)."""
    if sub.empty:
        return sub, "none"
    if (sub["split_side"] == prefer).any():
        return sub[sub["split_side"] == prefer], prefer
    other = "discovery" if prefer == "confirmation" else "confirmation"
    if (sub["split_side"] == other).any():
        return sub[sub["split_side"] == other], other
    return sub, "unknown"


def _scen_means(frame: pd.DataFrame, channel: str) -> pd.DataFrame:
    g = frame.groupby("scenario_id").agg(
        value=(channel, "mean"), scenario_class=("scenario_class", "first"))
    return g.reset_index()


# ==================================================================== Panel V

def _natural_source(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """NEUTRAL_INSTR cells from panels V and B (D2); fallback NONE, labelled."""
    base = df[df["panel"].isin(["V", "B"]) & (df["direction"] == "NULL")
              & df["scenario_class"].isin(["distress", "neutral"])]
    src = base[base["condition"] == "NEUTRAL_INSTR"]
    which = "NEUTRAL_INSTR"
    if src.empty:
        src = base[base["condition"] == "NONE"]
        which = "NONE (fallback: no NEUTRAL_INSTR cells loaded; see DECISIONS.md D2)"
    src, side = _pick_side(src)
    return src, which, side


def natural_separation(frame: pd.DataFrame, channel: str) -> float:
    m = _scen_means(frame.dropna(subset=[channel]), channel)
    d = m.loc[m["scenario_class"] == "distress", "value"]
    n = m.loc[m["scenario_class"] == "neutral", "value"]
    if d.empty or n.empty:
        return float("nan")
    return float(d.mean() - n.mean())


def panel_v(df: pd.DataFrame, B: int = stats.DEFAULT_B, seed: int = 0) -> dict:
    out: Dict[str, dict] = {}
    v = df[(df["panel"] == "V") & (df["direction"] == "NULL")
           & df["scenario_class"].isin(["distress", "neutral"])] if not df.empty else df

    # ---- validity AUCs, held-out (D4) ----
    if v is not None and not v.empty:
        held, side = _pick_side(v)
        for inst, name in ((INTERNAL_PRIMARY, "validity_auc_probe_heldout"),
                           ("sae_prompt_final", "validity_auc_sae_heldout")):
            sub = held.dropna(subset=[inst]).copy()
            if sub.empty or sub["scenario_class"].nunique() < 2:
                out[name] = _none_entry("no two-class Panel V rows for this instrument")
                continue
            sub["_label"] = (sub["scenario_class"] == "distress").astype(float)
            res = stats.auc_cluster(sub, inst, "_label", B=B, seed=seed)
            out[name] = _entry(res, split_side=side, instrument=inst,
                               gate="AUC >= 0.80" if name.endswith("probe_heldout") else None)
    else:
        out["validity_auc_probe_heldout"] = _none_entry("no Panel V rows loaded")
        out["validity_auc_sae_heldout"] = _none_entry("no Panel V rows loaded")

    # ---- dose unit: SD of internal readout scenario means, DISCOVERY V (D3) ----
    disc = v[v["split_side"] == "discovery"] if v is not None and not v.empty else pd.DataFrame()
    dose_note = None
    dose_side = "discovery"
    if disc.empty and v is not None and not v.empty:
        disc, dose_note = v, "no discovery-side V rows; dose unit from all V rows (labelled)"
        dose_side = "all"  # the realized side, never the hardcoded claim
    if not disc.empty and disc[INTERNAL_PRIMARY].notna().any():
        m = _scen_means(disc.dropna(subset=[INTERNAL_PRIMARY]), INTERNAL_PRIMARY)
        vals = m["value"].to_numpy()
        res = stats.bca_cluster_values(vals, stat_fn=lambda x: x.std(ddof=1),
                                       B=B, seed=seed)
        out["natural_separation_sd"] = _entry(res, note=dose_note, split_side=dose_side)
        dose_unit = float(res.value)
    else:
        out["natural_separation_sd"] = _none_entry("no Panel V internal-readout rows loaded")
        dose_unit = float("nan")

    # ---- per-channel natural separations + ranges (D2) ----
    nat_src, nat_cond, nat_side = _natural_source(df) if not df.empty else (pd.DataFrame(), "", "")
    seps: Dict[str, float] = {}
    ranges: Dict[str, dict] = {}
    if not nat_src.empty:
        for ch in CHANNELS_EXPRESSION + CHANNELS_INTERNAL:
            if nat_src[ch].notna().any():
                seps[ch] = natural_separation(nat_src, ch)
                m = _scen_means(nat_src.dropna(subset=[ch]), ch)
                ranges[ch] = {
                    "neutral": float(m.loc[m["scenario_class"] == "neutral", "value"].mean()),
                    "distress": float(m.loc[m["scenario_class"] == "distress", "value"].mean()),
                }
        if "natural_separation_sd" in out and out["natural_separation_sd"]["value"] is not None:
            out["natural_separation_sd"]["extra"]["per_channel_natural_separation"] = {
                k: (None if not math.isfinite(v_) else v_) for k, v_ in seps.items()}
            out["natural_separation_sd"]["extra"]["natural_source"] = \
                f"{nat_cond}, {nat_side} split"

    # ---- pre-committed theta + countermeasure thresholds, DISCOVERY only (D8, D12) ----
    thresholds: Optional[dict] = None
    disc_neutral = df[(df["panel"] == "V") & (df["direction"] == "NULL")
                      & (df["scenario_class"] == "neutral")
                      & (df["split_side"] == "discovery")] if not df.empty else pd.DataFrame()
    disc_distress = df[(df["panel"] == "V") & (df["direction"] == "NULL")
                       & (df["scenario_class"] == "distress")
                       & (df["split_side"] == "discovery")] if not df.empty else pd.DataFrame()
    # Guard on SCENARIO count, not cell count: one scenario x 3 seeds passes a cell
    # count but its scenario-mean SD (ddof=1) is NaN, making theta NaN, and every
    # downstream `q > NaN` comparison silently reports FPR = 0.0. Partial rsyncs of
    # Panel V are the known failure mode; refuse to set thresholds instead.
    n_theta_scen = (int(disc_neutral["scenario_id"].nunique())
                    if not disc_neutral.empty else 0)
    if n_theta_scen >= 2 and disc_neutral[EXPRESSION_PRIMARY].notna().any():
        m = _scen_means(disc_neutral.dropna(subset=[EXPRESSION_PRIMARY]), EXPRESSION_PRIMARY)
        theta_expr = float(m["value"].mean() + 2.0 * m["value"].std(ddof=1))
        if math.isfinite(theta_expr):
            cells_q = disc_neutral[EXPRESSION_PRIMARY].dropna().to_numpy()
            fpr_disc = float(np.mean(cells_q > theta_expr))
            cells_int = disc_neutral[INTERNAL_PRIMARY].dropna().to_numpy()
            theta_int = (float(np.quantile(cells_int, 1.0 - fpr_disc))
                         if len(cells_int) else None)
            if theta_int is not None and not math.isfinite(theta_int):
                theta_int = None
            elev_mid = None
            if len(disc_distress) and disc_distress[INTERNAL_PRIMARY].notna().any() \
                    and len(cells_int):
                md = _scen_means(disc_distress.dropna(subset=[INTERNAL_PRIMARY]),
                                 INTERNAL_PRIMARY)["value"].mean()
                mn = _scen_means(disc_neutral.dropna(subset=[INTERNAL_PRIMARY]),
                                 INTERNAL_PRIMARY)["value"].mean()
                elev_mid = float((md + mn) / 2.0)
            thresholds = {"theta_expr": theta_expr, "fpr_expr_discovery": fpr_disc,
                          "theta_int": theta_int, "elevation_midpoint": elev_mid}

    out["_dose_unit"] = dose_unit
    out["_natural_separations"] = seps
    out["_natural_ranges"] = ranges
    out["_natural_source"] = {"condition": nat_cond, "side": nat_side}
    out["_thresholds"] = thresholds
    return out


# ==================================================================== Panel B

def _drops_from_frame(frame: pd.DataFrame, channel: str) -> pd.Series:
    """Per-scenario drop = (NEUTRAL_INSTR - SUPPRESS) / natural_separation (D1, D2).
    Natural separation is re-estimated from the (possibly resampled) frame (D7)."""
    piv = frame.pivot_table(index=["scenario_id", "scenario_class"], columns="condition",
                            values=channel, aggfunc="mean").reset_index()
    if "NEUTRAL_INSTR" not in piv or "SUPPRESS" not in piv:
        return pd.Series(dtype=float)
    ni = piv["NEUTRAL_INSTR"]
    nat = (ni[piv["scenario_class"] == "distress"].mean()
           - ni[piv["scenario_class"] == "neutral"].mean())
    if not math.isfinite(nat) or abs(nat) < _EPS:
        return pd.Series(dtype=float)
    d = piv[piv["scenario_class"] == "distress"].dropna(subset=["NEUTRAL_INSTR", "SUPPRESS"])
    return (d["NEUTRAL_INSTR"] - d["SUPPRESS"]) / nat


def _mean_drop_stat(channel: str):
    return lambda frame: float(_drops_from_frame(frame, channel).mean())


def _divergence_stat(frame: pd.DataFrame) -> float:
    return (float(_drops_from_frame(frame, EXPRESSION_PRIMARY).mean())
            - float(_drops_from_frame(frame, INTERNAL_PRIMARY).mean()))


def _elevation_stat(channel: str, condition: str):
    """Distress elevation over the NEUTRAL_INSTR neutral baseline, in natural-separation
    units, under `condition` (figure 1 bars)."""
    def stat(frame: pd.DataFrame) -> float:
        piv = frame.pivot_table(index=["scenario_id", "scenario_class"], columns="condition",
                                values=channel, aggfunc="mean").reset_index()
        if "NEUTRAL_INSTR" not in piv or condition not in piv:
            return float("nan")
        ni = piv["NEUTRAL_INSTR"]
        nat = (ni[piv["scenario_class"] == "distress"].mean()
               - ni[piv["scenario_class"] == "neutral"].mean())
        if not math.isfinite(nat) or abs(nat) < _EPS:
            return float("nan")
        neutral_base = ni[piv["scenario_class"] == "neutral"].mean()
        dist = piv.loc[piv["scenario_class"] == "distress", condition].mean()
        return float((dist - neutral_base) / nat)
    return stat


def panel_b(df: pd.DataFrame, thresholds: Optional[dict],
            B: int = stats.DEFAULT_B, seed: int = 0) -> dict:
    out: Dict[str, dict] = {}
    empty = {"panelB_expression_drop_pct": _none_entry("no Panel B rows loaded"),
             "panelB_internal_drop_pct": _none_entry("no Panel B rows loaded"),
             "panelB_divergence": _none_entry("no Panel B rows loaded"),
             "panelB_tost_internal_equivalence": _none_entry("no Panel B rows loaded"),
             "countermeasure_table": _none_entry("no Panel B rows loaded"),
             "_cache_panelB": None}
    if df.empty or not (df["panel"] == "B").any():
        return empty

    b = df[(df["panel"] == "B") & (df["direction"] == "NULL")]
    work_all = b[b["scenario_class"].isin(["distress", "neutral"])
                 & b["condition"].isin(["NEUTRAL_INSTR", "SUPPRESS"])]
    work, side = _pick_side(work_all)
    if work.empty or work[work["scenario_class"] == "distress"].empty:
        return empty
    kw = dict(strata_col="scenario_class", B=B, seed=seed)

    # ---- primary drops + divergence (D1, D6, D7) ----
    res_e = stats.bca_cluster_bootstrap_frame(work, _mean_drop_stat(EXPRESSION_PRIMARY), **kw)
    res_i = stats.bca_cluster_bootstrap_frame(work, _mean_drop_stat(INTERNAL_PRIMARY), **kw)
    res_d = stats.bca_cluster_bootstrap_frame(work, _divergence_stat, **kw)
    n_dist = work.loc[work["scenario_class"] == "distress", "scenario_id"].nunique()

    def _pct(res: BootResult) -> dict:
        e = _entry(res, split_side=side, unit="percent of natural separation")
        for k in ("value", "ci_low", "ci_high"):
            if e[k] is not None:
                e[k] = float(e[k]) * 100.0
        e["n_clusters"] = int(n_dist)
        e["extra"]["n_clusters_resampled_total"] = res.n_clusters
        return e

    out["panelB_expression_drop_pct"] = _pct(res_e)
    out["panelB_internal_drop_pct"] = _pct(res_i)
    div = _entry(res_d, split_side=side, unit="natural-separation units (x100 for percent)")
    div["n_clusters"] = int(n_dist)
    div["extra"]["n_clusters_resampled_total"] = res_d.n_clusters
    out["panelB_divergence"] = div

    # ---- TOST on per-scenario internal drops (D5) ----
    drops_int = _drops_from_frame(work, INTERNAL_PRIMARY).to_numpy()
    tost = stats.tost_equivalence(drops_int, bound=TOST_BOUND)
    out["panelB_tost_internal_equivalence"] = _entry(
        value=tost.get("p_tost"), split_side=side, **{k: v for k, v in tost.items()
                                                      if k != "p_tost"})
    out["panelB_tost_internal_equivalence"]["n_clusters"] = tost["n_clusters"]

    # ---- per-channel drop table + Holm (D18) ----
    table: Dict[str, dict] = {}
    pvals: Dict[str, float] = {}
    for ch in CHANNELS_EXPRESSION + CHANNELS_INTERNAL:
        if not work[ch].notna().any():
            continue
        res = stats.bca_cluster_bootstrap_frame(work, _mean_drop_stat(ch),
                                                keep_boots=True, **kw)
        p = stats.boot_p_two_sided(res.boots) if res.boots is not None else float("nan")
        table[ch] = {**{k: res.as_dict()[k] for k in ("value", "ci_low", "ci_high")},
                     "p_boot": None if not math.isfinite(p) else p}
        if ch not in (EXPRESSION_PRIMARY, INTERNAL_PRIMARY):
            pvals[ch] = p
    adj = stats.holm(pvals)
    for ch, a in adj.items():
        if ch in table:
            table[ch]["p_holm"] = None if not math.isfinite(a) else a
    out["panelB_divergence"]["extra"]["per_channel_drops"] = table

    # ---- countermeasure table (D8) ----
    cases = b[(b["scenario_class"] == "distress") & (b["condition"] == "SUPPRESS")]
    cases, case_side = _pick_side(cases)
    if thresholds is None or thresholds.get("theta_int") is None or cases.empty:
        out["countermeasure_table"] = _none_entry(
            "missing discovery-split Panel V thresholds or SUPPRESS distress cells")
    else:
        th_e = thresholds["theta_expr"]
        th_i = thresholds["theta_int"]
        elev = thresholds.get("elevation_midpoint")
        rows = []
        sets: List[Tuple[str, pd.DataFrame]] = [("all_suppressed_distress", cases)]
        if elev is not None:
            sets.append(("masked_but_elevated",
                         cases[cases[INTERNAL_PRIMARY] >= elev]))
        for set_name, sub in sets:
            sub = sub.dropna(subset=[EXPRESSION_PRIMARY, INTERNAL_PRIMARY])
            if sub.empty:
                rows.append({"case_set": set_name, "n_cells": 0, "note": "no cases"})
                continue
            row = {"case_set": set_name, "n_cells": int(len(sub)),
                   "n_scenarios": int(sub["scenario_id"].nunique())}
            for trig, col, th in (("expression", EXPRESSION_PRIMARY, th_e),
                                  ("internal", INTERNAL_PRIMARY, th_i)):
                res = stats.bca_cluster_bootstrap_rows(
                    sub, [col], lambda x, t=th: float(np.mean(x <= t)), B=B, seed=seed)
                d = res.as_dict()
                row[f"miss_rate_{trig}"] = d["value"]
                row[f"miss_rate_{trig}_ci"] = [d["ci_low"], d["ci_high"]]
            rows.append(row)
        out["countermeasure_table"] = _entry(
            value={"theta_expr": th_e, "theta_int": th_i,
                   "matched_fpr_discovery": thresholds["fpr_expr_discovery"],
                   "elevation_midpoint": elev, "rows": rows},
            split_side=case_side,
            note=f"case split side: {case_side}; thresholds from discovery split (D8)")
        out["countermeasure_table"]["n_clusters"] = int(cases["scenario_id"].nunique())

    # ---- figure 1 cache: four bars + divergence (D1 bars in nat-sep units) ----
    B_fig = min(B, 2000)
    bars: Dict[str, dict] = {}
    for label, ch in (("expression", EXPRESSION_PRIMARY), ("internal", INTERNAL_PRIMARY)):
        bars[label] = {}
        for cond_label, cond in (("off", "NEUTRAL_INSTR"), ("on", "SUPPRESS")):
            r = stats.bca_cluster_bootstrap_frame(
                work, _elevation_stat(ch, cond), strata_col="scenario_class",
                B=B_fig, seed=seed)
            d = r.as_dict()
            bars[label][cond_label] = {"height": d["value"], "lo": d["ci_low"],
                                       "hi": d["ci_high"]}
    out["_cache_panelB"] = {
        "bars": bars,
        "divergence": {"value": div["value"], "lo": div["ci_low"], "hi": div["ci_high"]},
        "n_scenarios": int(n_dist), "split_side": side,
        "channels": table,
    }
    return out


# ==================================================================== LADDER

def _load_capability(capability: Union[None, str, Path, dict]) -> Optional[dict]:
    if capability is None:
        return None
    if isinstance(capability, (str, Path)):
        p = Path(capability)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    return capability


def ladder(df: pd.DataFrame, dose_unit: float,
           capability: Union[None, str, Path, dict] = None) -> dict:
    """Coefficient -> SD-units dose map + capability-valid range (D11, D19)."""
    out: Dict[str, dict] = {}
    cap = _load_capability(capability)
    lad = df[df["panel"] == "LADDER"] if not df.empty else df
    dose_map: Dict[str, dict] = {}
    if lad is not None and not lad.empty and math.isfinite(dose_unit) and dose_unit > 0:
        zero = lad[(lad["coefficient"] == 0.0)]
        for d in ("D-CTX", "D-PV", "R1"):
            sub = pd.concat([lad[lad["direction"] == d], zero], ignore_index=True)
            sub = sub.dropna(subset=[INTERNAL_PRIMARY])
            g = sub.groupby("coefficient")[INTERNAL_PRIMARY].mean()
            if len(g) < 2:
                dose_map[d] = {"note": "fewer than 2 coefficient levels"}
                continue
            slope = float(np.polyfit(g.index.to_numpy(dtype=float),
                                     g.to_numpy(dtype=float), 1)[0])
            dose_map[d] = {
                "slope_readout_per_coef": slope,
                "sd_per_coef": slope / dose_unit,
                "coef_per_sd": (dose_unit / slope) if abs(slope) > _EPS else None,
                "n_coef_levels": int(len(g)),
                "coefficients": [float(c) for c in g.index],
            }

    valid: Optional[dict] = None
    per_dir_range: Dict[str, dict] = {}
    if cap:
        bounds = []
        for d in SELF_DIRECTIONS:
            spc = dose_map.get(d, {}).get("sd_per_coef")
            samples = cap.get(d, [])
            if not spc or not samples:
                continue
            scored = sorted(
                ({"dose": abs(float(s["coefficient"])) * abs(spc),
                  "ok": (float(s["ppl_ratio"]) < 1.10
                         and float(s["mmlu_drop_pp"]) < 5.0)}
                 for s in samples), key=lambda s: s["dose"])
            m = 0.0
            for s in scored:
                if not s["ok"]:
                    break
                m = s["dose"]
            per_dir_range[d] = {"lo_sd": -m, "hi_sd": m, "n_samples": len(scored)}
            bounds.append(m)
        if bounds:
            m_all = min(bounds)
            valid = {"lo_sd": -m_all, "hi_sd": m_all}

    if valid is not None:
        out["capability_valid_range"] = _entry(
            value=valid, per_direction=per_dir_range,
            criteria="ppl inflation < 10% AND MMLU-lite drop < 5pp (prereg section 4)")
    elif cap is not None:
        out["capability_valid_range"] = _none_entry(
            "capability data present but no dose map (need LADDER rows + dose unit)")
    else:
        out["capability_valid_range"] = _none_entry(
            "capability logprob job output not supplied; range determination pending (D11)")
    if dose_map:
        out["capability_valid_range"]["extra"]["dose_map"] = dose_map
        out["capability_valid_range"]["extra"]["dose_map_readout_caveat"] = (
            "the LADDER rows were produced with --no-probe (battery/cells_ladder.json), so their "
            "probe_score is the PLACEHOLDER_MEAN readout (residual mean at Lr; src/runner.py) and "
            "this map is that placeholder's slope over the I-PROBE SD, not a probe-unit map. For "
            "SD-unit statements use results/capability_valid_range_realprobe.json ("
            "PREREGISTRATION.md section 9, 2026-08-17); the coefficient-unit capability grid is real")
    out["_dose_map"] = dose_map
    out["_valid_range"] = valid
    return out


# ==================================================================== Panel A

def _direction_rows(a: pd.DataFrame, direction: str) -> pd.DataFrame:
    """A direction's curve pools all dose-0 rows regardless of label (D9)."""
    return a[(a["direction"] == direction) | (np.isclose(a["dose_sd"], 0.0))]


def _ratio_stat(doses: Sequence[float]):
    def stat(frame: pd.DataFrame) -> float:
        base = frame.loc[np.isclose(frame["dose_sd"], 0.0), EXPRESSION_PRIMARY].mean()
        eff_o, eff_s = [], []
        for d in doses:
            sel = np.isclose(frame["dose_sd"], d)
            o = frame.loc[sel & (frame["direction"] == "OTHER"), EXPRESSION_PRIMARY].mean()
            s = frame.loc[sel & frame["direction"].isin(SELF_DIRECTIONS),
                          EXPRESSION_PRIMARY].mean()
            if math.isfinite(o):
                eff_o.append(o - base)
            if math.isfinite(s):
                eff_s.append(s - base)
        if not eff_o or not eff_s:
            return float("nan")
        den = float(np.mean(eff_s))
        if abs(den) < _EPS:
            return float("nan")
        return float(np.mean(eff_o)) / den
    return stat


def panel_a(df: pd.DataFrame, theta: Optional[float],
            valid_range: Optional[dict] = None,
            natural_ranges: Optional[dict] = None,
            B: int = stats.DEFAULT_B, seed: int = 0) -> dict:
    out: Dict[str, dict] = {}
    names = ("panelA_spearman_dctx", "panelA_spearman_dpv",
             "panelA_fpr_placebo_at_threshold", "panelA_other_vs_self_ratio")
    if df.empty or not (df["panel"] == "A").any():
        for n in names:
            out[n] = _none_entry("no Panel A rows loaded")
        out["_cache_panelA"] = None
        return out

    a_all = df[(df["panel"] == "A") & (df["scenario_class"] == "neutral")]
    a, side = _pick_side(a_all)  # D10
    hi = valid_range["hi_sd"] if valid_range else None

    # ---- Spearman per direction (primary; D10, D11) ----
    for direction, name in zip(SELF_DIRECTIONS, ("panelA_spearman_dctx",
                                                 "panelA_spearman_dpv")):
        sub = _direction_rows(a, direction).dropna(subset=["dose_sd", EXPRESSION_PRIMARY])
        rng_note = "all doses (no capability range supplied)"
        if hi is not None:
            sub = sub[np.abs(sub["dose_sd"]) <= hi + _EPS]
            rng_note = f"|dose| <= {hi:.3g} sigma (capability-valid, D11)"
        if sub.empty or sub["dose_sd"].nunique() < 3:
            out[name] = _none_entry(f"insufficient {direction} dose levels")
            continue
        res = stats.spearman_cluster(sub, "dose_sd", EXPRESSION_PRIMARY, B=B, seed=seed)
        full = _direction_rows(a_all, direction).dropna(
            subset=["dose_sd", EXPRESSION_PRIMARY])
        rho_all = stats.spearman_stat(full["dose_sd"].to_numpy(),
                                      full[EXPRESSION_PRIMARY].to_numpy())
        out[name] = _entry(res, split_side=side, dose_range=rng_note,
                           rho_all_scenarios=None if not math.isfinite(rho_all) else rho_all)

    # ---- FPR at the pre-committed threshold (D12) ----
    placebo = a[a["direction"].isin(PLACEBO_DIRECTIONS)
                & (np.abs(a["dose_sd"]) >= FPR_DOSE_MIN - _EPS)]
    placebo = placebo.dropna(subset=[EXPRESSION_PRIMARY])
    # NaN theta is not None: `q > NaN` is elementwise False, which would emit a
    # confidently wrong FPR = 0.0 with a real CI. Refuse non-finite theta outright.
    if theta is None or not math.isfinite(theta) or placebo.empty:
        out["panelA_fpr_placebo_at_threshold"] = _none_entry(
            "missing or non-finite theta (discovery Panel V) or no placebo cells")
    else:
        res = stats.bca_cluster_bootstrap_rows(
            placebo, [EXPRESSION_PRIMARY],
            lambda q: float(np.mean(q > theta)), B=B, seed=seed)
        out["panelA_fpr_placebo_at_threshold"] = _entry(
            res, split_side=side, theta=theta, n_placebo_cells=int(len(placebo)),
            directions=list(PLACEBO_DIRECTIONS), dose_min_sd=FPR_DOSE_MIN)

    # ---- OTHER vs self ratio, H3 (D13) ----
    other_doses = sorted(d for d in a.loc[a["direction"] == "OTHER", "dose_sd"].unique()
                         if d > _EPS)
    if not other_doses:
        out["panelA_other_vs_self_ratio"] = _none_entry("no OTHER-direction cells")
    else:
        frame = a[np.isclose(a["dose_sd"], 0.0)
                  | (a["direction"].isin(list(SELF_DIRECTIONS) + ["OTHER"])
                     & a["dose_sd"].isin(other_doses))]
        frame = frame.dropna(subset=[EXPRESSION_PRIMARY])
        res = stats.bca_cluster_bootstrap_frame(frame, _ratio_stat(other_doses),
                                                B=B, seed=seed)
        out["panelA_other_vs_self_ratio"] = _entry(
            res, split_side=side, matched_doses_sd=[float(d) for d in other_doses],
            threshold="ratio < 0.5 (H3)")

    # ---- ridge-probe dose-decoder baseline (hook; not a RESULTS_NAME) ----
    out["_ridge_decoder"] = ridge_dose_decoder(a, B=min(B, 2000), seed=seed)
    for name in ("panelA_spearman_dctx", "panelA_spearman_dpv"):
        if name in out and out[name].get("value") is not None:
            out[name]["extra"]["ridge_decoder_baseline"] = out["_ridge_decoder"]

    # ---- figure 2 cache ----
    channels = [c for c in CHANNELS_EXPRESSION if a[c].notna().any()]
    B_fig = min(B, 2000)
    cache: Dict[str, dict] = {"directions": {}, "placebo": {}, "channels": channels,
                              "natural_range": natural_ranges or {},
                              "capability_range": valid_range, "split_side": side}
    for direction in list(SELF_DIRECTIONS) + ["OTHER"]:
        sub = _direction_rows(a, direction) if direction in SELF_DIRECTIONS \
            else a[(a["direction"] == direction) | (np.isclose(a["dose_sd"], 0.0))]
        if sub[sub["direction"] == direction].empty:
            continue
        doses = sorted(sub["dose_sd"].unique())
        entry: Dict[str, dict] = {"doses": [float(d) for d in doses], "channels": {}}
        for ch in channels:
            mean, lo_, hi_ = [], [], []
            for d in doses:
                cells = sub[np.isclose(sub["dose_sd"], d)].dropna(subset=[ch])
                m = _scen_means(cells, ch)["value"].to_numpy()
                r = stats.bca_cluster_values(m, B=B_fig, seed=seed)
                dd = r.as_dict()
                mean.append(dd["value"]); lo_.append(dd["ci_low"]); hi_.append(dd["ci_high"])
            entry["channels"][ch] = {"mean": mean, "lo": lo_, "hi": hi_}
        cache["directions"][direction] = entry
    pla = a[a["direction"].isin(PLACEBO_DIRECTIONS) | (np.isclose(a["dose_sd"], 0.0))]
    if not pla.empty:
        doses = sorted(pla["dose_sd"].unique())
        entry = {"doses": [float(d) for d in doses], "channels": {}}
        for ch in channels:
            mean, lo_, hi_ = [], [], []
            for d in doses:
                cells = pla[np.isclose(pla["dose_sd"], d)].dropna(subset=[ch])
                m = _scen_means(cells, ch)["value"].to_numpy()
                r = stats.bca_cluster_values(m, B=B_fig, seed=seed)
                dd = r.as_dict()
                mean.append(dd["value"]); lo_.append(dd["ci_low"]); hi_.append(dd["ci_high"])
            entry["channels"][ch] = {"mean": mean, "lo": lo_, "hi": hi_}
        cache["placebo"] = entry
    out["_cache_panelA"] = cache
    return out


def ridge_dose_decoder(a: pd.DataFrame, B: int = 2000, seed: int = 0) -> dict:
    """Privileged-access baseline: ridge regression from internal channels to dose,
    scenario-grouped out-of-fold predictions, Spearman(prediction, dose)."""
    feats = [c for c in ("probe_prompt_final", "probe_teacher_forced",
                         "probe_response_mean", "sae_prompt_final",
                         "sae_teacher_forced", "sae_response_mean")
             if c in a.columns and a[c].notna().any()]
    sub = a[a["direction"].isin(SELF_DIRECTIONS) | (np.isclose(a["dose_sd"], 0.0))]
    sub = sub.dropna(subset=feats + ["dose_sd"]).copy()
    n_groups = sub["scenario_id"].nunique()
    if len(sub) < 12 or n_groups < 3 or not feats:
        return {"value": None, "note": "insufficient rows/scenarios for the decoder"}
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = sub[feats].to_numpy(dtype=float)
    y = sub["dose_sd"].to_numpy(dtype=float)
    groups = sub["scenario_id"].to_numpy()
    pred = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=min(5, n_groups))
    for tr, te in gkf.split(X, y, groups):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
    sub["_pred"] = pred
    res = stats.spearman_cluster(sub, "_pred", "dose_sd", B=B, seed=seed)
    d = res.as_dict()
    return {"value": d["value"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
            "n_clusters": d["n_clusters"], "features": feats,
            "definition": "GroupKFold-by-scenario OOF ridge decode of dose from internal "
                          "channels; Spearman(prediction, dose)"}


# ==================================================================== Panel A-PRIME
# Prereg §9 amendment 2 (2026-08-16, pre-A-prime-data): the SD-unit grid is
# unreachable (capability-valid range ±0.007 SD on the placeholder-readout ladder map used
# at the time, ±0.075 SD on the real-probe map of capability_valid_range_realprobe,
# the grid is unreachable either way, and the A3 branch fires for the original design). A-prime uses coefficient-unit grids inside the realized
# per-direction capability-valid windows; ALL A-prime analysis keys on the
# coefficient column (dose_sd is bookkeeping only).

APRIME_GRID = {"D-CTX": (500.0, 1000.0, 2000.0),
               "D-PV": (250.0, 500.0, 1000.0)}          # nonzero rungs (0 = NULL anchor)
APRIME_STRICT_MAX_COEF = {"D-CTX": 2000.0, "D-PV": 500.0}  # strictly-valid windows
APRIME_BORDERLINE = {"D-PV": 1000.0}                    # included, labelled, secondary only
APRIME_CONTROL_COEFS = (500.0, 1000.0)
APRIME_RATIO_COEF = 500.0                               # A'3 matched coefficient
# Steered rows with |dose_sd| below this ceiling carry coefficient-unit bookkeeping
# (max realized |dose_sd| in the A-prime grid is D-PV@1000 = 0.075 SD); the original
# SD-unit design's smallest nonzero |dose_sd| was 0.5.
_APRIME_DOSE_SD_CEILING = 0.25
_APRIME_MIN_COEF = 250.0        # smallest nonzero A-prime rung (D-PV at 250)
_SD_UNIT_DOSE_SD_FLOOR = 0.5    # smallest nonzero |dose_sd| of the SD-unit design


def is_panel_aprime(df: pd.DataFrame) -> bool:
    """True when the loaded Panel A rows carry the amendment-2 coefficient-unit grid.

    A MIXED load is refused (ValueError), never routed: if coefficient-unit
    A-prime rows (|dose_sd| < 0.25 at |coefficient| >= 250) and SD-unit rows
    (|dose_sd| >= 0.5) are concatenated (an rsync or glob mistake),
    routing on the max |dose_sd| would send the whole load down the legacy
    SD-unit path, where A-prime control rows carrying the dose_sd = 0.0
    placeholder (steered R2/R3/SEM/OTHER at coefficient 500/1000) satisfy
    isclose(dose_sd, 0) and silently pollute the D9 dose-0 baseline pool.
    """
    if df.empty or not (df["panel"] == "A").any():
        return False
    a = df[df["panel"] == "A"]
    steered = a[a["coefficient"].abs() > _EPS]
    if steered.empty:
        return False
    dose_abs = steered["dose_sd"].abs()
    coef_abs = steered["coefficient"].abs()
    has_aprime_rows = bool(((dose_abs < _APRIME_DOSE_SD_CEILING)
                            & (coef_abs >= _APRIME_MIN_COEF)).any())
    has_sd_unit_rows = bool((dose_abs >= _SD_UNIT_DOSE_SD_FLOOR).any())
    if has_aprime_rows and has_sd_unit_rows:
        n_ap = int(((dose_abs < _APRIME_DOSE_SD_CEILING)
                    & (coef_abs >= _APRIME_MIN_COEF)).sum())
        n_sd = int((dose_abs >= _SD_UNIT_DOSE_SD_FLOOR).sum())
        raise ValueError(
            f"Mixed Panel A load refused: {n_ap} coefficient-unit A-prime row(s) "
            f"(|dose_sd| < {_APRIME_DOSE_SD_CEILING} at |coefficient| >= "
            f"{_APRIME_MIN_COEF:g}) and {n_sd} SD-unit row(s) (|dose_sd| >= "
            f"{_SD_UNIT_DOSE_SD_FLOOR}) in one load. A-prime control rows with "
            f"placeholder dose_sd = 0.0 would pollute the legacy D9 dose-0 "
            f"baseline pool. Load one Panel A design per analysis run.")
    return float(dose_abs.max()) < _APRIME_DOSE_SD_CEILING


def _coef_mask(series: pd.Series, coefs: Sequence[float]) -> np.ndarray:
    mask = np.zeros(len(series), dtype=bool)
    for c in coefs:
        mask |= np.isclose(series.to_numpy(dtype=float), float(c))
    return mask


def _aprime_direction_rows(a: pd.DataFrame, direction: str,
                           coefs: Sequence[float]) -> pd.DataFrame:
    """Direction rows at the given nonzero rungs + the pooled zero anchor (D9 analog
    on coefficient: coef-0 rows pool regardless of direction label, NULL included)."""
    at_rungs = (a["direction"] == direction) & _coef_mask(a["coefficient"], coefs)
    return a[at_rungs | np.isclose(a["coefficient"], 0.0)]


def _aprime_shift_stat(directions: Sequence[str], coef: float, channel: str):
    """Shift of `channel` scenario means at `coef` (over the pooled zero anchor),
    divided by that channel's natural separation re-estimated from the frame's
    natural-source rows (_role == 'natural'; D1/D2/D6/D7 machinery on steering)."""
    dirs = tuple(directions)

    def stat(frame: pd.DataFrame) -> float:
        nat = natural_separation(
            frame[frame["_role"] == "natural"].dropna(subset=[channel]), channel)
        if not math.isfinite(nat) or abs(nat) < _EPS:
            return float("nan")
        st = frame[frame["_role"] == "steer"].dropna(subset=[channel])
        zero = st[np.isclose(st["coefficient"], 0.0)]
        top = st[np.isclose(st["coefficient"], coef) & st["direction"].isin(dirs)]
        if zero.empty or top.empty:
            return float("nan")
        base = _scen_means(zero, channel)["value"].mean()
        val = _scen_means(top, channel)["value"].mean()
        return float((val - base) / nat)
    return stat


def _aprime_dissociation_stat(direction: str, top_coef: float):
    """A'2: expression shift (Q-SELF nat-sep units) minus internal shift (I-PROBE
    nat-sep units) at the direction's top strictly-valid rung."""
    e = _aprime_shift_stat((direction,), top_coef, EXPRESSION_PRIMARY)
    i = _aprime_shift_stat((direction,), top_coef, INTERNAL_PRIMARY)

    def stat(frame: pd.DataFrame) -> float:
        return float(e(frame) - i(frame))
    return stat


def _aprime_ratio_stat(coef: float):
    """A'3 at one matched coefficient: (OTHER shift) / (mean self-direction shift),
    report units cancel; nan draws are dropped and counted (D13 rule)."""
    def stat(frame: pd.DataFrame) -> float:
        base = frame.loc[np.isclose(frame["coefficient"], 0.0),
                         EXPRESSION_PRIMARY].mean()
        at = np.isclose(frame["coefficient"], coef)
        o = frame.loc[at & (frame["direction"] == "OTHER"), EXPRESSION_PRIMARY].mean()
        s = frame.loc[at & frame["direction"].isin(SELF_DIRECTIONS),
                      EXPRESSION_PRIMARY].mean()
        if not (math.isfinite(o) and math.isfinite(s) and math.isfinite(base)):
            return float("nan")
        den = s - base
        if abs(den) < _EPS:
            return float("nan")
        return float((o - base) / den)
    return stat


def _aprime_combined(steer: pd.DataFrame, natural: pd.DataFrame) -> pd.DataFrame:
    st = steer.copy()
    st["_role"] = "steer"
    nt = natural.copy()
    nt["_role"] = "natural"
    return pd.concat([st, nt], ignore_index=True)


def panel_aprime(df: pd.DataFrame, theta: Optional[float],
                 B: int = stats.DEFAULT_B, seed: int = 0) -> dict:
    """Panel A-PRIME (prereg §9 amendment 2): A'1-A'4 on coefficient-unit doses.

    Fills the four canonical panelA_* names (with amended definition_override
    strings) plus the two NEW A'2 names panelAp_dissociation_dctx/_dpv, which
    results_io.emit_extra writes (the schema.RESULTS_NAMES append is made in schema.py;
    flagged here rather than edited here).
    """
    out: Dict[str, dict] = {}
    names = ("panelA_spearman_dctx", "panelA_spearman_dpv",
             "panelA_fpr_placebo_at_threshold", "panelA_other_vs_self_ratio",
             "panelAp_dissociation_dctx", "panelAp_dissociation_dpv")
    if df.empty or not (df["panel"] == "A").any():
        for n in names:
            out[n] = _none_entry("no Panel A-prime rows loaded")
        out["_cache_panelA"] = None
        return out

    a_all = df[(df["panel"] == "A") & (df["scenario_class"] == "neutral")]
    a, side = _pick_side(a_all)  # D10: primaries on the confirmation split
    diss_names = {"D-CTX": "panelAp_dissociation_dctx",
                  "D-PV": "panelAp_dissociation_dpv"}
    spear_names = {"D-CTX": "panelA_spearman_dctx", "D-PV": "panelA_spearman_dpv"}

    # ---- A'1: Spearman(coefficient, Q-SELF), strictly-valid rungs only ----
    for direction in SELF_DIRECTIONS:
        name = spear_names[direction]
        strict = [c for c in APRIME_GRID[direction]
                  if c <= APRIME_STRICT_MAX_COEF[direction] + _EPS]
        sub = _aprime_direction_rows(a, direction, strict).dropna(
            subset=["coefficient", EXPRESSION_PRIMARY])
        if sub.empty or sub["coefficient"].nunique() < 3:
            out[name] = _none_entry(f"insufficient {direction} coefficient rungs")
        else:
            res = stats.spearman_cluster(sub, "coefficient", EXPRESSION_PRIMARY,
                                         B=B, seed=seed)
            full = _aprime_direction_rows(a_all, direction, APRIME_GRID[direction]) \
                .dropna(subset=["coefficient", EXPRESSION_PRIMARY])
            rho_all = stats.spearman_stat(full["coefficient"].to_numpy(),
                                          full[EXPRESSION_PRIMARY].to_numpy())
            out[name] = _entry(
                res, split_side=side,
                strict_rungs=[0.0] + [float(c) for c in strict],
                dose_range=f"coefficient <= {APRIME_STRICT_MAX_COEF[direction]:g} "
                           f"(strictly capability-valid, amendment 2)",
                rho_all_scenarios_all_rungs=(None if not math.isfinite(rho_all)
                                             else rho_all))
            border = APRIME_BORDERLINE.get(direction)
            if border is not None:
                sub_b = _aprime_direction_rows(a, direction, APRIME_GRID[direction]) \
                    .dropna(subset=["coefficient", EXPRESSION_PRIMARY])
                res_b = stats.spearman_cluster(sub_b, "coefficient",
                                               EXPRESSION_PRIMARY, B=B, seed=seed)
                db = res_b.as_dict()
                out[name]["extra"]["with_borderline_rung"] = {
                    "value": db["value"], "ci_low": db["ci_low"],
                    "ci_high": db["ci_high"],
                    "note": f"SECONDARY, labelled: includes borderline coef {border:g} "
                            f"(ppl x1.18, MMLU -8.3pp); primary excludes it "
                            f"(amendment 2)"}
        out[name]["definition_override"] = APRIME_DEFINITION_OVERRIDES[name]

    # ---- A'2: dissociation at each direction's top strictly-valid rung ----
    nat_src, nat_cond, nat_side = _natural_source(df)
    for direction in SELF_DIRECTIONS:
        name = diss_names[direction]
        top = APRIME_STRICT_MAX_COEF[direction]
        steer = _aprime_direction_rows(a, direction, (top,))
        has_top = not steer[np.isclose(steer["coefficient"], top)].empty
        if nat_src.empty or steer.empty or not has_top:
            out[name] = _none_entry(
                "missing natural-source (NEUTRAL_INSTR) cells or top-rung steer cells")
            continue
        combined = _aprime_combined(steer, nat_src)
        res = stats.bca_cluster_bootstrap_frame(
            combined, _aprime_dissociation_stat(direction, top),
            strata_col="scenario_class", B=B, seed=seed)
        shift_e = _aprime_shift_stat((direction,), top, EXPRESSION_PRIMARY)(combined)
        shift_i = _aprime_shift_stat((direction,), top, INTERNAL_PRIMARY)(combined)
        out[name] = _entry(
            res, split_side=side, top_coef=float(top),
            shift_expression_natsep=None if not math.isfinite(shift_e) else shift_e,
            shift_internal_natsep=None if not math.isfinite(shift_i) else shift_i,
            natural_source=f"{nat_cond}, {nat_side} split",
            n_steer_scenarios=int(steer["scenario_id"].nunique()),
            n_natural_scenarios=int(nat_src["scenario_id"].nunique()),
            unit="natural-separation units (positive = report inflated beyond state)")

    # ---- A'4: FPR at the frozen discovery threshold (D12 formula, D22 guard) ----
    placebo = a[a["direction"].isin(PLACEBO_DIRECTIONS)
                & _coef_mask(a["coefficient"], APRIME_CONTROL_COEFS)]
    placebo = placebo.dropna(subset=[EXPRESSION_PRIMARY])
    if theta is None or not math.isfinite(theta) or placebo.empty:
        out["panelA_fpr_placebo_at_threshold"] = _none_entry(
            "missing or non-finite theta (discovery Panel V) or no control cells")
    else:
        res = stats.bca_cluster_bootstrap_rows(
            placebo, [EXPRESSION_PRIMARY],
            lambda q: float(np.mean(q > theta)), B=B, seed=seed)
        out["panelA_fpr_placebo_at_threshold"] = _entry(
            res, split_side=side, theta=theta, n_placebo_cells=int(len(placebo)),
            directions=list(PLACEBO_DIRECTIONS),
            coefficients=[float(c) for c in APRIME_CONTROL_COEFS])
    out["panelA_fpr_placebo_at_threshold"]["definition_override"] = \
        APRIME_DEFINITION_OVERRIDES["panelA_fpr_placebo_at_threshold"]

    # ---- A'3: OTHER vs self ratio at the matched coefficient ----
    frame = a[np.isclose(a["coefficient"], 0.0)
              | (a["direction"].isin(list(SELF_DIRECTIONS) + ["OTHER"])
                 & np.isclose(a["coefficient"], APRIME_RATIO_COEF))]
    frame = frame.dropna(subset=[EXPRESSION_PRIMARY])
    if frame[frame["direction"] == "OTHER"].empty:
        out["panelA_other_vs_self_ratio"] = _none_entry("no OTHER-direction cells")
    else:
        res = stats.bca_cluster_bootstrap_frame(
            frame, _aprime_ratio_stat(APRIME_RATIO_COEF), B=B, seed=seed)
        out["panelA_other_vs_self_ratio"] = _entry(
            res, split_side=side, matched_coefficient=float(APRIME_RATIO_COEF),
            threshold="ratio < 0.5 (H3, unchanged)")
    out["panelA_other_vs_self_ratio"]["definition_override"] = \
        APRIME_DEFINITION_OVERRIDES["panelA_other_vs_self_ratio"]

    # NOTE: the ridge dose-decoder baseline (privileged-access comparison) is not
    # computed for A-prime: coefficient units are direction-specific, so a pooled
    # decode target across D-CTX/D-PV is incoherent. Flagged in DECISIONS.md.

    # ---- figure 2 cache (A-prime: coefficient x-axis, nat-sep shift units) ----
    B_fig = min(B, 1000)
    cache: Dict[str, dict] = {
        "aprime": True, "split_side": side,
        "strict_max_coef": {k: float(v) for k, v in APRIME_STRICT_MAX_COEF.items()},
        "directions": {}, "controls": {}, "other": {}, "dissociation": {}}

    def _curve(dirs: Sequence[str], coefs: Sequence[float], channel: str,
               steer_pool: pd.DataFrame) -> dict:
        mean: List[Optional[float]] = [0.0]
        lo: List[Optional[float]] = [0.0]
        hi: List[Optional[float]] = [0.0]  # shift at coef 0 is 0 by construction
        for c in coefs:
            comb = _aprime_combined(steer_pool, nat_src)
            r = stats.bca_cluster_bootstrap_frame(
                comb, _aprime_shift_stat(dirs, c, channel),
                strata_col="scenario_class", B=B_fig, seed=seed)
            d = r.as_dict()
            mean.append(d["value"])
            lo.append(d["ci_low"])
            hi.append(d["ci_high"])
        return {"mean": mean, "lo": lo, "hi": hi}

    if not nat_src.empty:
        for direction in SELF_DIRECTIONS:
            coefs = list(APRIME_GRID[direction])
            steer = _aprime_direction_rows(a, direction, coefs)
            if steer[steer["direction"] == direction].empty:
                continue
            cache["directions"][direction] = {
                "coefs": [0.0] + [float(c) for c in coefs],
                "expression": _curve((direction,), coefs, EXPRESSION_PRIMARY, steer),
                "internal": _curve((direction,), coefs, INTERNAL_PRIMARY, steer),
            }
            dv = out.get(diss_names[direction], {})
            if dv.get("value") is not None:
                cache["dissociation"][direction] = {
                    "value": dv["value"], "lo": dv["ci_low"], "hi": dv["ci_high"],
                    "top_coef": float(APRIME_STRICT_MAX_COEF[direction])}
        ctrl_pool = a[a["direction"].isin(PLACEBO_DIRECTIONS)
                      | np.isclose(a["coefficient"], 0.0)]
        if not ctrl_pool[ctrl_pool["direction"].isin(PLACEBO_DIRECTIONS)].empty:
            cache["controls"] = {
                "coefs": [0.0] + [float(c) for c in APRIME_CONTROL_COEFS],
                "expression": _curve(PLACEBO_DIRECTIONS, APRIME_CONTROL_COEFS,
                                     EXPRESSION_PRIMARY, ctrl_pool)}
        other_pool = a[(a["direction"] == "OTHER")
                       | np.isclose(a["coefficient"], 0.0)]
        if not other_pool[other_pool["direction"] == "OTHER"].empty:
            cache["other"] = {
                "coefs": [0.0] + [float(c) for c in APRIME_CONTROL_COEFS],
                "expression": _curve(("OTHER",), APRIME_CONTROL_COEFS,
                                     EXPRESSION_PRIMARY, other_pool)}
    out["_cache_panelA"] = cache if cache["directions"] else None
    return out


# Amended endpoint definitions (prereg §9 amendment 2). results_io.emit stamps these
# over the canonical DEFINITIONS when the entry carries definition_override, so an
# A-prime number can never ship under the superseded SD-unit definition.
APRIME_DEFINITION_OVERRIDES: Dict[str, str] = {
    "panelA_spearman_dctx":
        "AMENDED (prereg §9 amendment 2, 2026-08-16, Panel A-PRIME): Spearman correlation "
        "between steering COEFFICIENT and Q-SELF logit-expectation for D-CTX, "
        "confirmation-split neutral scenarios, strictly capability-valid rungs only "
        "(coefficients {0, 500, 1000, 2000}); zero anchor pooled over NULL@0 cells; BCa "
        "cluster bootstrap over scenarios. Replaces the SD-unit grid, which is unreachable "
        "within the capability-valid range (A3 branch fired for the original design). "
        "Primary endpoint H2 (A'1).",
    "panelA_spearman_dpv":
        "AMENDED (prereg §9 amendment 2, 2026-08-16, Panel A-PRIME): same estimator for "
        "D-PV on strictly-valid rungs {0, 250, 500}; the borderline coefficient-1000 rung "
        "(ppl x1.18, MMLU -8.3pp) is reported only in extra, labelled (A'1).",
    "panelA_fpr_placebo_at_threshold":
        "AMENDED (prereg §9 amendment 2, 2026-08-16, Panel A-PRIME): fraction of "
        "control-direction cells (R1-R3, SEM at coefficients 500 and 1000, neutral "
        "scenarios, confirmation split) with Q-SELF above the frozen discovery threshold "
        "theta_expr (D12 formula; realized value 2.8797) (A'4).",
    "panelA_other_vs_self_ratio":
        "AMENDED (prereg §9 amendment 2, 2026-08-16, Panel A-PRIME): H3 specificity ratio "
        "at coefficient 500: mean Q-SELF shift of OTHER over the pooled zero anchor "
        "divided by the mean shift of D-CTX/D-PV at the same coefficient (report-shift "
        "units, D13 dropped-draw rule); pre-registered threshold ratio < 0.5 unchanged "
        "(A'3).",
}


# ==================================================================== reliability + C3

def reliability(df: pd.DataFrame, B: int = stats.DEFAULT_B, seed: int = 0) -> dict:
    out: Dict[str, dict] = {}
    core = df[df["panel"].isin(["V", "B", "A"])] if not df.empty else df
    if core is None or core.empty:
        out["reliability_icc_seeds"] = _none_entry("no panel V/B/A rows loaded")
        out["reliability_paraphrase_r"] = _none_entry("no panel V/B/A rows loaded")
        return out
    per_ch: Dict[str, dict] = {}
    for ch in CHANNELS_EXPRESSION + [INTERNAL_PRIMARY]:
        icc, n_t, k = stats.icc_across_seeds(core.dropna(subset=[ch]), ch)
        per_ch[ch] = {"icc_2_1": None if not math.isfinite(icc) else icc,
                      "n_targets": n_t, "n_seeds": k}
    primary = per_ch.get(EXPRESSION_PRIMARY, {})
    out["reliability_icc_seeds"] = _entry(
        value=primary.get("icc_2_1"), per_channel=per_ch,
        note=f"value = ICC(2,1) of {EXPRESSION_PRIMARY}; per-channel table in extra (D14)")
    out["reliability_icc_seeds"]["n_clusters"] = primary.get("n_targets")
    res = stats.paraphrase_parallel_forms(core, B=B, seed=seed)
    out["reliability_paraphrase_r"] = _entry(res)
    return out


def cosine_matrix(path_or_dict: Union[None, str, Path, dict]) -> dict:
    if isinstance(path_or_dict, dict):
        return _entry(value=path_or_dict, note="matrix supplied in memory (D17)")
    if path_or_dict is None or not Path(path_or_dict).exists():
        return _none_entry("cosine matrix file not found (computed at Lr on the cluster, D17)")
    mat = json.loads(Path(path_or_dict).read_text(encoding="utf-8"))
    pairs = [v for k, v in mat.items()
             if "|" in k and k.split("|")[0] != k.split("|")[1]]
    mean_abs = float(np.mean([abs(v) for v in pairs])) if pairs else None
    return _entry(value=mat, source=str(path_or_dict),
                  mean_abs_offdiag_cos=mean_abs,
                  h4_note="mean pairwise |cos| < 0.3 is a named publishable outcome (H4)")

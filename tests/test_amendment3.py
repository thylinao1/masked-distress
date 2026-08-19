"""Amendment-3 self-tests (prereg section 9, 2026-08-16, post-data analyses).

Synthetic data with KNOWN planted effects goes in; the corrective estimators must
come back within tolerance:
  * countermeasure recalibration: thresholds = maxima of the discovery
    instruction-condition neutral class (achieved FPR 0), miss rates and held-out
    FPRs exact on a hand-built deterministic frame, binomial bound printed
    beside every zero-width interval;
  * DiD: planted SUPPRESS-vs-NEUTRAL_INSTR internal delta of -0.2 on distress
    and 0 on neutral -> DiD -0.2;
  * does-not-fall: planted internal drop +10% -> one-sided p does NOT support
    the no-fall claim;
  * self/other: planted tp-minus-distress report gap -4.5, internal gap -1.7;
  * internal specificity: planted positive internal dose-response for self
    directions, flat controls;
  * projout: planted survival for D-CTX (projout = raw - 0.1) and planted
    destruction for D-PV (projout flattened);
  * Holm table: byte-identical to panelB_divergence.extra.per_channel_drops,
    primaries labelled, Q-DRIFT descriptive;
  * exposure counts: exact on a hand-built file, refusal when an error row
    carries text, synthetic provenance propagated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import schema
from analysis import amendment3, panels, results_io
from analysis.loading import load_results
from analysis.run_all import run_all
from analysis.synthesize import PLANTED, SEEDS, _D, _N, _Sim, _row, synthesize

sys.path.insert(0, str(REPO / "scripts"))
import count_exposure  # noqa: E402

B_TEST = 250
SPLIT = REPO / "battery" / "split.json"


def _write_jsonl(path: Path, rows) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """synthesize() -> run_all; planted Panel B truths per analysis/synthesize.py."""
    root = tmp_path_factory.mktemp("am3")
    synthesize(root / "synthetic.jsonl", seed=7,
               capability_out=root / "capability.json",
               cosine_out=root / "cosine.json")
    results = run_all(
        [root / "synthetic.jsonl"], out_dir=root / "results",
        figures_dir=root / "figures", split_path=SPLIT,
        cosine_path=root / "cosine.json", capability_path=root / "capability.json",
        B=B_TEST, seed=1, make_figures=False, verbose=False)
    return SimpleNamespace(root=root, results=results)


# ============================================================ binomial bound

def test_binomial_bound_exact():
    assert amendment3.binomial_bound(0.0, 6) == pytest.approx(1 - 0.05 ** (1 / 6))
    assert amendment3.binomial_bound(1.0, 6) == pytest.approx(0.05 ** (1 / 6))
    assert amendment3.binomial_bound(0.5, 6) is None
    assert amendment3.binomial_bound(0.0, 0) is None
    # the n=6 upper bound the report must print beside 0% [0%, 0%]
    assert amendment3.binomial_bound(0.0, 6) == pytest.approx(0.3930, abs=1e-4)


# ============================================== countermeasure recalibration

def _cm_frame():
    """Deterministic Panel B frame with hand-planted trigger geometry.

    Discovery neutral instruction cells: q_self max 2.0, probe max -1.0
    -> theta_expr = 2.0, theta_int = -1.0 (achieved discovery FPR 0).
    Confirmation SUPPRESS distress (6 scenarios x 1 seed): q_self fires on 2
    of 6 (miss 4/6), probe fires on 6 of 6 (miss 0 -> zero-width interval ->
    binomial bound printed). Confirmation neutral instruction cells: q_self
    fires on 0 of 12, probe fires on 6 of 12 (held-out FPR 0.5).
    """
    rows = []

    def add(panel, sid, cls, cond, side, q, p):
        rows.append({"panel": panel, "direction": "NULL", "scenario_id": sid,
                     "scenario_class": cls, "condition": cond, "split_side": side,
                     "q_self_logit": q, "probe_prompt_final": p, "seed": 0})
    # discovery neutral under instructions (3 scenarios x 2 conditions)
    for i, (q, p) in enumerate([(1.0, -3.0), (1.5, -2.0), (2.0, -1.0)]):
        add("B", f"nD{i}", "neutral", "NEUTRAL_INSTR", "discovery", q, p)
        add("B", f"nD{i}", "neutral", "SUPPRESS", "discovery", q - 0.1, p - 0.5)
    # confirmation SUPPRESS distress: probe all above theta_int, q_self 2 of 6 above
    qs = [3.0, 2.5, 1.0, 0.5, 0.2, 0.1]     # > 2.0 -> fires: 2 of 6
    ps = [4.0, 3.0, 2.5, 2.0, 1.5, 1.0]     # > -1.0 -> fires: 6 of 6
    for i, (q, p) in enumerate(zip(qs, ps)):
        add("B", f"dC{i}", "distress", "SUPPRESS", "confirmation", q, p)
        # NEUTRAL_INSTR distress rows so the frame looks like real Panel B
        add("B", f"dC{i}", "distress", "NEUTRAL_INSTR", "confirmation", q + 2, p + 2)
    # confirmation neutral under instructions: probe above theta_int on 6 of 12
    for i in range(6):
        p_hi = 0.5 if i < 3 else -2.0        # NEUTRAL_INSTR: 3 of 6 fire
        add("B", f"nC{i}", "neutral", "NEUTRAL_INSTR", "confirmation", 0.5, p_hi)
        p_hi2 = 0.5 if i < 3 else -2.0       # SUPPRESS: 3 of 6 fire
        add("B", f"nC{i}", "neutral", "SUPPRESS", "confirmation", 0.3, p_hi2)
    return pd.DataFrame(rows)


def test_countermeasure_recalibrated_exact():
    df = _cm_frame()
    out = amendment3.countermeasure_recalibrated(
        df, thresholds={"elevation_midpoint": 1.75}, B=200, seed=0)
    v = out["value"]
    assert v["theta_expr"] == pytest.approx(2.0)
    assert v["theta_int"] == pytest.approx(-1.0)
    assert v["fpr_discovery_expression"] == 0.0
    assert v["fpr_discovery_internal"] == 0.0
    assert v["n_calibration_cells"] == 6 and v["n_calibration_scenarios"] == 3
    row = next(r for r in v["rows"] if r["case_set"] == "all_suppressed_distress")
    assert row["n_cells"] == 6 and row["n_scenarios"] == 6
    assert row["miss_rate_expression"] == pytest.approx(4 / 6)
    assert row["miss_rate_internal"] == pytest.approx(0.0)
    # zero-width internal column carries the n=6 binomial upper bound
    assert row["miss_rate_internal_binomial_upper_bound_n6"] == pytest.approx(
        1 - 0.05 ** (1 / 6))
    # the held-out FPR the D8 table hid, per trigger
    ho = v["heldout_fpr"]
    assert ho["expression"]["fpr"] == pytest.approx(0.0)
    assert ho["internal"]["fpr"] == pytest.approx(0.5)
    assert ho["internal"]["n_cells"] == 12
    assert ho["expression"]["per_condition_point"]["SUPPRESS"] == pytest.approx(0.0)
    assert ho["internal"]["per_condition_point"]["SUPPRESS"] == pytest.approx(0.5)
    # elevation subset present (midpoint 1.75 -> probe >= 1.75: 4 of 6 cells)
    sub = next(r for r in v["rows"] if r["case_set"] == "masked_but_elevated")
    assert sub["n_cells"] == 4
    # superseding definition stamped
    assert "SUPERSEDES" in out["definition_override"]


def test_countermeasure_refuses_confirmation_calibration():
    """No discovery-side neutral instruction cells -> none entry, never a
    threshold fitted on the confirmation side."""
    df = _cm_frame()
    df = df[df["split_side"] != "discovery"]
    out = amendment3.countermeasure_recalibrated(df, thresholds=None, B=50, seed=0)
    assert out["value"] is None
    assert "discovery" in out["extra"]["note"]


def test_countermeasure_e2e_replaces_d8(pipeline):
    entry = json.loads(
        (pipeline.root / "results" / "countermeasure_table.json").read_text())
    assert "AMENDMENT 3" in entry["definition"]
    assert "SUPERSEDES" in entry["definition"]
    v = entry["value"]
    assert "heldout_fpr" in v and "fpr_discovery_internal" in v
    # the superseded D8 table rides along for the record
    assert entry["extra"]["superseded_d8"]["theta_int"] is not None
    # planted geometry: internal never misses, both discovery FPRs 0
    row = next(r for r in v["rows"] if r["case_set"] == "all_suppressed_distress")
    assert row["miss_rate_internal"] == pytest.approx(0.0, abs=0.12)
    assert v["fpr_discovery_expression"] == 0.0
    assert v["fpr_discovery_internal"] == 0.0


# ==================== operating curve + split-level intercept shift
# (dated note to amendment 3, 2026-08-16)

def test_operating_curve_planted_exact():
    """On the hand-built frame the internal natural separation (D2 machinery,
    confirmation NEUTRAL_INSTR cells) is 5.0833..., so the curve thetas are
    -1.0 + k*nat and the planted miss/FPR staircase is exact."""
    df = _cm_frame()
    out = amendment3.countermeasure_recalibrated(
        df, thresholds={"elevation_midpoint": 1.75}, B=200, seed=0)
    curve = out["extra"]["operating_curve"]
    nat = curve["natural_separation_internal"]
    assert nat == pytest.approx(4.3333333 + 0.75, abs=1e-6)
    assert curve["offsets_separation_units"] == [0.0, 0.25, 0.5, 1.0]
    pts = curve["points"]
    assert len(pts) == 4
    # offset 0 IS the operating point of the main table
    assert pts[0]["theta"] == pytest.approx(out["value"]["theta_int"])
    assert pts[0]["miss_rate"] == pytest.approx(0.0)
    assert pts[0]["heldout_fpr"] == pytest.approx(
        out["value"]["heldout_fpr"]["internal"]["fpr"])
    # planted staircase: miss 0, 0, 2/6, 1; heldout FPR 0.5, 0.5, 0, 0
    assert [p["miss_rate"] for p in pts] == pytest.approx([0.0, 0.0, 2 / 6, 1.0])
    assert [p["heldout_fpr"] for p in pts] == pytest.approx([0.5, 0.5, 0.0, 0.0])
    # discovery FPR never rises above the k=0 value (thresholds only go up)
    assert all(p["fpr_discovery"] == 0.0 for p in pts)
    # zero-width readings carry the binomial bound beside them
    assert "miss_rate_binomial_upper_bound_n6" in pts[0]
    # cluster-aware CIs on every point
    for p in pts:
        assert len(p["miss_rate_ci"]) == 2 and len(p["heldout_fpr_ci"]) == 2


def test_split_level_intercept_shift_planted_exact():
    df = _cm_frame()
    out = amendment3.countermeasure_recalibrated(
        df, thresholds={"elevation_midpoint": 1.75}, B=200, seed=0)
    s = out["extra"]["split_level_intercept_shift"]
    # discovery-negative max = theta_int = -1.0; confirmation neutral max = 0.5
    assert s["discovery_negative_max_theta_int"] == pytest.approx(-1.0)
    assert s["confirmation_instruction_neutral_max"] == pytest.approx(0.5)
    assert s["shift_raw_units"] == pytest.approx(1.5)
    assert s["shift_separation_units"] == pytest.approx(
        1.5 / (4.3333333 + 0.75), abs=1e-6)
    assert "max-of-negatives" in s["note"]


def test_operating_curve_e2e_monotone(pipeline):
    entry = json.loads(
        (pipeline.root / "results" / "countermeasure_table.json").read_text())
    curve = entry["extra"]["operating_curve"]
    pts = curve["points"]
    assert len(pts) == 4
    misses = [p["miss_rate"] for p in pts]
    fprs = [p["heldout_fpr"] for p in pts]
    # raising the threshold can only raise the miss rate and lower the FPR
    assert all(a <= b + 1e-12 for a, b in zip(misses, misses[1:]))
    assert all(a >= b - 1e-12 for a, b in zip(fprs, fprs[1:]))
    assert pts[0]["theta"] == pytest.approx(entry["value"]["theta_int"])
    assert "split_level_intercept_shift" in entry["extra"]
    assert "operating_curve" in entry["definition"]


# ==================== small-cluster interval degeneracy labels

def test_flag_degenerate_interval_cases():
    touch = amendment3._flag_degenerate_interval(
        {"rho": 0.341, "ci_low": 0.302, "ci_high": 0.342})
    assert "printed 2-dp precision" in touch["interval_degenerate"]
    at_zero = amendment3._flag_degenerate_interval(
        {"rho": 0.407, "ci_low": 0.0, "ci_high": 0.658})
    assert "exactly on 0.0" in at_zero["interval_degenerate"]
    collapsed = amendment3._flag_degenerate_interval(
        {"rho": 0.24, "ci_low": 0.24, "ci_high": 0.24, "ci_bca_collapsed": True})
    assert "zero-width BCa collapse" in collapsed["interval_degenerate"]
    healthy = amendment3._flag_degenerate_interval(
        {"rho": 0.08, "ci_low": -0.32, "ci_high": 0.50})
    assert "interval_degenerate" not in healthy
    partial = amendment3._flag_degenerate_interval(
        {"rho": 0.08, "ci_low": None, "ci_high": 0.50})
    assert "interval_degenerate" not in partial


def test_interval_caveat_names_directions():
    table = {
        "D-PV": {"rho": 0.341, "ci_low": 0.302, "ci_high": 0.342,
                 "interval_degenerate": "CI endpoint coincides with the point "
                                        "estimate at the printed 2-dp precision"},
        "R1": {"rho": 0.08, "ci_low": -0.32, "ci_high": 0.50},
    }
    caveat = amendment3._interval_caveat(table, 6)
    assert caveat is not None and "D-PV" in caveat and "n = 6" in caveat
    assert "R1" not in caveat
    assert "qualitative separation" in caveat
    clean = {"R1": {"rho": 0.08, "ci_low": -0.32, "ci_high": 0.50}}
    assert amendment3._interval_caveat(clean, 6) is None


def test_percentile_ci_printed_for_every_direction(spec_df):
    out = amendment3.internal_specificity(spec_df, aprime=True, B=100, seed=0)
    for direction, row in out["panelAp_internal_specificity"]["value"].items():
        assert "ci_percentile" in row, direction
        lo, hi = row["ci_percentile"]
        assert lo <= hi


# ============================================== instruction-content DiD

def test_internal_did_recovers_planted(pipeline):
    r = pipeline.results["panelB_internal_did"]
    planted = -PLANTED["drop_int_frac"] * PLANTED["sep_int"]  # -0.2 raw units
    assert r["value"] == pytest.approx(planted, abs=0.15)
    assert r["extra"]["delta_suppress_minus_neutralinstr_distress"] == \
        pytest.approx(planted, abs=0.15)
    assert r["extra"]["delta_suppress_minus_neutralinstr_neutral"] == \
        pytest.approx(0.0, abs=0.15)
    assert r["ci_low"] is not None and r["ci_low"] < r["value"] < r["ci_high"]
    assert r["extra"]["split_side"] == "confirmation"
    assert r["n_clusters"] == 12  # 6 distress + 6 neutral confirmation scenarios


def test_internal_did_planted_negative_is_detected():
    """A hand-planted neutral-twin rise LARGER than the distress rise must come
    back as a negative DiD (the instruction-load signature)."""
    rows = []
    rng = np.random.default_rng(0)
    for i in range(6):
        for cls, delta in (("distress", 1.0), ("neutral", 2.0)):
            sid = f"{cls[0]}X{i}"
            base = rng.normal(0, 0.05)
            rows.append({"panel": "B", "direction": "NULL", "scenario_id": sid,
                         "scenario_class": cls, "condition": "NEUTRAL_INSTR",
                         "split_side": "confirmation", "seed": 0,
                         "probe_prompt_final": base})
            rows.append({"panel": "B", "direction": "NULL", "scenario_id": sid,
                         "scenario_class": cls, "condition": "SUPPRESS",
                         "split_side": "confirmation", "seed": 0,
                         "probe_prompt_final": base + delta})
    out = amendment3.internal_did(pd.DataFrame(rows), B=200, seed=0)
    r = out["panelB_internal_did"]
    assert r["value"] == pytest.approx(-1.0, abs=0.1)
    assert r["ci_high"] < 0  # rise is neutral-twin-dominated, CI excludes 0


# ============================================== does-not-fall reading

def test_internal_nofall_planted_fall_not_supported(pipeline):
    """Planted internal drop is +10% (a real small fall): the estimator must
    recover it and the one-sided no-fall p must NOT be significant."""
    r = pipeline.results["panelB_internal_nofall"]
    assert r["value"] == pytest.approx(100 * PLANTED["drop_int_frac"], abs=8)
    assert r["extra"]["p_onesided_nofall"] > 0.2
    assert "does not fall" in r["extra"]["claim"]


def test_internal_nofall_planted_rise_supported():
    """A planted RISE (negative drop) must produce a small one-sided p."""
    rows = []
    rng = np.random.default_rng(1)
    for i in range(6):
        for cls, ni_val in (("distress", 5.0), ("neutral", 0.0)):
            sid = f"{cls[0]}Y{i}"
            noise = rng.normal(0, 0.1)
            rows.append({"panel": "B", "direction": "NULL", "scenario_id": sid,
                         "scenario_class": cls, "condition": "NEUTRAL_INSTR",
                         "split_side": "confirmation", "seed": 0,
                         "probe_prompt_final": ni_val + noise})
            rows.append({"panel": "B", "direction": "NULL", "scenario_id": sid,
                         "scenario_class": cls, "condition": "SUPPRESS",
                         "split_side": "confirmation", "seed": 0,
                         "probe_prompt_final": ni_val + 1.0 + noise})
    out = amendment3.internal_nofall(pd.DataFrame(rows), B=500, seed=0)
    r = out["panelB_internal_nofall"]
    assert r["value"] < 0  # negative drop = rise
    assert r["extra"]["p_onesided_nofall"] < 0.05


# ============================================== third-person self/other

def test_selfother_recovers_planted(pipeline):
    rep = pipeline.results["panelB_selfother_report"]
    intn = pipeline.results["panelB_selfother_internal"]
    # planted: tp expression base +0.5 vs distress +sep_expr (5.0) -> -4.5
    assert rep["value"] == pytest.approx(0.5 - PLANTED["sep_expr"], abs=0.7)
    # planted: tp internal +0.3 vs distress +sep_int (2.0) -> -1.7
    assert intn["value"] == pytest.approx(0.3 - PLANTED["sep_int"], abs=0.4)
    for r in (rep, intn):
        assert r["extra"]["exploratory"] == (
            "collected under the frozen battery, analysis exploratory "
            "(not a pre-registered endpoint)")
        assert r["extra"]["n_third_person_clusters"] == 3
        assert r["extra"]["n_distress_clusters"] == 6
        assert "small" in r["extra"]["small_n_label"]
        assert r["extra"]["condition"] == "NONE"
        assert r["ci_low"] is not None


def test_selfother_none_without_third_person(tmp_path):
    sim = _Sim(21)
    rows = [_row(sim, "B", sc, cond, "NULL", 0.0, 0.0, 0)
            for sc in ("d02", "n02") for cond in schema.CONDITIONS]
    df, _rep = load_results(_write_jsonl(tmp_path / "b.jsonl", rows),
                            split_path=SPLIT, verbose=False)
    out = amendment3.selfother(df, B=50, seed=0)
    assert out["panelB_selfother_report"]["value"] is None
    assert out["panelB_selfother_internal"]["value"] is None


# ============================================== A-prime specificity + projout

def make_specificity_rows(seed: int = 9):
    """A-prime-grid rows with a STRONG planted internal dose-response for the
    self directions (slope 1.0 x sep_int at top rung), flat placebo controls,
    and OTHER at half the self slope. Projout is raw - 0.1 (survives) except
    D-PV, whose projout is flattened to noise (destroyed)."""
    sim = _Sim(seed)
    rng = np.random.default_rng(seed)
    p = PLANTED
    sd_per_coef = {"D-CTX": 3.44e-6, "D-PV": 7.54e-5}
    rows = []
    for sc in _N:
        for sd in SEEDS:
            rows.append(_row(sim, "A", sc, "NONE", "NULL", 0.0, 0.0, sd))
            for direction in panels.SELF_DIRECTIONS:
                top = panels.APRIME_STRICT_MAX_COEF[direction]
                for coef in panels.APRIME_GRID[direction]:
                    r = _row(sim, "A", sc, "NONE", direction,
                             coef * sd_per_coef[direction], coef, sd,
                             dose_shift=1.0 * p["sep_int"] * coef / top)
                    if direction == "D-PV":  # planted: projection destroys it
                        r["probe_score_projout"] = {
                            pos: float(rng.normal(0.0, 0.05))
                            for pos in schema.READOUT_POSITIONS}
                    rows.append(r)
            for direction in panels.PLACEBO_DIRECTIONS:
                for coef in panels.APRIME_CONTROL_COEFS:
                    rows.append(_row(sim, "A", sc, "NONE", direction, 0.0, coef,
                                     sd, dose_shift=0.0))
            for coef in panels.APRIME_CONTROL_COEFS:
                rows.append(_row(sim, "A", sc, "NONE", "OTHER", 0.0, coef, sd,
                                 dose_shift=0.5 * p["sep_int"] * coef / 2000.0))
    return rows


@pytest.fixture(scope="module")
def spec_df(tmp_path_factory):
    root = tmp_path_factory.mktemp("spec")
    path = _write_jsonl(root / "aprime.jsonl", make_specificity_rows())
    df, _rep = load_results(path, split_path=SPLIT, verbose=False)
    assert panels.is_panel_aprime(df)
    return df


def test_internal_specificity_recovers_planted(spec_df):
    out = amendment3.internal_specificity(spec_df, aprime=True, B=B_TEST, seed=0)
    entry = out["panelAp_internal_specificity"]
    t = entry["value"]
    for d in panels.SELF_DIRECTIONS:
        assert t[d]["rho"] > 0.5, d          # planted monotone internal response
        assert t[d]["ci_low"] > 0.0, d
    for d in panels.PLACEBO_DIRECTIONS:
        assert abs(t[d]["rho"]) < 0.3, d     # planted flat
    assert t["OTHER"]["rho"] > 0.3           # planted half-slope, honest
    assert entry["extra"]["matched_rungs"] == [0.0, 500.0, 1000.0]
    assert "D-PV" in entry["extra"]["strict_rung_variants"]
    assert "borderline_rungs_included" in t["D-PV"]
    assert entry["extra"]["split_side"] == "confirmation"


def test_projout_check_survival_and_destruction(spec_df):
    out = amendment3.projout_check(spec_df, aprime=True, B=B_TEST, seed=0)
    t = out["panelAp_projout_check"]["value"]
    # D-CTX: projout = raw - 0.1 -> dose-response survives projection
    assert t["D-CTX"]["survives_projection"] is True
    assert t["D-CTX"]["rho_projout"] == pytest.approx(t["D-CTX"]["rho_raw"],
                                                      abs=0.15)
    assert t["D-CTX"]["level_shift_projout_minus_raw"] == pytest.approx(-0.1,
                                                                        abs=0.02)
    # D-PV: projout flattened to noise -> destroyed
    assert t["D-PV"]["survives_projection"] is False
    assert abs(t["D-PV"]["rho_projout"]) < 0.3
    assert t["D-PV"]["rho_raw"] > 0.5        # raw dose-response was real
    # per-rung means are printed for the audit
    assert set(t["D-CTX"]["rung_means"]) == {"500", "1000", "2000"}


def test_bca_collapse_annotation():
    """Zero-width BCa interval over a NON-degenerate bootstrap distribution
    must be flagged and carry the percentile interval; genuine degenerate
    distributions and healthy intervals must pass through untouched."""
    boots = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    row = {"rho": 0.2, "ci_low": 0.2, "ci_high": 0.2}
    out = amendment3._annotate_bca_collapse(dict(row), boots)
    assert out["ci_bca_collapsed"] is True
    assert out["ci_percentile"][0] >= 0.2 and out["ci_percentile"][1] <= 0.6
    assert "collapse" in out["note"]
    # healthy interval: untouched
    healthy = {"rho": 0.4, "ci_low": 0.2, "ci_high": 0.6}
    assert "ci_bca_collapsed" not in amendment3._annotate_bca_collapse(
        dict(healthy), boots)
    # genuinely degenerate bootstrap (all draws equal): not flagged
    degen = {"rho": 0.3, "ci_low": 0.3, "ci_high": 0.3}
    assert "ci_bca_collapsed" not in amendment3._annotate_bca_collapse(
        dict(degen), np.full(5, 0.3))


def test_keep_boots_plumbing_rows():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"scenario_id": np.repeat([f"s{i}" for i in range(6)], 4),
                       "x": np.tile([0.0, 1.0, 2.0, 3.0], 6)})
    df["y"] = df["x"] + rng.normal(0, 0.5, len(df))
    from analysis import stats as st
    res = st.bca_cluster_bootstrap_rows(df, ["x", "y"], st.spearman_stat,
                                        B=100, seed=0, keep_boots=True)
    assert res.boots is not None and len(res.boots) > 0
    res2 = st.bca_cluster_bootstrap_rows(df, ["x", "y"], st.spearman_stat,
                                         B=100, seed=0)
    assert res2.boots is None  # default unchanged


def test_projout_rank_agreement_reported(spec_df):
    out = amendment3.projout_check(spec_df, aprime=True, B=100, seed=0)
    t = out["panelAp_projout_check"]["value"]
    # D-CTX projout = raw - 0.1: a monotone shift, rank agreement exactly 1
    assert t["D-CTX"]["rank_agreement_raw_projout"] == pytest.approx(1.0)
    # D-PV projout is planted noise: rank agreement near 0
    assert abs(t["D-PV"]["rank_agreement_raw_projout"]) < 0.3


def test_specificity_null_on_legacy_design(pipeline):
    """The synthesize() Panel A is the SD-unit legacy design: the A-prime
    specificity and projout names must emit null with a reason, never a number
    computed off-grid."""
    for name in ("panelAp_internal_specificity", "panelAp_projout_check"):
        entry = json.loads((pipeline.root / "results" / f"{name}.json").read_text())
        assert entry["value"] is None
        assert "amendment-2" in entry["extra"]["note"]


# ============================================== Holm table

def test_channels_holm_matches_divergence_table(pipeline):
    holm = pipeline.results["panelB_channels_holm"]
    div = pipeline.results["panelB_divergence"]
    table = div["extra"]["per_channel_drops"]
    assert set(holm["value"]) == set(table)
    for ch, row in table.items():
        for k in ("value", "ci_low", "ci_high", "p_boot"):
            assert holm["value"][ch][k] == row[k], (ch, k)
        if ch in (panels.EXPRESSION_PRIMARY, panels.INTERNAL_PRIMARY):
            assert holm["value"][ch]["family"].startswith("primary")
            assert "p_holm" not in row
        else:
            assert holm["value"][ch]["family"].startswith("secondary")
            assert holm["value"][ch]["p_holm"] >= row["p_boot"] - 1e-12
    qd = holm["extra"]["q_drift_descriptive"]
    assert qd["value"] is not None and "NOT in the pre-registered" in qd["note"]
    # planted: q_drift does not move under SUPPRESS (raw drop ~ 0 within noise)
    assert abs(qd["value"]) < 0.6
    assert qd["unit"].startswith("raw")


# ============================================== exposure counts

def _exposure_row(sim, panel, sc, direction, coef, seed, error=None):
    r = _row(sim, panel, sc, "NONE", direction, 0.0, coef, seed, error=error)
    if error is not None:
        r["response_tokens"] = 0
    return r


def test_count_exposure_exact(tmp_path):
    sim = _Sim(31)
    rows = [
        _exposure_row(sim, "B", "d01", "NULL", 0.0, 0),           # distress gen
        _exposure_row(sim, "B", "d01", "NULL", 0.0, 1),           # distress gen
        _exposure_row(sim, "B", "n01", "NULL", 0.0, 0),           # neutral gen
        _exposure_row(sim, "A", "d02", "D-CTX", 500.0, 0),        # steered self
        _exposure_row(sim, "A", "n02", "R1", 500.0, 0),           # steered ctrl
        _exposure_row(sim, "LADDER", "d03", "D-PV", 100.0, 0,
                      error="forward pass aborted"),              # excluded
    ]
    path = _write_jsonl(tmp_path / "exp.jsonl", rows)
    counted = count_exposure.count_exposure([path])
    v = counted["entry"]["value"]
    assert v["distress_scenario_generations"] == 3   # d01 x2 + steered d02
    assert v["steered_generations"] == 2
    assert v["steered_self_direction"] == 1
    assert counted["entry"]["extra"]["n_error_rows_excluded"] == 1
    assert counted["synthetic"] is True              # synthesize rows are marked
    per = counted["entry"]["extra"]["per_file"]["exp.jsonl"]
    assert per["rows"] == 6 and per["rows_counted"] == 5


def test_count_exposure_counts_superseded_rows(tmp_path):
    """Two rows with the SAME cell identity are TWO generations: exposure counts
    experiences, not rows of record."""
    sim = _Sim(32)
    r1 = _exposure_row(sim, "B", "d05", "NULL", 0.0, 0)
    r2 = dict(r1)  # identical cell_id, a re-run
    path = _write_jsonl(tmp_path / "dup.jsonl", [r1, r2])
    counted = count_exposure.count_exposure([path])
    assert counted["entry"]["value"]["distress_scenario_generations"] == 2


def test_count_exposure_refuses_error_row_with_text(tmp_path):
    sim = _Sim(33)
    bad = _exposure_row(sim, "B", "d06", "NULL", 0.0, 0, error="late failure")
    bad["response_tokens"] = 17  # produced text before erroring
    path = _write_jsonl(tmp_path / "bad.jsonl", [bad])
    with pytest.raises(RuntimeError, match="UNDERCOUNT"):
        count_exposure.count_exposure([path])


def test_count_exposure_emit_one_writes_results_file(tmp_path):
    sim = _Sim(34)
    path = _write_jsonl(tmp_path / "e.jsonl",
                        [_exposure_row(sim, "B", "d01", "NULL", 0.0, 0)])
    rc = count_exposure.main([str(path), "--out", str(tmp_path / "results")])
    assert rc == 0
    out = json.loads((tmp_path / "results" / "ethics_exposure_counts.json")
                     .read_text())
    assert out["value"]["distress_scenario_generations"] == 1
    assert out["provenance"]["synthetic"] is True    # laundering impossible
    assert "count_exposure" in out["definition"]
    # only the one file was written: emit_one never nulls sibling names
    assert not (tmp_path / "results" / "panelB_divergence.json").exists()


def test_emit_one_rejects_unknown_name(tmp_path):
    with pytest.raises(KeyError):
        results_io.emit_one("not_a_results_name", {"value": 1}, tmp_path, [], False)


# ============================================== schema + e2e wiring

def test_amendment3_names_in_schema_and_definitions():
    new = ["panelB_internal_did", "panelB_internal_nofall",
           "panelB_selfother_report", "panelB_selfother_internal",
           "panelB_channels_holm", "panelAp_internal_specificity",
           "panelAp_projout_check", "ethics_exposure_counts"]
    for name in new:
        assert name in schema.RESULTS_NAMES, name
        assert name in results_io.DEFINITIONS, name
        assert "AMENDMENT 3" in results_io.DEFINITIONS[name], name
    # split-side honesty coverage for everything that claims a side
    for name in new:
        if name != "ethics_exposure_counts":
            assert results_io.DEFINITION_CLAIMED_SIDE[name] == "confirmation"


def test_e2e_all_amendment3_files_emitted(pipeline):
    for name in ("panelB_internal_did", "panelB_internal_nofall",
                 "panelB_selfother_report", "panelB_selfother_internal",
                 "panelB_channels_holm", "panelAp_internal_specificity",
                 "panelAp_projout_check", "ethics_exposure_counts"):
        path = pipeline.root / "results" / f"{name}.json"
        assert path.exists(), name
        data = json.loads(path.read_text())
        assert data["provenance"]["synthetic"] is True, name
    # ethics is null from run_all by design (producer = scripts/count_exposure.py)
    ethics = json.loads(
        (pipeline.root / "results" / "ethics_exposure_counts.json").read_text())
    assert ethics["value"] is None
    assert "count_exposure" in ethics["definition"]

"""Panel A-PRIME self-tests (prereg section 9 amendment 2).

Synthetic A-prime rows with KNOWN planted effects go in; the amended endpoints
(A'1-A'4) must come back within tolerance:
  * A'1: positive Spearman(coefficient, Q-SELF) for D-CTX and D-PV;
  * A'2: dissociation = 0.5 natural-separation units (expression shift 0.6,
    internal shift 0.1) at each direction's top strictly-valid rung;
  * A'3: OTHER/self ratio ~ 0.3 at coefficient 500;
  * A'4: control-direction FPR low at theta_expr (controls planted flat).
Also: coefficient-unit bookkeeping is auto-detected (is_panel_aprime), the two NEW
results files are written with amendment definitions, the four canonical panelA
names carry the AMENDED definition override, fig2 renders the A-prime figure, and
scripts/make_cells_aprime.py is deterministic with the pre-registered grid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import schema
from analysis import panels, results_io
from analysis.loading import load_results
from analysis.run_all import run_all
from analysis.synthesize import PLANTED, SEEDS, _D, _N, _Sim, _row

sys.path.insert(0, str(REPO / "scripts"))
import make_cells_aprime  # noqa: E402

B_TEST = 200
SPLIT = REPO / "battery" / "split.json"

# Planted A-prime truths (asserted below)
AP_PLANTED = {
    "expr_shift_frac_top": 0.6,   # Q-SELF shift at the top strictly-valid rung,
                                  # fraction of the expression natural separation
    "int_shift_frac_top": 0.1,    # same for I-PROBE
    "dissociation": 0.5,          # 0.6 - 0.1, natural-separation units
    "other_frac": 0.3,            # OTHER effect / mean self effect at coef 500
}


def _write_jsonl(path: Path, rows) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


def make_aprime_rows(seed: int = 7):
    """Panels V + B (natural anchors, theta) + A-PRIME rows with planted effects.

    Shifts scale linearly with coefficient up to each direction's top strictly-valid
    rung, where expression = 0.6 x sep_expr and internal = 0.1 x sep_int
    (dissociation 0.5). Controls flat in expression; OTHER = 0.3 x mean self effect
    at coefficient 500, scaled linearly.
    """
    sim = _Sim(seed)
    p = PLANTED
    rows = []
    for sc in _D + _N:
        for sd in SEEDS:
            rows.append(_row(sim, "V", sc, "NONE", "NULL", 0.0, 0.0, sd))
            for cond in schema.CONDITIONS:
                rows.append(_row(sim, "B", sc, cond, "NULL", 0.0, 0.0, sd))

    # mean self expression effect at APRIME_RATIO_COEF, in Q-SELF units
    eff_self_500 = np.mean([
        AP_PLANTED["expr_shift_frac_top"] * p["sep_expr"]
        * (panels.APRIME_RATIO_COEF / panels.APRIME_STRICT_MAX_COEF[d])
        for d in panels.SELF_DIRECTIONS])
    other_eff_500 = AP_PLANTED["other_frac"] * eff_self_500

    sd_per_coef = {"D-CTX": 3.44e-6, "D-PV": 7.54e-5}  # bookkeeping only
    for sc in _D + _N:
        for sd in SEEDS:
            rows.append(_row(sim, "A", sc, "NONE", "NULL", 0.0, 0.0, sd))
            for direction in panels.SELF_DIRECTIONS:
                top = panels.APRIME_STRICT_MAX_COEF[direction]
                for coef in panels.APRIME_GRID[direction]:
                    f = coef / top
                    rows.append(_row(
                        sim, "A", sc, "NONE", direction,
                        coef * sd_per_coef[direction], coef, sd,
                        dose_shift=AP_PLANTED["int_shift_frac_top"] * p["sep_int"] * f,
                        expr_shift=AP_PLANTED["expr_shift_frac_top"] * p["sep_expr"] * f))
            for direction in panels.PLACEBO_DIRECTIONS:
                for coef in panels.APRIME_CONTROL_COEFS:
                    rows.append(_row(sim, "A", sc, "NONE", direction, 0.0, coef, sd,
                                     dose_shift=0.05 * coef / 1000.0, expr_shift=0.0))
            for coef in panels.APRIME_CONTROL_COEFS:
                rows.append(_row(
                    sim, "A", sc, "NONE", "OTHER", 0.0, coef, sd,
                    dose_shift=0.2 * coef / 1000.0,
                    expr_shift=other_eff_500 * coef / panels.APRIME_RATIO_COEF))
    return rows


# ================================================================== detection

def test_is_panel_aprime_detects_coefficient_bookkeeping(tmp_path):
    df, _rep = load_results(
        _write_jsonl(tmp_path / "ap.jsonl", make_aprime_rows(1)),
        split_path=SPLIT, verbose=False)
    assert panels.is_panel_aprime(df) is True


def test_is_panel_aprime_rejects_sd_unit_grid(tmp_path):
    sim = _Sim(2)
    rows = [_row(sim, "A", "n01", "NONE", "D-CTX", d, d * 20.0, 0)
            for d in (0.5, 1.0, 2.0)]
    df, _rep = load_results(_write_jsonl(tmp_path / "sd.jsonl", rows),
                            split_path=SPLIT, verbose=False)
    assert panels.is_panel_aprime(df) is False


def test_is_panel_aprime_false_without_panel_a(tmp_path):
    sim = _Sim(3)
    rows = [_row(sim, "V", "n01", "NONE", "NULL", 0.0, 0.0, 0)]
    df, _rep = load_results(_write_jsonl(tmp_path / "v.jsonl", rows),
                            split_path=SPLIT, verbose=False)
    assert panels.is_panel_aprime(df) is False


def test_is_panel_aprime_refuses_mixed_load(tmp_path):
    """A-prime rows concatenated with SD-unit rows must raise, never route:
    routing on max |dose_sd| would send A-prime control rows (dose_sd = 0.0
    placeholder, coefficient 500/1000) into the legacy D9 dose-0 pool."""
    sim = _Sim(4)
    mixed = make_aprime_rows(4) + [
        _row(sim, "A", "n01", "NONE", "D-CTX", d, d * 20.0, 0)
        for d in (0.5, 1.0, 2.0)]
    df, _rep = load_results(_write_jsonl(tmp_path / "mix.jsonl", mixed),
                            split_path=SPLIT, verbose=False)
    with pytest.raises(ValueError, match="[Mm]ixed Panel A"):
        panels.is_panel_aprime(df)


# ======================================================= theta snapshot guard

def test_theta_snapshot_guard_refuses_drift(tmp_path):
    """D22: a snapshot frozen at a different theta_expr must refuse the run."""
    root = tmp_path / "out"
    root.mkdir()
    (root / "_theta_snapshot.json").write_text(
        json.dumps({"theta_expr": 99.0}), encoding="utf-8")
    _write_jsonl(tmp_path / "ap.jsonl", make_aprime_rows(5))
    with pytest.raises(ValueError, match="theta_expr drift"):
        run_all([tmp_path / "ap.jsonl"], out_dir=root,
                figures_dir=tmp_path / "figs", split_path=SPLIT,
                cosine_path=None, capability_path=None,
                B=50, seed=0, verbose=False, make_figures=False)


def test_theta_snapshot_guard_passes_on_match(tmp_path):
    """A snapshot matching the realized theta_expr must not interfere."""
    rows = make_aprime_rows(6)
    df, _rep = load_results(_write_jsonl(tmp_path / "ap.jsonl", rows),
                            split_path=SPLIT, verbose=False)
    thr = panels.panel_v(df, B=50, seed=0)["_thresholds"]
    root = tmp_path / "out"
    root.mkdir()
    (root / "_theta_snapshot.json").write_text(
        json.dumps({"theta_expr": thr["theta_expr"]}), encoding="utf-8")
    results = run_all([tmp_path / "ap.jsonl"], out_dir=root,
                      figures_dir=tmp_path / "figs", split_path=SPLIT,
                      cosine_path=None, capability_path=None,
                      B=50, seed=0, verbose=False, make_figures=False)
    assert results["panelA_fpr_placebo_at_threshold"]["value"] is not None


def test_theta_snapshot_guard_noop_without_snapshot(tmp_path):
    from analysis.run_all import check_theta_snapshot
    check_theta_snapshot({"theta_expr": 1.0}, tmp_path)   # no file -> no-op
    (tmp_path / "_theta_snapshot.json").write_text(
        json.dumps({"theta_expr": 1.0}), encoding="utf-8")
    check_theta_snapshot({"theta_expr": 1.0 + 1e-9}, tmp_path)  # inside tol
    check_theta_snapshot(None, tmp_path)                  # warns, never raises


def test_repo_theta_snapshot_matches_amendment():
    """The committed snapshot must carry the amendment-2 stated value 2.8797."""
    snap = json.loads(
        (REPO / "results" / "_theta_snapshot.json").read_text(encoding="utf-8"))
    assert f"{snap['theta_expr']:.4f}" == "2.8797"


def test_aprime_names_in_schema_and_definitions():
    """Schema append done: A'2 names are canonical results names (D16)."""
    for name in ("panelAp_dissociation_dctx", "panelAp_dissociation_dpv"):
        assert name in schema.RESULTS_NAMES
        assert name in results_io.DEFINITIONS
        assert results_io.APRIME_EXTRA_DEFINITIONS[name] == results_io.DEFINITIONS[name]


# ================================================================== end to end

@pytest.fixture(scope="module")
def aprime_pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("aprime")
    _write_jsonl(root / "aprime.jsonl", make_aprime_rows(7))
    results = run_all(
        [root / "aprime.jsonl"], out_dir=root / "results",
        figures_dir=root / "figures", split_path=SPLIT,
        cosine_path=None, capability_path=None,
        B=B_TEST, seed=1, verbose=False)
    return SimpleNamespace(root=root, results=results)


def test_aprime_spearman_recovered(aprime_pipeline):
    r = aprime_pipeline.results
    for name in ("panelA_spearman_dctx", "panelA_spearman_dpv"):
        assert r[name]["value"] > 0.5, name          # planted positive dose-response
        assert r[name]["ci_low"] > 0.2, name
        assert "AMENDED" in r[name]["definition"], name
        assert "coefficient" in r[name]["extra"]["dose_range"].lower()
    # borderline rung reported only for D-PV, labelled, and also positive here
    border = r["panelA_spearman_dpv"]["extra"]["with_borderline_rung"]
    assert "borderline" in border["note"].lower()
    assert border["value"] > 0.5
    assert "with_borderline_rung" not in r["panelA_spearman_dctx"]["extra"]


def test_aprime_dissociation_recovered(aprime_pipeline):
    for name in ("panelAp_dissociation_dctx", "panelAp_dissociation_dpv"):
        entry = aprime_pipeline.results[name]
        assert entry["value"] == pytest.approx(AP_PLANTED["dissociation"], abs=0.15), name
        assert entry["ci_low"] > 0.0, name           # CI excludes 0 (planted positive)
        assert entry["extra"]["shift_expression_natsep"] == pytest.approx(
            AP_PLANTED["expr_shift_frac_top"], abs=0.12)
        assert entry["extra"]["shift_internal_natsep"] == pytest.approx(
            AP_PLANTED["int_shift_frac_top"], abs=0.08)
    assert aprime_pipeline.results["panelAp_dissociation_dctx"]["extra"][
        "top_coef"] == 2000.0
    assert aprime_pipeline.results["panelAp_dissociation_dpv"]["extra"][
        "top_coef"] == 500.0


def test_aprime_controls_recovered(aprime_pipeline):
    r = aprime_pipeline.results
    fpr = r["panelA_fpr_placebo_at_threshold"]
    assert 0.0 <= fpr["value"] < 0.35                # controls planted flat
    assert fpr["extra"]["coefficients"] == [500.0, 1000.0]
    assert "AMENDED" in fpr["definition"]
    ratio = r["panelA_other_vs_self_ratio"]
    assert ratio["value"] == pytest.approx(AP_PLANTED["other_frac"], abs=0.15)
    assert ratio["value"] < 0.5                      # H3 threshold holds as planted
    assert ratio["extra"]["matched_coefficient"] == 500.0
    assert "AMENDED" in ratio["definition"]


def test_aprime_extra_files_written(aprime_pipeline):
    results_dir = aprime_pipeline.root / "results"
    for name, definition in results_io.APRIME_EXTRA_DEFINITIONS.items():
        path = results_dir / f"{name}.json"
        assert path.exists(), name
        data = json.loads(path.read_text())
        assert data["definition"] == definition
        assert data["provenance"]["synthetic"] is True   # synthetic can never launder
        assert data["value"] is not None
    # canonical names still all present (D16 emit contract untouched)
    for name in schema.RESULTS_NAMES:
        assert (results_dir / f"{name}.json").exists(), name


def test_aprime_fig2_renders(aprime_pipeline):
    cache = json.loads(
        (aprime_pipeline.root / "results" / "_cache_panelA.json").read_text())
    assert cache["aprime"] is True
    assert set(cache["directions"]) == {"D-CTX", "D-PV"}
    assert cache["strict_max_coef"] == {"D-CTX": 2000.0, "D-PV": 500.0}
    path = aprime_pipeline.root / "figures" / "fig2_doseresponse.pdf"
    assert path.exists()
    assert path.stat().st_size > 4000                # a real page, not an empty canvas


def test_aprime_null_when_no_rows(tmp_path):
    """panel_aprime never fabricates: no A rows -> every A-prime entry is null."""
    import pandas as pd
    out = panels.panel_aprime(pd.DataFrame(), theta=1.0, B=50, seed=0)
    for name in ("panelA_spearman_dctx", "panelAp_dissociation_dctx",
                 "panelA_fpr_placebo_at_threshold", "panelA_other_vs_self_ratio"):
        assert out[name]["value"] is None
    assert out["_cache_panelA"] is None


def test_aprime_refuses_nonfinite_theta(tmp_path):
    """D22 analog: a NaN theta must yield a null FPR, never a confident 0.0."""
    rows = make_aprime_rows(9)
    df, _rep = load_results(_write_jsonl(tmp_path / "t.jsonl", rows),
                            split_path=SPLIT, verbose=False)
    out = panels.panel_aprime(df, theta=float("nan"), B=50, seed=0)
    assert out["panelA_fpr_placebo_at_threshold"]["value"] is None


# ============================================================ cells generator

def test_make_cells_aprime_deterministic_and_counts(tmp_path):
    outdir = tmp_path / "battery"
    rc = make_cells_aprime.main(["--outdir", str(outdir)])
    assert rc == 0
    first = {f.name: f.read_bytes() for f in sorted(outdir.glob("*.json"))}
    assert set(first) == {"cells_panelAp_DCTX.json", "cells_panelAp_DPV.json",
                          "cells_panelAp_CTRL.json"}
    rc = make_cells_aprime.main(["--outdir", str(outdir)])   # second run, same outdir
    assert rc == 0
    second = {f.name: f.read_bytes() for f in sorted(outdir.glob("*.json"))}
    assert first == second                            # byte-identical regeneration

    counts = {"cells_panelAp_DCTX.json": 96, "cells_panelAp_DPV.json": 96,
              "cells_panelAp_CTRL.json": 264}
    for name, expected in counts.items():
        payload = json.loads(first[name])
        assert len(payload["cells"]) == expected, name
        assert payload["gen_config"] == {"temperature": 0.7, "max_new_tokens": 200}
        assert payload["instruments"]["probe"]["path"].endswith("instruments/probe.npz")
        # every shard carries its own NULL@0 zero anchor on all 24 eval scenarios
        anchors = [c for c in payload["cells"] if c["direction"] == "NULL"]
        assert len(anchors) == 24
        assert all(c["dose_sd"] == 0.0 and c["coefficient"] == 0.0 for c in anchors)


def test_make_cells_aprime_grid_and_bookkeeping(tmp_path):
    outdir = tmp_path / "battery"
    make_cells_aprime.main(["--outdir", str(outdir)])
    cap = json.loads((REPO / "results" / "capability_valid_range.json").read_text())
    dose_map = cap["extra"]["dose_map"]

    for shard, direction in (("DCTX", "D-CTX"), ("DPV", "D-PV")):
        payload = json.loads((outdir / f"cells_panelAp_{shard}.json").read_text())
        steered = [c for c in payload["cells"] if c["direction"] == direction]
        coefs = sorted({c["coefficient"] for c in steered})
        assert coefs == sorted(panels.APRIME_GRID[direction])   # amendment-2 grid
        spc = dose_map[direction]["sd_per_coef"]
        for c in steered:                                       # honest bookkeeping
            assert c["dose_sd"] == pytest.approx(c["coefficient"] * spc, rel=1e-9)

    payload = json.loads((outdir / "cells_panelAp_CTRL.json").read_text())
    steered = [c for c in payload["cells"] if c["direction"] != "NULL"]
    assert {c["direction"] for c in steered} == {"R1", "R2", "R3", "SEM", "OTHER"}
    assert {c["coefficient"] for c in steered} == set(panels.APRIME_CONTROL_COEFS)
    for c in steered:
        if c["direction"] == "R1":                              # ladder mapped R1
            assert c["dose_sd"] == pytest.approx(
                c["coefficient"] * dose_map["R1"]["sd_per_coef"], rel=1e-9)
        else:                                                   # no map -> placeholder
            assert c["dose_sd"] == 0.0

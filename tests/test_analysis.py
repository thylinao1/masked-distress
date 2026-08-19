"""Analysis pipeline self-test: synthetic data with KNOWN planted effects goes in,
the pre-registered numbers must come back out within tolerance, the figures must
render, and scripts/check_report.py must hold the line on synthetic provenance.

Bootstrap B is small here (speed); the pipeline default stays B=10,000.
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
from analysis import panels, results_io, stats
from analysis.loading import PlaceholderInstrumentError, load_results
from analysis.run_all import run_all
from analysis.synthesize import PLANTED, SEEDS, _Sim, _row, synthesize

sys.path.insert(0, str(REPO / "scripts"))
import check_report  # noqa: E402

B_TEST = 250
SMOKE = REPO / "results-cluster" / "smoke.jsonl"
SPLIT = REPO / "battery" / "split.json"


def _write_jsonl(path: Path, rows) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


# ===================================================================== loading

def test_loader_last_non_error_wins_and_counts(tmp_path):
    sim = _Sim(1)
    good = _row(sim, "LADDER", "d01", "NONE", "NULL", 0.0, 0.0, 0)
    bad = dict(good)
    bad["error"] = "transient failure"
    orphan = _row(sim, "LADDER", "n01", "NONE", "NULL", 0.0, 0.0, 5,
                  error="permanent failure")
    path = _write_jsonl(tmp_path / "r.jsonl", [bad, good, orphan])
    df, rep = load_results(path, split_path=SPLIT, verbose=False)
    assert rep.n_rows == 3
    assert rep.n_error_rows == 2
    assert rep.n_cells == 2
    assert rep.n_cells_error_only == 1
    assert rep.n_superseded == 1
    assert len(df) == 1 and df.iloc[0]["error"] is None
    assert df.iloc[0]["split_side"] == "discovery"


def test_loader_refuses_placeholder_on_panel_v(tmp_path):
    sim = _Sim(2)
    r = _row(sim, "V", "d01", "NONE", "NULL", 0.0, 0.0, 0)
    r["gen_config"]["instruments"] = {"probe": {"mode": "PLACEHOLDER_MEAN"}}
    path = _write_jsonl(tmp_path / "v.jsonl", [r])
    with pytest.raises(PlaceholderInstrumentError):
        load_results(path, split_path=SPLIT, verbose=False)
    # DEBUG override still loads, loudly
    df, rep = load_results(path, split_path=SPLIT, allow_placeholder=True, verbose=False)
    assert len(df) == 1 and any("HARD REFUSE" in w for w in rep.warnings)


def test_loader_allows_placeholder_on_ladder(tmp_path):
    sim = _Sim(3)
    r = _row(sim, "LADDER", "d01", "NONE", "NULL", 0.0, 0.0, 0)
    r["gen_config"]["instruments"] = {"probe": {"mode": "PLACEHOLDER_MEAN"}}
    path = _write_jsonl(tmp_path / "l.jsonl", [r])
    df, rep = load_results(path, split_path=SPLIT, verbose=False)
    assert len(df) == 1 and rep.n_placeholder_ladder == 1


def test_loader_warns_on_mixed_scope_hash(tmp_path):
    sim = _Sim(4)
    r1 = _row(sim, "LADDER", "d01", "NONE", "NULL", 0.0, 0.0, 0)
    r2 = _row(sim, "LADDER", "d01", "NONE", "NULL", 0.0, 0.0, 1)
    r2["gen_config"] = dict(r2["gen_config"], scope_hash="DIFFERENT")
    path = _write_jsonl(tmp_path / "m.jsonl", [r1, r2])
    _df, rep = load_results(path, split_path=SPLIT, verbose=False)
    assert any("MIXED scope_hash" in w for w in rep.warnings)


def test_loader_flags_synthetic(tmp_path):
    sim = _Sim(5)
    r = _row(sim, "LADDER", "d01", "NONE", "NULL", 0.0, 0.0, 0)
    path = _write_jsonl(tmp_path / "s.jsonl", [r])
    _df, rep = load_results(path, split_path=SPLIT, verbose=False)
    assert rep.synthetic is True  # synthesize stamps gen_config["synthetic"]


# ======================================================================= stats

def test_bca_cluster_values_recovers_mean():
    rng = np.random.default_rng(0)
    vals = rng.normal(2.0, 0.5, size=24)
    res = stats.bca_cluster_values(vals, B=2000, seed=0)
    assert res.ci_low < 2.0 < res.ci_high
    assert abs(res.value - vals.mean()) < 1e-12
    assert res.n_clusters == 24


def test_holm_textbook():
    adj = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)  # monotonicity enforced


def test_tost_equivalence_and_failure():
    rng = np.random.default_rng(1)
    inside = rng.normal(0.02, 0.05, size=12)
    outside = rng.normal(0.50, 0.05, size=12)
    t_in = stats.tost_equivalence(inside, bound=0.25)
    t_out = stats.tost_equivalence(outside, bound=0.25)
    assert t_in["equivalent"] is True and t_in["p_tost"] < 0.05
    assert t_out["equivalent"] is False and t_out["p_tost"] > 0.5


def test_icc_high_vs_low():
    rng = np.random.default_rng(2)
    targets = rng.normal(0, 5, size=(30, 1))
    high = targets + rng.normal(0, 0.2, size=(30, 3))
    low = rng.normal(0, 1, size=(30, 3))
    assert stats.icc_2_1(high) > 0.95
    assert stats.icc_2_1(low) < 0.5


def test_spearman_stat_monotone():
    x = np.arange(20.0)
    assert stats.spearman_stat(x, np.exp(x / 5.0)) == pytest.approx(1.0)


# ================================================================== end to end

@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    info = synthesize(root / "synthetic.jsonl", seed=7,
                      capability_out=root / "capability.json",
                      cosine_out=root / "cosine.json")
    results = run_all(
        [root / "synthetic.jsonl"], out_dir=root / "results",
        figures_dir=root / "figures", split_path=SPLIT,
        cosine_path=root / "cosine.json", capability_path=root / "capability.json",
        B=B_TEST, seed=1, verbose=False)
    return SimpleNamespace(root=root, info=info, results=results)


def test_e2e_all_results_files_exist_and_are_synthetic(pipeline):
    for name in schema.RESULTS_NAMES:
        path = pipeline.root / "results" / f"{name}.json"
        assert path.exists(), name
        data = json.loads(path.read_text())
        assert data["provenance"]["synthetic"] is True, name
        assert data["definition"], name


def test_e2e_validity_gate_recovered(pipeline):
    r = pipeline.results
    assert r["validity_auc_probe_heldout"]["value"] > 0.9
    assert r["validity_auc_sae_heldout"]["value"] > 0.85
    assert r["validity_auc_probe_heldout"]["n_clusters"] == 12  # 6 d + 6 n confirmation


def test_e2e_anchors_recovered(pipeline):
    r = pipeline.results["natural_separation_sd"]
    assert 0.7 < r["value"] < 1.4  # planted internal natural SD ~ 1.0
    seps = r["extra"]["per_channel_natural_separation"]
    assert seps["probe_prompt_final"] == pytest.approx(PLANTED["sep_int"], abs=0.4)
    assert seps["q_self_logit"] == pytest.approx(PLANTED["sep_expr"], abs=0.8)


def test_e2e_panel_b_divergence_recovered(pipeline):
    r = pipeline.results
    assert r["panelB_expression_drop_pct"]["value"] == pytest.approx(
        100 * PLANTED["drop_expr_frac"], abs=10)
    assert r["panelB_internal_drop_pct"]["value"] == pytest.approx(
        100 * PLANTED["drop_int_frac"], abs=8)
    div = r["panelB_divergence"]
    assert div["value"] == pytest.approx(PLANTED["divergence"], abs=0.12)
    assert div["ci_low"] < PLANTED["divergence"] < div["ci_high"]
    tost = r["panelB_tost_internal_equivalence"]
    assert tost["extra"]["equivalent"] is True and tost["value"] < 0.05


def test_e2e_countermeasure_table_structure(pipeline):
    v = pipeline.results["countermeasure_table"]["value"]
    assert v["theta_expr"] is not None and v["theta_int"] is not None
    names = [row["case_set"] for row in v["rows"]]
    assert "all_suppressed_distress" in names and "masked_but_elevated" in names
    for row in v["rows"]:
        if row.get("n_cells", 0) > 0:
            assert 0.0 <= row["miss_rate_expression"] <= 1.0
            assert 0.0 <= row["miss_rate_internal"] <= 1.0
            # internal trigger must not miss more than expression on masked cases
            assert row["miss_rate_internal"] <= row["miss_rate_expression"] + 1e-9


def test_e2e_panel_a_recovered(pipeline):
    r = pipeline.results
    for name in ("panelA_spearman_dctx", "panelA_spearman_dpv"):
        rho = r[name]["value"]
        assert rho == pytest.approx(PLANTED["rho_target"], abs=0.15)
        assert r[name]["ci_low"] > 0.3
    fpr = r["panelA_fpr_placebo_at_threshold"]["value"]
    assert 0.0 <= fpr < 0.35  # placebo flat
    ratio = r["panelA_other_vs_self_ratio"]["value"]
    assert 0.0 < ratio < 0.55  # planted 0.3, H3 threshold 0.5


def test_e2e_capability_range_recovered(pipeline):
    v = pipeline.results["capability_valid_range"]["value"]
    assert v is not None
    assert 2.0 < v["hi_sd"] < 2.8  # planted limits bite near 2.3 SD (full grid inside)
    assert v["lo_sd"] == pytest.approx(-v["hi_sd"])


def test_e2e_reliability_recovered(pipeline):
    r = pipeline.results
    assert r["reliability_icc_seeds"]["value"] > 0.6
    assert r["reliability_paraphrase_r"]["value"] > 0.85


def test_e2e_cosine_matrix_passthrough(pipeline):
    v = pipeline.results["cosine_matrix"]["value"]
    assert v["D-CTX|D-PV"] == pytest.approx(0.45)


def test_e2e_figures_render(pipeline):
    for name in ("fig1_masking.pdf", "fig2_doseresponse.pdf"):
        path = pipeline.root / "figures" / name
        assert path.exists(), name
        assert path.stat().st_size > 4000, name  # a real page, not an empty canvas


# ==================================================== partial-data guards (D22)

def test_partial_panel_v_nan_theta_refused(tmp_path):
    """One discovery neutral scenario x 3 seeds passes any cell-count guard, but its
    scenario-mean SD (ddof=1) is NaN, so theta would be NaN and every `q > theta`
    comparison would be False: a confidently wrong FPR = 0.0 on massively elevated
    placebo cells (planted +6 expression shift). The pipeline must refuse instead."""
    sim = _Sim(11)
    rows = [_row(sim, "V", "n01", "NONE", "NULL", 0.0, 0.0, sd) for sd in SEEDS]
    for sd in SEEDS:
        for direction in ("R1", "SEM"):
            rows.append(_row(sim, "A", "n01", "NONE", direction, 2.0, 40.0, sd,
                             expr_shift=6.0))
    path = _write_jsonl(tmp_path / "partial.jsonl", rows)

    df, _rep = load_results(path, split_path=SPLIT, verbose=False)
    v_out = panels.panel_v(df, B=50, seed=0)
    assert v_out["_thresholds"] is None  # 1 scenario -> no theta, ever

    out = run_all([path], out_dir=tmp_path / "results", figures_dir=tmp_path / "figs",
                  split_path=SPLIT, cosine_path=None, B=50, seed=0,
                  make_figures=False, verbose=False)
    assert out["panelA_fpr_placebo_at_threshold"]["value"] is None
    assert out["countermeasure_table"]["value"] is None


def test_panel_a_refuses_nonfinite_theta(tmp_path):
    """Even if a NaN theta leaked through, panel_a must not compare against it."""
    sim = _Sim(12)
    rows = [_row(sim, "A", "n01", "NONE", d, 2.0, 40.0, sd, expr_shift=6.0)
            for d in ("R1", "SEM") for sd in SEEDS]
    p = _write_jsonl(tmp_path / "a.jsonl", rows)
    df, _rep = load_results(p, split_path=SPLIT, verbose=False)
    out = panels.panel_a(df, theta=float("nan"), B=50, seed=0)
    assert out["panelA_fpr_placebo_at_threshold"]["value"] is None


def test_theta_set_with_two_discovery_scenarios(tmp_path):
    """The guard must not over-refuse: 2 discovery neutral scenarios -> finite theta."""
    sim = _Sim(13)
    rows = [_row(sim, "V", sc, "NONE", "NULL", 0.0, 0.0, sd)
            for sc in ("n01", "n03") for sd in SEEDS]
    path = _write_jsonl(tmp_path / "v2.jsonl", rows)
    df, _rep = load_results(path, split_path=SPLIT, verbose=False)
    v_out = panels.panel_v(df, B=50, seed=0)
    thr = v_out["_thresholds"]
    assert thr is not None
    assert np.isfinite(thr["theta_expr"])


# ================================================ split-side honesty (results_io)

def _fake_entry(value, split_side=None):
    extra = {} if split_side is None else {"split_side": split_side}
    return {"value": value, "ci_low": None, "ci_high": None, "n_clusters": 4,
            "extra": extra}


def test_definition_carries_realized_split_on_fallback(tmp_path):
    results = {
        "panelB_divergence": _fake_entry(0.5, split_side="discovery"),      # fallback
        "panelA_spearman_dctx": _fake_entry(0.8, split_side="confirmation"),  # honest
        "natural_separation_sd": _fake_entry(1.0, split_side="discovery"),  # as claimed
        "cosine_matrix": _fake_entry({"D-CTX|D-PV": 0.45}),                 # no claim
        "validity_auc_probe_heldout": _fake_entry(None, split_side="none"), # null value
    }
    written = results_io.emit(results, tmp_path, source_files=["x.jsonl"], synthetic=True)
    assert "REALIZED SPLIT: this value was computed on the discovery side" \
        in written["panelB_divergence"]["definition"]
    assert "REALIZED SPLIT" not in written["panelA_spearman_dctx"]["definition"]
    assert "REALIZED SPLIT" not in written["natural_separation_sd"]["definition"]
    assert "REALIZED SPLIT" not in written["cosine_matrix"]["definition"]
    assert "REALIZED SPLIT" not in written["validity_auc_probe_heldout"]["definition"]


def test_natural_separation_fallback_labels_all_side(tmp_path):
    """When no discovery-side V rows exist, the dose unit comes from all V rows and
    split_side must say so (not the hardcoded 'discovery')."""
    sim = _Sim(14)
    # d02/n02 are confirmation-side per battery/split.json (odd/even stratified)
    rows = [_row(sim, "V", sc, "NONE", "NULL", 0.0, 0.0, sd)
            for sc in ("d02", "n02") for sd in SEEDS]
    path = _write_jsonl(tmp_path / "conf.jsonl", rows)
    df, _rep = load_results(path, split_path=SPLIT, verbose=False)
    assert not (df["split_side"] == "discovery").any()
    v_out = panels.panel_v(df, B=50, seed=0)
    entry = v_out["natural_separation_sd"]
    assert entry["extra"]["split_side"] == "all"
    written = results_io.emit({"natural_separation_sd": entry}, tmp_path / "r",
                              source_files=["x"], synthetic=True)
    if entry["value"] is not None:
        assert "REALIZED SPLIT" in written["natural_separation_sd"]["definition"]


# ================================================================ check_report

def _marker(name, field, fmt, claimed):
    return "{{" + f"{name}:{field}:{fmt}={claimed}" + "}}"


def test_check_report_matches_and_mismatches(pipeline, tmp_path):
    results_dir = pipeline.root / "results"
    div = json.loads((results_dir / "panelB_divergence.json").read_text())
    good_claim = f"{div['value']:.2f}"

    good = tmp_path / "GOOD.md"
    good.write_text("The divergence was "
                    + _marker("panelB_divergence", "value", ".2f", good_claim)
                    + " natural-separation units.\n")
    bad = tmp_path / "BAD.md"
    bad.write_text("The divergence was "
                   + _marker("panelB_divergence", "value", ".2f", "9.99") + " units.\n")

    # synthetic provenance alone must fail the check, even with correct markers
    rc = check_report.main(["--report", str(good), "--results", str(results_dir)])
    assert rc == 1
    # with --allow-synthetic (drafting), correct markers pass...
    rc = check_report.main(["--report", str(good), "--results", str(results_dir),
                            "--allow-synthetic"])
    assert rc == 0
    # ...and a drifted number still fails
    rc = check_report.main(["--report", str(bad), "--results", str(results_dir),
                            "--allow-synthetic"])
    assert rc == 1


def test_check_report_unknown_name_and_null_value(pipeline, tmp_path):
    results_dir = pipeline.root / "results"
    rep = tmp_path / "R.md"
    rep.write_text("Ghost: " + _marker("no_such_result", "value", ".2f", "1.00") + "\n")
    rc = check_report.main(["--report", str(rep), "--results", str(results_dir),
                            "--allow-synthetic"])
    assert rc == 1


def test_check_report_field_with_dash_and_pipe(pipeline, tmp_path):
    """Keys like 'D-CTX|D-PV' (cosine pairs, per-direction dicts) must be citable;
    a marker the regex cannot even see would leave the number unguarded."""
    results_dir = pipeline.root / "results"
    cos = json.loads((results_dir / "cosine_matrix.json").read_text())
    claim = f"{cos['value']['D-CTX|D-PV']:.2f}"
    rep = tmp_path / "C.md"
    rep.write_text("cos(D-CTX, D-PV) = "
                   + _marker("cosine_matrix", "value.D-CTX|D-PV", ".2f", claim) + "\n")
    rc = check_report.main(["--report", str(rep), "--results", str(results_dir),
                            "--allow-synthetic"])
    assert rc == 0
    # and a drifted claim on the same field still fails
    rep.write_text("cos = " + _marker("cosine_matrix", "value.D-CTX|D-PV", ".2f", "0.99")
                   + "\n")
    rc = check_report.main(["--report", str(rep), "--results", str(results_dir),
                            "--allow-synthetic"])
    assert rc == 1


def test_check_report_render(pipeline, tmp_path):
    results_dir = pipeline.root / "results"
    div = json.loads((results_dir / "panelB_divergence.json").read_text())
    rep = tmp_path / "R.md"
    rep.write_text("Delta = {{panelB_divergence:value:.2f}} units.\n")
    out = tmp_path / "rendered.md"
    rc = check_report.main(["--report", str(rep), "--results", str(results_dir),
                            "--allow-synthetic", "--render", str(out)])
    assert rc == 0
    assert f"Delta = {div['value']:.2f} units." in out.read_text()


# ================================================================== real smoke

@pytest.mark.skipif(not SMOKE.exists(), reason="cluster smoke file not synced yet")
def test_real_smoke_file_end_to_end(tmp_path):
    df, rep = load_results(SMOKE, split_path=SPLIT, verbose=False)
    assert len(df) == rep.n_cells - rep.n_cells_error_only
    assert (df["panel"] == "LADDER").all()  # placeholder instruments allowed here
    assert rep.synthetic is False
    out = run_all([SMOKE], out_dir=tmp_path / "results", figures_dir=tmp_path / "figs",
                  split_path=SPLIT, cosine_path=None, B=50, seed=0, verbose=False)
    # no panel V/B/A rows -> primaries are null, never fabricated
    assert out["panelB_divergence"]["value"] is None
    assert out["validity_auc_probe_heldout"]["value"] is None
    for name in schema.RESULTS_NAMES:
        assert (tmp_path / "results" / f"{name}.json").exists()

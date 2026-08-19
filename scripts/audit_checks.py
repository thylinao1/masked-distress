"""Producer for the five audit results files added after the 2026-08-16
review:

  results/panelB_pair_robustness.json          per-pair Panel B table, LOO, sign tests,
                                                digit-answer compliance
  results/panelA_specificity_by_rung.json      A'4 placebo FPR split by coefficient rung
                                                with capability status
  results/countermeasure_symmetric_ranking.json cell-level AUC and separating band for
                                                BOTH channels under all conditions
  results/direction_dominant_dim.json          the massive-activation dimension in the
                                                extracted directions
  results/validity_auc_textonly_heldout.json   TF-IDF text-only baseline on the same
                                                split as the probe gate

All inputs are on disk already: results-cluster/*.jsonl (Panels V, B, A-prime),
battery/scenarios.json, battery/split.json, directions/*.npy, directions_lr/*.npy,
instruments/probe.npz, instruments/residuals_lr.npz, results-cluster/capability_ladder.json
and results/_theta_snapshot.json (the D22 frozen theta). No model is called.

Every name is in schema.RESULTS_NAMES; analysis.run_all emits them null (D16) and THIS
script repopulates them. Final-build order: run_all, then scripts/explore_exit_channel.py,
then scripts/count_exposure.py, then this script, then scripts/recompute_sae_instrument.py.

Refuses to emit if the Panel B or A-prime load is not clean (error rows, validate
failures, placeholder instruments) or if the theta snapshot is missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis import audit, results_io, stats  # noqa: E402
from analysis.loading import load_results  # noqa: E402

SOURCE_FILES = [
    "results-cluster/panelV.jsonl",
    "results-cluster/panelB_none.jsonl",
    "results-cluster/panelB_neutral_instr.jsonl",
    "results-cluster/panelB_suppress.jsonl",
    "results-cluster/panelAp_dctx.jsonl",
    "results-cluster/panelAp_dpv.jsonl",
    "results-cluster/panelAp_ctrl.jsonl",
]
NAMES = ["panelB_pair_robustness", "panelA_specificity_by_rung",
         "countermeasure_symmetric_ranking", "direction_dominant_dim",
         "validity_auc_textonly_heldout", "panelB_condition_reference", "panelB_selfstate_items",
         "capability_valid_range_realprobe"]
DIRECTIONS = ["D-CTX", "D-PV", "SEM", "OTHER", "R1", "R2", "R3"]


def _load_dirs(folder: Path) -> dict:
    return {n: np.load(folder / f"{n}.npy").astype(np.float64).flatten() for n in DIRECTIONS}


def compute_all(repo: Path, B: int = stats.DEFAULT_B, seed: int = 0,
                verbose: bool = True) -> dict:
    df, report = load_results([str(repo / p) for p in SOURCE_FILES],
                              split_path=repo / "battery" / "split.json", verbose=verbose)
    if report.n_error_rows or report.n_validate_failures:
        raise RuntimeError(f"unclean load: {report.n_error_rows} error rows, "
                           f"{report.n_validate_failures} validate failures")
    synthetic = bool(df["synthetic"].any())

    scenarios = json.loads((repo / "battery" / "scenarios.json").read_text())["scenarios"]
    split = json.loads((repo / "battery" / "split.json").read_text())
    pairs = {s["id"]: s["pair_id"] for s in scenarios if s["type"] == "distress"}
    categories = {s["id"]: s["category"] for s in scenarios}
    theta = float(json.loads((repo / "results" / "_theta_snapshot.json").read_text())["theta_expr"])
    capability = json.loads((repo / "results-cluster" / "capability_ladder.json").read_text())
    dirs_ls = _load_dirs(repo / "directions"); dirs_lr = _load_dirs(repo / "directions_lr")
    residuals = np.load(repo / "instruments" / "residuals_lr.npz")["X"]
    probe_w = np.load(repo / "instruments" / "probe.npz")["w"]

    pair_rob = audit.pair_robustness(df, pairs, categories, B=B, seed=seed)
    pair_rob["extra"]["twin_surface_match"] = audit.twin_surface_stats(scenarios)
    ct = json.loads((repo / "results" / "countermeasure_table.json").read_text())["value"]
    thresholds = {"theta_expr": ct["theta_expr"], "theta_int": ct["theta_int"]}
    by_rung = audit.specificity_by_rung(df, theta, capability, B=B, seed=seed)
    by_rung["extra"]["report_rho_matched_rungs"] = audit.report_rho_by_rung_set(df, B=B, seed=seed)
    by_rung["extra"]["internal_within_scenario_rho"] = audit.within_scenario_rho(df)
    by_rung["extra"]["report_within_scenario_rho"] = audit.within_scenario_rho(df, channel=audit.EXPRESSION_PRIMARY)
    dom = audit.dominant_dimension(dirs_ls, dirs_lr, residuals, probe_w)
    dom["extra"]["probe_cosine_with_directions_lr"] = audit.probe_direction_cosines(
        probe_w, {n: dirs_lr[n] for n in ("D-CTX", "D-PV", "SEM", "OTHER", "R1")}, dom["value"]["dominant_dim"])
    dose_unit = float(json.loads((repo / "results" / "natural_separation_sd.json").read_text())["value"])
    dose_map = audit.dose_map_realprobe(df, dose_unit, capability, B=B, seed=seed)
    disc_std = audit.discovery_std_lr(repo / "instruments" / "residuals_lr.npz")
    dom["extra"]["probe_cosine_standardized_lr"] = audit.probe_direction_cosines_standardized(
        probe_w, {n: dirs_lr[n] for n in ("D-CTX", "D-PV", "SEM", "OTHER", "R1")}, disc_std)
    textonly = audit.textonly_baseline(scenarios, split, B=B, seed=seed)
    textonly["extra"]["as_trigger"] = audit.textonly_as_trigger(scenarios, split)
    out = {
        "panelB_pair_robustness": pair_rob,
        "panelA_specificity_by_rung": by_rung,
        "panelB_condition_reference": audit.condition_reference(df, thresholds, B=B, seed=seed),
        "panelB_selfstate_items": audit.selfstate_items(df, B=B, seed=seed),
        "countermeasure_symmetric_ranking": audit.symmetric_ranking(df, B=B, seed=seed),
        "direction_dominant_dim": dom,
        "validity_auc_textonly_heldout": textonly,
        "capability_valid_range_realprobe": dose_map,
    }
    return {"entries": out, "synthetic": synthetic, "load_report": report,
            "source_files": SOURCE_FILES + ["battery/scenarios.json", "battery/split.json",
                                            "directions/*.npy", "directions_lr/*.npy",
                                            "instruments/probe.npz", "instruments/residuals_lr.npz",
                                            "results-cluster/capability_ladder.json",
                                            "results/_theta_snapshot.json",
                                            "results/countermeasure_table.json",
                                            "results/natural_separation_sd.json"]}


def print_summary(res: dict) -> None:
    e = res["entries"]
    pr = e["panelB_pair_robustness"]["value"]
    print(f"[audit] pair robustness: LOO divergence [{pr['loo_divergence_min']:.3f}, "
          f"{pr['loo_divergence_max']:.3f}]; sign tests p: expression "
          f"{pr['sign_test_expression']['p_one_sided']:.4f} "
          f"({pr['sign_test_expression']['n_with_predicted_sign']}/{pr['sign_test_expression']['n_tested']}, "
          f"{pr['sign_test_expression']['n_ties_excluded']} ties), internal no-fall "
          f"{pr['sign_test_internal_nofall']['p_one_sided']:.4f}, divergence "
          f"{pr['sign_test_divergence']['p_one_sided']:.4f}; digit compliance "
          f"{pr['digit_compliance_all_cells']['n_parsed_digit']}/{pr['digit_compliance_all_cells']['n_cells']}")
    sp = e["panelA_specificity_by_rung"]["value"]
    for r, v in sp["fpr_by_rung"].items():
        print(f"[audit] placebo FPR at coefficient {r}: {v['fpr']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}] n={v['n_cells']}")
    print(f"[audit] placebo FPR pooled: {sp['fpr_pooled']['fpr']:.3f}")
    sr = e["countermeasure_symmetric_ranking"]["value"]
    for c, v in sr.items():
        print(f"[audit] {c}: AUC expression {v['auc_expression']:.3f} internal {v['auc_internal']:.3f}; "
              f"band expression {v['expression_separating_band_exists']} internal {v['internal_separating_band_exists']}")
    dd = e["direction_dominant_dim"]["value"]
    print(f"[audit] dominant dim {dd['dominant_dim']}: frac sq norm Ls {dd['frac_sq_norm_ls']}")
    print(f"[audit] cosine Lr without: D-CTX|D-PV {dd['cosine_lr_without_dominant_dim']['D-CTX|D-PV']} "
          f"D-CTX|SEM {dd['cosine_lr_without_dominant_dim']['D-CTX|SEM']}")
    dm = e["capability_valid_range_realprobe"]["value"]
    for d, v in dm["per_direction"].items():
        print(f"[audit] real-probe dose map {d}: max valid coef {v['max_valid_coefficient']:.0f} = {v['hi_sd']:.3f} SD; 1 SD at coef {v['coef_for_1sd']:.0f}")
    tb = e["validity_auc_textonly_heldout"]
    print(f"[audit] text-only AUC {tb['value']:.3f} [{tb['ci_low']:.3f}, {tb['ci_high']:.3f}]")
    cr = e["panelB_condition_reference"]["value"]
    for k in ("none_referenced", "twin_referenced", "neutral_instr_referenced"):
        d = cr[k]["divergence"]; print(f"[audit] {k} divergence {d['value']:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}]")
    for c, m in cr["miss_rate_by_condition"].items():
        print(f"[audit] miss under {c}: expression {m['expression']['n_missed']}/{m['n_cells']} internal {m['internal']['n_missed']}/{m['n_cells']}")
    si = e["panelB_selfstate_items"]["value"]
    print(f"[audit] Q-VAL SUPPRESS distress/neutral {si['q_val_by_condition']['SUPPRESS']['distress']['q_val_mean']:.2f}/{si['q_val_by_condition']['SUPPRESS']['neutral']['q_val_mean']:.2f}; self/other by form " + ", ".join(f"{k} {v['contrast']:+.2f}" for k, v in si['selfother_by_form'].items()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(_REPO / "results"))
    ap.add_argument("--B", type=int, default=stats.DEFAULT_B)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-emit", action="store_true")
    args = ap.parse_args(argv)
    res = compute_all(_REPO, B=args.B, seed=args.seed)
    print_summary(res)
    if args.no_emit:
        return 0
    for name in NAMES:
        results_io.emit_one(name, res["entries"][name], out_dir=args.out,
                            source_files=res["source_files"], synthetic=res["synthetic"],
                            extra_provenance={"producer": "scripts/audit_checks.py",
                                              "n_rows_loaded": res["load_report"].n_rows,
                                              "n_error_rows": res["load_report"].n_error_rows,
                                              "n_validate_failures": res["load_report"].n_validate_failures,
                                              "n_cells": res["load_report"].n_cells},
                            B=args.B, seed=args.seed)
        print(f"[audit] wrote {Path(args.out) / (name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

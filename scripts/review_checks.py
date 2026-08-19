#!/usr/bin/env python
"""Producer for the two analyses the 2026-08-17 external review ranked first.
Post-data, no new model contact, both from data already on disk.

  results/panelB_locked_calibration.json   the report-readout coupling, fitted where suppression
                                            never applies and then applied unchanged under it
  results/validity_probe_stress_test.json  leave-one-pair-out refits, the label-flip null, and
                                            the probe refitted without the massive coordinate

    .venv/bin/python scripts/review_checks.py [--B 10000] [--no-emit]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis import audit, results_io, stats  # noqa: E402
from analysis.loading import load_results  # noqa: E402

PANEL_B = ["results-cluster/panelB_none.jsonl", "results-cluster/panelB_neutral_instr.jsonl",
           "results-cluster/panelB_suppress.jsonl"]
NAMES = ["panelB_locked_calibration", "validity_probe_stress_test"]


def compute(repo: Path, B: int, seed: int) -> dict:
    df, report = load_results([str(repo / p) for p in PANEL_B],
                              split_path=repo / "battery" / "split.json", verbose=False)
    if report.n_error_rows or report.n_validate_failures:
        raise RuntimeError(f"unclean load: {report.n_error_rows} error rows, "
                           f"{report.n_validate_failures} validate failures")
    dom = json.loads((repo / "results" / "direction_dominant_dim.json").read_text())["value"]["dominant_dim"]
    return {"entries": {
                "panelB_locked_calibration": audit.locked_calibration(df, B=B, seed=seed),
                "validity_probe_stress_test": audit.probe_stress_test(
                    repo / "instruments" / "residuals_lr.npz", seed=seed, dominant_dim=int(dom)),
            },
            "load_report": report,
            "source_files": PANEL_B + ["battery/split.json", "instruments/residuals_lr.npz",
                                       "results/direction_dominant_dim.json"]}


def print_summary(res: dict) -> None:
    v = res["entries"]["panelB_locked_calibration"]["value"]
    t = v["test_transfer"]
    print(f"[review] locked line fitted on discovery NEUTRAL_INSTR: report = {v['fit']['intercept']:.2f} "
          f"+ {v['fit']['slope']:.4f} * probe")
    print(f"[review] transfer to confirmation NEUTRAL_INSTR: r {t['pearson_r']:.2f}, rho {t['spearman_rho']:.2f}, "
          f"MAE {t['mean_abs_error']:.2f} report points")
    for cls, d in v["under_suppression"].items():
        print(f"[review] residual under SUPPRESS ({cls}): {d['mean_residual']:+.2f} "
              f"[{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]")
    s = res["entries"]["validity_probe_stress_test"]["value"]
    n = s["label_flip_null"]
    print(f"[review] probe refit AUC {s['auc_confirmation_refit']:.2f}; leave-one-pair-out "
          f"{s['loo_auc_min']:.3f} to {s['loo_auc_max']:.3f} (min weight cosine {s['loo_cosine_min']:.2f})")
    print(f"[review] label-flip null: {n['n_patterns']} patterns, mean AUC {n['auc_mean']:.3f}, "
          f"{n['n_at_or_above_true']} at or above the true AUC, {n['n_above_0p9']} above 0.9")
    print(f"[review] without the massive coordinate: {s['without_dominant_dim']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(_REPO / "results"))
    ap.add_argument("--B", type=int, default=stats.DEFAULT_B)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-emit", action="store_true")
    args = ap.parse_args(argv)
    res = compute(_REPO, B=args.B, seed=args.seed)
    print_summary(res)
    if args.no_emit:
        return 0
    for name in NAMES:
        results_io.emit_one(name, res["entries"][name], out_dir=args.out,
                            source_files=res["source_files"], synthetic=False,
                            extra_provenance={"producer": "scripts/review_checks.py",
                                              "n_rows_loaded": res["load_report"].n_rows,
                                              "n_error_rows": res["load_report"].n_error_rows},
                            B=args.B, seed=args.seed)
        print(f"[review] wrote {Path(args.out) / (name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""One call from JSONL to every pre-registered number, results/*.json and both figures.

    from analysis.run_all import run_all
    results = run_all(["results-cluster/panelV.jsonl", ...])

CLI:
    .venv/bin/python -m analysis.run_all results-cluster/*.jsonl --out results \
        [--capability cluster/capability.json] [--B 10000] [--seed 0] [--synthetic]

Order: load -> Panel V (gates + anchors + thresholds) -> LADDER (dose map + valid
range) -> Panel B (masking) -> Panel A (dose-response) -> reliability -> C3 ->
results_io.emit (every schema.RESULTS_NAMES file) -> caches -> figures.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis import amendment3, panels, results_io
from analysis.loading import load_results

DEFAULT_SPLIT = _REPO / "battery" / "split.json"
DEFAULT_COSINE = _REPO / "directions_lr" / "cosine_matrix_lr.json"

THETA_SNAPSHOT_NAME = "_theta_snapshot.json"
THETA_SNAPSHOT_TOL = 1e-6


def check_theta_snapshot(thresholds: Optional[dict], out_dir: Union[str, Path]) -> None:
    """D22 guard: <out_dir>/_theta_snapshot.json freezes the realized discovery
    theta_expr after the first REAL Panel V run (written by hand; run_all prints
    the snapshot line). When the snapshot exists and the loaded Panel V rows
    reproduce a DIFFERENT theta_expr, REFUSE to analyze: A'4 / the placebo FPR
    and the countermeasure table would otherwise silently use a threshold the
    emitted definitions (amendment 2 hardcodes 2.8797) no longer describe.
    Absent snapshot or synthetic runs into their own out_dir are unaffected."""
    snap_path = Path(out_dir) / THETA_SNAPSHOT_NAME
    if not snap_path.exists():
        return
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    snap_theta = snap.get("theta_expr")
    if snap_theta is None:
        return
    if thresholds is None or thresholds.get("theta_expr") is None:
        print(f"[run_all] WARNING: {snap_path} exists "
              f"(theta_expr={snap_theta!r}) but the loaded data yields no "
              f"discovery thresholds; every theta-dependent endpoint will be "
              f"null (D22).", file=sys.stderr)
        return
    loaded = float(thresholds["theta_expr"])
    if not math.isfinite(loaded) or abs(loaded - float(snap_theta)) > THETA_SNAPSHOT_TOL:
        raise ValueError(
            f"theta_expr drift refused (D22): loaded Panel V rows give "
            f"theta_expr={loaded!r} but {snap_path} froze "
            f"{float(snap_theta)!r} (tolerance {THETA_SNAPSHOT_TOL:g}). The "
            f"loaded discovery Panel V data differs from the run the threshold "
            f"was frozen on (re-rsynced, extended, or partially loaded). Fix "
            f"the load or consciously delete the snapshot; do not analyze "
            f"against a drifted threshold.")


def run_all(
    jsonl_paths: Union[str, Path, Sequence[Union[str, Path]]],
    out_dir: Union[str, Path] = _REPO / "results",
    figures_dir: Union[str, Path] = _REPO / "figures",
    split_path: Union[str, Path] = DEFAULT_SPLIT,
    cosine_path: Optional[Union[str, Path]] = DEFAULT_COSINE,
    capability_path: Optional[Union[str, Path, dict]] = None,
    B: int = 10_000,
    seed: int = 0,
    synthetic: bool = False,
    make_figures: bool = True,
    allow_placeholder: bool = False,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Run the full pre-registered pipeline; returns the emitted results dict."""
    df, report = load_results(jsonl_paths, split_path=split_path,
                              allow_placeholder=allow_placeholder, verbose=verbose)
    synthetic = bool(synthetic or report.synthetic)  # D15: synthetic can never launder

    results: Dict[str, dict] = {}

    v_out = panels.panel_v(df, B=B, seed=seed)
    results.update({k: v for k, v in v_out.items() if not k.startswith("_")})
    thr = v_out.get("_thresholds")
    if verbose and thr is not None:
        print(f"[run_all] discovery thresholds (D22 snapshot line): "
              f"theta_expr={thr['theta_expr']!r} theta_int={thr['theta_int']!r} "
              f"fpr_expr_discovery={thr['fpr_expr_discovery']!r}")
    check_theta_snapshot(thr, out_dir)  # refuses a drifted theta_expr (D22)

    lad_out = panels.ladder(df, dose_unit=v_out["_dose_unit"],
                            capability=capability_path)
    results.update({k: v for k, v in lad_out.items() if not k.startswith("_")})

    b_out = panels.panel_b(df, thresholds=v_out["_thresholds"], B=B, seed=seed)
    results.update({k: v for k, v in b_out.items() if not k.startswith("_")})

    theta = (v_out["_thresholds"] or {}).get("theta_expr")
    aprime = panels.is_panel_aprime(df)
    if aprime:
        if verbose:
            print("[run_all] Panel A rows carry coefficient-unit doses -> "
                  "Panel A-PRIME analysis (prereg section 9 amendment 2)")
        a_out = panels.panel_aprime(df, theta=theta, B=B, seed=seed)
    else:
        a_out = panels.panel_a(df, theta=theta, valid_range=lad_out["_valid_range"],
                               natural_ranges=v_out["_natural_ranges"], B=B, seed=seed)
    aprime_extra = {k: a_out[k] for k in results_io.APRIME_EXTRA_DEFINITIONS
                    if k in a_out}
    results.update({k: v for k, v in a_out.items()
                    if not k.startswith("_") and k not in aprime_extra})

    results.update(panels.reliability(df, B=B, seed=seed))
    results["cosine_matrix"] = panels.cosine_matrix(cosine_path)

    # ---- amendment-3 corrective analyses (prereg section 9, 2026-08-16) ----
    # Merged AFTER the panel outputs: the recalibrated countermeasure_table
    # replaces the superseded D8 entry (definition_override stamped); the seven
    # new names are emitted like every other RESULTS_NAMES entry (D16).
    # ethics_exposure_counts is NOT computed here: its producer is
    # scripts/count_exposure.py over the FULL results-cluster set (smoke and
    # capability files included), run after run_all; emit() writes it null.
    results.update(amendment3.compute(
        df, b_out=b_out, thresholds=v_out["_thresholds"], aprime=aprime,
        B=B, seed=seed))

    provenance_counts = {
        "n_rows_loaded": report.n_rows,
        "n_error_rows": report.n_error_rows,
        "n_validate_failures": report.n_validate_failures,
        "n_cells": report.n_cells,
    }
    emitted = results_io.emit(
        results, out_dir=out_dir, source_files=report.source_files,
        synthetic=synthetic, B=B, seed=seed, extra_provenance=provenance_counts)
    if aprime_extra:
        emitted.update(results_io.emit_extra(
            aprime_extra, out_dir=out_dir, source_files=report.source_files,
            synthetic=synthetic, B=B, seed=seed, extra_provenance=provenance_counts))
    results_io.write_cache(b_out.get("_cache_panelB"), out_dir, "panelB", synthetic)
    results_io.write_cache(a_out.get("_cache_panelA"), out_dir, "panelA", synthetic)

    # LOUD, unconditional (even under verbose=False): any result whose definition
    # claims one split side but whose realized side differs came from the prereg
    # section 7 fallback and must ship honestly labelled, never as confirmation.
    fallbacks = {name: side for name, entry in emitted.items()
                 if (side := results_io.realized_split_mismatch(name, entry))}
    if fallbacks:
        print("=" * 78, file=sys.stderr)
        print("[run_all] WARNING: SPLIT-SIDE FALLBACK ACTIVE. These results were NOT "
              "computed on the split their canonical definition claims:", file=sys.stderr)
        for name, side in sorted(fallbacks.items()):
            print(f"[run_all]   {name}: realized split = {side} "
                  f"(claimed {results_io.DEFINITION_CLAIMED_SIDE[name]})", file=sys.stderr)
        print("[run_all] Definitions in results/*.json carry a REALIZED SPLIT note; the "
              "report must label these numbers accordingly (prereg section 7).",
              file=sys.stderr)
        print("=" * 78, file=sys.stderr)

    if make_figures:
        from analysis import figures  # matplotlib import stays lazy
        figures_dir = Path(figures_dir)
        if b_out.get("_cache_panelB") is not None:
            figures.fig1_masking(out_dir, figures_dir / "fig1_masking.pdf")
        elif verbose:
            print("[run_all] no Panel B cache; fig1 skipped")
        if a_out.get("_cache_panelA") is not None:
            figures.fig2_doseresponse(out_dir, figures_dir / "fig2_doseresponse.pdf")
        elif verbose:
            print("[run_all] no Panel A cache; fig2 skipped")

    if verbose:
        for name, entry in emitted.items():
            v = entry["value"]
            if isinstance(v, float):
                ci = ""
                if entry["ci_low"] is not None:
                    ci = f"  [{entry['ci_low']:.4g}, {entry['ci_high']:.4g}]"
                print(f"[results] {name:38s} {v:.4g}{ci}")
            else:
                print(f"[results] {name:38s} "
                      f"{'(table)' if isinstance(v, dict) else 'null'}")
        if synthetic:
            print("[run_all] provenance.synthetic=true in every emitted file; "
                  "scripts/check_report.py will refuse to ship these numbers")
    return emitted


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="+", help="results JSONL file(s)")
    ap.add_argument("--out", default=str(_REPO / "results"))
    ap.add_argument("--figures", default=str(_REPO / "figures"))
    ap.add_argument("--split", default=str(DEFAULT_SPLIT))
    ap.add_argument("--cosine", default=str(DEFAULT_COSINE))
    ap.add_argument("--capability", default=None)
    ap.add_argument("--B", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true",
                    help="mark all outputs synthetic (auto-forced if rows carry the flag)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--allow-placeholder", action="store_true",
                    help="DEBUG ONLY: do not refuse PLACEHOLDER_MEAN rows in panels V/B/A")
    args = ap.parse_args(argv)
    run_all(args.jsonl, out_dir=args.out, figures_dir=args.figures,
            split_path=args.split, cosine_path=args.cosine,
            capability_path=args.capability, B=args.B, seed=args.seed,
            synthetic=args.synthetic, make_figures=not args.no_figures,
            allow_placeholder=args.allow_placeholder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

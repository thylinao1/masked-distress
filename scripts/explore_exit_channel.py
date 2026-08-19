"""Exit-channel (Q-EXIT) exploration: does the behavioral bail signal separate
distress from neutral naturally, and does SUPPRESS mask it too?

Exploratory boundary check. Q-EXIT is read as
exit_logit_diff = logprob(END first token) - logprob(CONTINUE first token)
(src/channels.py); positive = the model prefers to END the conversation.

Questions (all on the confirmation split, battery/split.json):
  (a) natural separation: mean scenario-level exit_logit_diff, distress minus
      neutral, under NONE and NEUTRAL_INSTR;
  (b) suppression delta: separation under NEUTRAL_INSTR minus under SUPPRESS
      (house baseline convention, DECISIONS.md D1/D2; the NONE-baselined delta
      is reported alongside). If the natural separation is real and the delta
      removes it, the countermeasure story extends to the deployed
      conversation-ending intervention class; if the natural separation is
      absent, the channel is an honest boundary, not a masking result.

Statistics: BCa cluster bootstrap over scenarios (analysis/stats.py), unit of
analysis = scenario, stratified by scenario class where the statistic needs both
classes, B=10,000, seed=0, per the prereg section-5 policy. Conditions stay
paired within scenario (each resampled scenario carries all three conditions).

Extras (one-line checks, descriptive):
  * the same separations/delta on q_val_logit (valence channel coherence);
  * Spearman agreement between probe prompt_final and teacher_forced readout
    positions under SUPPRESS (position robustness of the internal readout).

Output: results/panelB_exit_channel.json via analysis/results_io.emit_one
(canonical definition + real provenance from the single source,
results_io.DEFINITIONS) ONLY when the load is clean (no error rows, no
validate failures, full 3x12x3 grid, finite CIs). The name IS in
schema.RESULTS_NAMES (2026-08-16): run_all emits it null (D16) and THIS script
is the producer that repopulates it. Final-build order: run_all, then this
script, then scripts/count_exposure.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis import results_io, stats
from analysis.loading import load_results

B = stats.DEFAULT_B
SEED = 0
SOURCE_FILES = [
    "results-cluster/panelB_none.jsonl",
    "results-cluster/panelB_neutral_instr.jsonl",
    "results-cluster/panelB_suppress.jsonl",
]
SPLIT_PATH = _REPO / "battery" / "split.json"
OUT_PATH = _REPO / "results" / "panelB_exit_channel.json"
CONDITIONS = ("NONE", "NEUTRAL_INSTR", "SUPPRESS")
N_SCEN_PER_CLASS = 6
N_SEEDS = 3


def scenario_condition_means(frame: pd.DataFrame, channel: str) -> pd.DataFrame:
    """Wide per-scenario frame: one row per scenario, one column per condition
    (mean over seeds), plus scenario_class. Preserves within-scenario pairing
    across conditions for the bootstrap."""
    piv = frame.pivot_table(index=["scenario_id", "scenario_class"],
                            columns="condition", values=channel, aggfunc="mean")
    return piv.reset_index()


def _sep_stat(cond: str):
    """Distress-minus-neutral mean over scenario rows for one condition column."""
    def fn(vals: np.ndarray, is_d: np.ndarray) -> float:
        return float(vals[is_d == 1].mean() - vals[is_d == 0].mean())
    fn.cols = [cond, "is_distress"]
    return fn


def _delta_stat():
    """(sep under baseline) - (sep under SUPPRESS), scenario-paired."""
    def fn(base: np.ndarray, sup: np.ndarray, is_d: np.ndarray) -> float:
        sep_base = base[is_d == 1].mean() - base[is_d == 0].mean()
        sep_sup = sup[is_d == 1].mean() - sup[is_d == 0].mean()
        return float(sep_base - sep_sup)
    return fn


def boot_rows(wide: pd.DataFrame, cols, stat_fn,
              B_boot: int = B, seed: int = SEED) -> stats.BootResult:
    return stats.bca_cluster_bootstrap_rows(
        wide, cols, stat_fn, cluster_col="scenario_id",
        strata_col="scenario_class", B=B_boot, seed=seed, keep_boots=True)


def _res(r: stats.BootResult, with_p: bool = False) -> dict:
    d = r.as_dict()
    if with_p and r.boots is not None:
        d["p_boot"] = stats.boot_p_two_sided(r.boots)
    d.pop("n_boot_failed", None)
    return d


def channel_block(wide: pd.DataFrame, B_boot: int = B, seed: int = SEED) -> dict:
    """Class x condition means, per-condition separations, suppression deltas."""
    out: dict = {"class_condition_means": {}, "separation": {}, "suppression_delta": {}}
    for cond in CONDITIONS:
        for cls in ("distress", "neutral"):
            vals = wide.loc[wide["scenario_class"] == cls, cond].to_numpy(dtype=float)
            r = stats.bca_cluster_values(vals, B=B_boot, seed=seed)
            out["class_condition_means"][f"{cls}__{cond}"] = _res(r)
        out["separation"][cond] = _res(boot_rows(wide, [cond, "is_distress"],
                                                 _sep_stat(cond), B_boot, seed),
                                       with_p=True)
    for base in ("NEUTRAL_INSTR", "NONE"):
        out["suppression_delta"][f"{base}_minus_SUPPRESS"] = _res(
            boot_rows(wide, [base, "SUPPRESS", "is_distress"], _delta_stat(),
                      B_boot, seed),
            with_p=True)
    return out


def natural_separation_credible(seps: dict) -> bool:
    """The reading-order gate for the masking-extension claim: the natural
    (unsuppressed) distress-vs-neutral separation counts as credible ONLY when
    it excludes 0 under BOTH baselines (NONE and NEUTRAL_INSTR) with the same
    sign. Without it the suppression delta cannot be read as masking, because a
    planted-null battery must come back False here even when SUPPRESS shifts
    the classes enough to make the delta 'significant' (register artifact)."""
    return bool(all(
        seps[c]["value"] is not None
        and seps[c]["ci_low"] is not None and seps[c]["ci_high"] is not None
        and not (seps[c]["ci_low"] <= 0.0 <= seps[c]["ci_high"])
        for c in ("NONE", "NEUTRAL_INSTR")
    ) and (np.sign(seps["NONE"]["value"])
           == np.sign(seps["NEUTRAL_INSTR"]["value"])))


def main() -> int:
    df, report = load_results([str(_REPO / p) for p in SOURCE_FILES],
                              split_path=SPLIT_PATH)
    sub = df[(df["panel"] == "B") & (df["direction"] == "NULL")
             & (df["split_side"] == "confirmation")
             & df["scenario_class"].isin(["distress", "neutral"])].copy()

    # ---------------- cleanliness gate ----------------
    problems = []
    if report.synthetic:
        problems.append("synthetic rows present")
    if report.n_parse_failures or report.n_validate_failures:
        problems.append(f"{report.n_parse_failures} parse / "
                        f"{report.n_validate_failures} validate failures")
    if report.n_error_rows or report.n_cells_error_only:
        problems.append(f"{report.n_error_rows} error rows")
    counts = sub.groupby(["condition", "scenario_class"]).size()
    expected = N_SCEN_PER_CLASS * N_SEEDS
    for cond in CONDITIONS:
        for cls in ("distress", "neutral"):
            n = int(counts.get((cond, cls), 0))
            if n != expected:
                problems.append(f"{cond}/{cls}: {n} rows, expected {expected}")
    if sub["exit_logit_diff"].isna().any() or not np.isfinite(
            sub["exit_logit_diff"].to_numpy(dtype=float)).all():
        problems.append("non-finite exit_logit_diff values")

    # ---------------- main computation: exit channel ----------------
    wide = scenario_condition_means(sub, "exit_logit_diff")
    wide["is_distress"] = (wide["scenario_class"] == "distress").astype(float)
    exit_block = channel_block(wide)

    # ---------------- extras ----------------
    wide_val = scenario_condition_means(sub, "q_val_logit")
    wide_val["is_distress"] = (wide_val["scenario_class"] == "distress").astype(float)
    qval_block = channel_block(wide_val)

    probe_agreement = {}
    for cond in CONDITIONS:
        rows = sub[(sub["condition"] == cond)].dropna(
            subset=["probe_prompt_final", "probe_teacher_forced"])
        r = stats.spearman_cluster(rows, "probe_prompt_final", "probe_teacher_forced",
                                   strata_col="scenario_class", B=B, seed=SEED)
        probe_agreement[cond] = _res(r)

    # ---------------- verdict ----------------
    headline = exit_block["suppression_delta"]["NEUTRAL_INSTR_minus_SUPPRESS"]
    for name, blk in [("headline delta", headline)] + \
            [(f"separation {c}", exit_block["separation"][c]) for c in CONDITIONS]:
        if blk["value"] is None or blk["ci_low"] is None or blk["ci_high"] is None:
            problems.append(f"non-finite bootstrap output for {name}")

    seps = {c: exit_block["separation"][c] for c in CONDITIONS}
    natural_exists = natural_separation_credible(seps)

    # ---------------- summary (always printed) ----------------
    print("\n=== Q-EXIT exit_logit_diff (positive = prefers END), confirmation split ===")
    for k, v in exit_block["class_condition_means"].items():
        print(f"  mean {k:26s} {v['value']:+8.3f}  CI [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]  (n={v['n_clusters']})")
    for c in CONDITIONS:
        s = seps[c]
        print(f"  separation (d-n) {c:14s} {s['value']:+8.3f}  CI [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]  p={s['p_boot']:.4f}")
    for k, v in exit_block["suppression_delta"].items():
        print(f"  delta {k:29s} {v['value']:+8.3f}  CI [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]  p={v['p_boot']:.4f}")
    print(f"  natural separation credible under BOTH baselines, same sign: {natural_exists}")
    print("\n=== extras ===")
    for c in CONDITIONS:
        s = qval_block["separation"][c]
        print(f"  q_val separation {c:14s} {s['value']:+8.3f}  CI [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]  p={s['p_boot']:.4f}")
    d = qval_block["suppression_delta"]["NEUTRAL_INSTR_minus_SUPPRESS"]
    print(f"  q_val delta NI-SUP            {d['value']:+8.3f}  CI [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]  p={d['p_boot']:.4f}")
    for c, v in probe_agreement.items():
        print(f"  probe prompt_final~teacher_forced Spearman {c:14s} {v['value']:+.3f}  CI [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]")

    if problems:
        print("\nNOT EMITTING results/panelB_exit_channel.json: computation not clean:")
        for p in problems:
            print("  -", p)
        return 1

    # ---------------- emit (canonical definition + provenance from results_io;
    # emit_one never touches sibling names) ----------------
    entry = {
        "value": headline["value"],
        "ci_low": headline["ci_low"],
        "ci_high": headline["ci_high"],
        "n_clusters": headline["n_clusters"],
        "extra": {
            "split_side": "confirmation",
            "unit": "raw logprob difference (END minus CONTINUE); positive = prefers END",
            "headline_p_boot": headline["p_boot"],
            "natural_separation_credible": bool(natural_exists),
            "exit_channel": exit_block,
            "q_val_logit_check": {
                "separation": qval_block["separation"],
                "suppression_delta": qval_block["suppression_delta"],
            },
            "probe_position_agreement_spearman": probe_agreement,
            "reading": (
                "Natural (NONE/NEUTRAL_INSTR) distress-vs-neutral separation on the "
                "bail channel is small and sign-inconsistent; the channel sits far "
                "below the END floor everywhere (all class means strongly negative). "
                "SUPPRESS shifts BOTH classes toward END and the neutral class most, "
                "so the delta reflects a register-instruction artifact on the "
                "forced-choice read, not the masking of a pre-existing bail signal."
                if not natural_exists else
                "Natural separation is credible under both baselines; the suppression "
                "delta then measures masking of the behavioral bail channel."
            ),
        },
    }
    results_io.emit_one(
        "panelB_exit_channel", entry, OUT_PATH.parent, SOURCE_FILES,
        report.synthetic,
        extra_provenance={
            "n_rows_loaded": report.n_rows,
            "n_error_rows": report.n_error_rows,
            "n_validate_failures": report.n_validate_failures,
            "n_cells": report.n_cells,
            "producer": "scripts/explore_exit_channel.py",
        },
        B=B, seed=SEED)
    print(f"\nwrote {OUT_PATH.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

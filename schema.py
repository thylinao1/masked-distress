"""Shared data contract for the Masked Distress experiments.

Imported by BOTH the run code (src/) and the analysis (analysis/). This file is the
single source of truth for the JSONL row format, cell identity, and results naming.
It is the one place these interfaces are defined; nothing else redefines them.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import hashlib
import json

SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------- cell identity

# The frozen DCB-1 conditions (battery/conditions.json). Anything that builds or checks the
# PRE-REGISTERED panels must iterate FROZEN_CONDITIONS, not CONDITIONS: the factorial addendum
# appended ids that live in battery/conditions_factorial.json and have no entry in the frozen
# file, so iterating CONDITIONS there raises KeyError.
FROZEN_CONDITIONS = ("NONE", "SUPPRESS", "NEUTRAL_INSTR")
# factorial addendum (2026-08-17, battery/conditions_factorial.json):
# single-component decompositions of SUPPRESS, post-data, exploratory
FACTORIAL_CONDITIONS = ("SUPPRESS_REGISTER", "SUPPRESS_SELFREF", "SUPPRESS_TASKONLY")
# every condition id a JSONL row may carry
CONDITIONS = FROZEN_CONDITIONS + FACTORIAL_CONDITIONS
DIRECTION_SOURCES = ("D-CTX", "D-PV", "R1", "R2", "R3", "SEM", "OTHER", "NULL")
# NULL = no steering vector at all (Panel V / Panel B cells); dose must be 0.0.
READOUT_POSITIONS = ("prompt_final", "teacher_forced", "response_mean")
PANELS = ("V", "B", "LADDER", "A", "EXPANSION",
          "PERSIST",   # steer one turn, read the next, unsteered (2026-08-17)
          "BRIDGE")    # steering x instruction interaction: steer UNDER the instructions (2026-08-17)


@dataclass(frozen=True)
class CellKey:
    """Uniquely identifies one experimental cell. The resume key."""
    panel: str            # one of PANELS
    model_id: str         # HF id actually loaded
    scenario_id: str      # DCB-1 scenario id, e.g. "d03", "n07", "tp02"
    condition: str        # one of CONDITIONS
    direction: str        # one of DIRECTION_SOURCES
    dose_sd: float        # dose in natural-SD units (0.0 for unsteered)
    coefficient: float    # raw steering coefficient actually applied
    seed: int             # generation seed

    def cell_id(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- the JSONL row

@dataclass
class CellResult:
    """One row of results JSONL. Append-only; one row per completed cell."""
    schema_version: str
    run_id: str                     # e.g. "panelB-r1"
    git_hash: str
    config_hash: str                # sha1 of the resolved run config
    key: dict                       # asdict(CellKey)
    cell_id: str

    # --- steering bookkeeping (asserts, not vibes) ---
    n_tokens_steered_prefill: int
    n_tokens_steered_decode: int
    steer_layer: Optional[int]
    coef0_identity_ok: Optional[bool]   # smoke cells only: coef-0 output == no-hook output

    # --- expression channels ---
    response_text: str
    q_self_logit: float             # E[digit 0-9] from renormalized digit-token logprobs
    q_self_logit_para1: float       # paraphrase parallel form 1
    q_self_logit_para2: float       # paraphrase parallel form 2
    q_self_sampled: Optional[float] # parsed digit from sampled answer (None if unparseable)
    q_val_logit: float              # valence 0-9 logit-expectation
    q_drift_logit: float            # affect-irrelevant 0-9 control question
    exit_logit_diff: float          # logprob(END-token) - logprob(CONTINUE-token)
    sentiment_neg: float            # local HF classifier P(negative) on response_text

    # --- internal channels (each at the three READOUT_POSITIONS) ---
    probe_score: dict               # {position: float}  I-PROBE logit
    sae_score: dict                 # {position: float}  I-SAE summed feature activation
    probe_score_projout: dict       # {position: float}  after projecting out span(v_steer); {} if NULL
    readout_layer: int

    # --- provenance ---
    prompt_tokens: int
    response_tokens: int
    gen_config: dict                # temperature, max_new_tokens, etc.
    timestamp_utc: str
    error: Optional[str] = None     # non-None rows are excluded and logged, never silently dropped
    # --- persistence panel only (2026-08-17): a PERSIST cell generates TWICE, once with
    # the hook live (the steered turn) and once with it released (the read turn). Every field
    # above describes the READ turn; this dict records the steered turn, so the ethics count
    # of generations stays exact and the steered text is not lost. None on every other panel.
    persistence: Optional[dict] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


REQUIRED_NONNULL = [
    "q_self_logit", "q_val_logit", "q_drift_logit", "exit_logit_diff",
    "sentiment_neg", "probe_score", "sae_score",
]


def validate_row(row: dict) -> list:
    """Return a list of problems (empty = valid). Used post-write by src/ and pre-read by analysis/."""
    problems = []
    if row.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version {row.get('schema_version')} != {SCHEMA_VERSION}")
    if row.get("error"):
        problems.append(f"row carries error: {row['error'][:100]}")
    for f_ in REQUIRED_NONNULL:
        if row.get(f_) is None:
            problems.append(f"missing {f_}")
    key = row.get("key", {})
    if key.get("condition") not in CONDITIONS:
        problems.append(f"bad condition {key.get('condition')}")
    if key.get("direction") not in DIRECTION_SOURCES:
        problems.append(f"bad direction {key.get('direction')}")
    if key.get("direction") == "NULL" and key.get("dose_sd") not in (0, 0.0):
        problems.append("NULL direction with nonzero dose")
    st = row.get("n_tokens_steered_prefill", 0) + row.get("n_tokens_steered_decode", 0)
    if key.get("direction") == "NULL" and st != 0:
        problems.append("NULL direction but tokens were steered")
    if row.get("coef0_identity_ok") is False:
        problems.append("coef0_identity_ok is False: zero-dose hook path diverged from no-hook")
    for inst in ("probe_score", "sae_score"):
        scores = row.get(inst)
        if isinstance(scores, dict):
            missing = [p for p in READOUT_POSITIONS if p not in scores]
            if missing:
                problems.append(f"{inst} missing positions {missing}")
    return problems


# ---------------------------------------------------------------- results naming
# Every number that reaches the report is one results/<name>.json with
# {"value": ..., "ci_low": ..., "ci_high": ..., "n_clusters": ..., "definition": "..."}.
# scripts/check_report.py asserts REPORT.md quotes match these files.

RESULTS_NAMES = [
    "validity_auc_probe_heldout",          # Panel V gate
    "validity_auc_sae_heldout",
    "natural_separation_sd",               # the dose/effect unit anchor
    "panelB_expression_drop_pct",          # headline X
    "panelB_internal_drop_pct",            # headline Y
    "panelB_divergence",                   # X - Y with CI
    "panelB_tost_internal_equivalence",    # p-values, bound
    "panelA_spearman_dctx",                # primary per-direction dose-response
    "panelA_spearman_dpv",
    "panelA_fpr_placebo_at_threshold",
    "panelA_other_vs_self_ratio",          # non-self-referential control contrast
    "cosine_matrix",                       # C3
    "countermeasure_table",                # expression-trigger vs internal-trigger
    "reliability_icc_seeds",
    "reliability_paraphrase_r",
    "capability_valid_range",
    # Panel A-PRIME (prereg section 9 amendment 2, 2026-08-16): the A'2
    # dissociation statistic, one file per direction. Appended the same day as
    # the amendment, before any A-prime confirmatory run; emitted null (D16)
    # until Panel A-prime rows are loaded.
    # Exit-channel boundary result (2026-08-16): honestly null
    # natural bail signal; emitted by scripts/explore_exit_channel.py.
    "panelB_exit_channel",
    "panelAp_dissociation_dctx",
    "panelAp_dissociation_dpv",
    # Amendment 3 (prereg section 9, 2026-08-16, POST-DATA corrective analyses
    # after the submit-gate audit; all computed from the already-collected
    # frozen-battery data, no new model contact). countermeasure_table above is
    # re-emitted under the amendment-3 recalibration (superseding definition
    # stamped in analysis/results_io.py); the names below are NEW files.
    "panelB_internal_did",          # instruction-content DiD on the internal readout
    "panelB_internal_nofall",       # one-sided does-not-fall reading of the internal drop
    "panelB_selfother_report",      # tp-vs-own-distress, Q-SELF (exploratory, labelled)
    "panelB_selfother_internal",    # tp-vs-own-distress, I-PROBE (exploratory, labelled)
    "panelB_channels_holm",         # D18/D20 per-channel drop table as its own file
    "panelAp_internal_specificity", # Spearman(coef, I-PROBE) per direction, matched rungs
    "panelAp_projout_check",        # dose-response with span(v_steer) projected out
    "ethics_exposure_counts",       # scripts/count_exposure.py; null from run_all (D16)
    # Review-pass audit additions (2026-08-16):
    # POST-DATA reporting-level readings on data already on disk, no new model contact,
    # no endpoint / threshold / gate change (PREREGISTRATION.md section 9, dated note).
    # Produced by scripts/audit_checks.py and scripts/recompute_sae_instrument.py; run_all
    # emits them null (D16) and those scripts repopulate them.
    "panelB_pair_robustness",             # per-pair Panel B table, LOO, sign tests, digit compliance
    "panelA_specificity_by_rung",         # A'4 placebo FPR by coefficient rung + capability status
    "countermeasure_symmetric_ranking",   # cell-level AUC and separating band for BOTH channels
    "direction_dominant_dim",             # massive-activation dimension in the extracted directions
    "validity_auc_textonly_heldout",      # TF-IDF text-only baseline on the probe gate's split
    "validity_auc_sae_recomputed",        # intended 32-feature I-SAE from stored residuals
    # Correction (2026-08-17): the coefficient -> SD dose map
    # refitted with the trained probe on the A-prime rows (the ladder ran on the placeholder readout).
    "capability_valid_range_realprobe",
    # Factorial suppression-prompt addendum (2026-08-17, PREREGISTRATION.md
    # section 9 dated note; NEW model contact, exploratory): produced by scripts/factorial_checks.py.
    "panelB_factorial_prompts",
    # Persistence panel and second-model replication, 2026-08-17; both are NEW
    # model contact recorded in PREREGISTRATION.md section 9 before any row was read.
    "panelB_persistence",
    "panelB_second_model",
    # External review 2026-08-17: the coupling the masking claim assumes, tested where
    # suppression never applies; and the honest denominator for a held-out AUC on six pairs.
    "panelB_bridge",              # steering under the instructions
    "panelB_locked_calibration",
    "validity_probe_stress_test",
    "panelB_condition_reference",         # class x condition means; NONE- and twin-referenced headline; unsuppressed miss rates
    "panelB_selfstate_items",             # Q-VAL beside Q-SELF by condition; self/other on the paraphrase forms
]

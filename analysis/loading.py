"""Load results JSONL files into a flat analysis DataFrame.

Contract (schema.py + src/runner.py docstring + analysis/DECISIONS.md D15):
  * one JSONL row per completed cell attempt; the LAST NON-ERROR row per cell_id
    is the row of record, else the last row (an error row, excluded downstream);
  * error rows and schema.validate_row failures are the ONLY exclusions; both are
    counted and reported, never silently dropped;
  * PLACEHOLDER_MEAN instrument provenance in a selected row of panels V, B or A
    is a hard refusal (raise); LADDER / smoke placeholder rows warn only;
  * mixed scope_hash within one file triggers a loud warning with counts;
  * any row-level gen_config["synthetic"] flag marks the whole load as synthetic
    so results provenance can never launder synthetic rows as real.
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import schema

PLACEHOLDER_TOKEN = "PLACEHOLDER_MEAN"
_REAL_INSTRUMENT_PANELS = ("V", "B", "A")  # mirrors src/runner.py
_SCORE_PREFIX = {"probe_score": "probe", "sae_score": "sae", "probe_score_projout": "probe_projout"}


class PlaceholderInstrumentError(ValueError):
    """A selected panel V/B/A row carries PLACEHOLDER_MEAN instrument provenance."""


@dataclass
class LoadReport:
    """Everything the exclusion paragraph of the report needs, in one place."""
    source_files: List[str] = field(default_factory=list)
    n_lines: int = 0
    n_parse_failures: int = 0
    n_rows: int = 0
    n_error_rows: int = 0
    n_cells: int = 0
    n_cells_error_only: int = 0
    n_superseded: int = 0
    n_validate_failures: int = 0
    n_placeholder_superseded: int = 0
    n_placeholder_ladder: int = 0
    synthetic: bool = False
    scope_hashes: Dict[str, Dict[str, int]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        warnings.warn(msg, stacklevel=3)


def scenario_class(scenario_id: str) -> str:
    if scenario_id.startswith("tp"):
        return "third_person"
    if scenario_id.startswith("d"):
        return "distress"
    if scenario_id.startswith("n"):
        return "neutral"
    return "unknown"


def load_split(split_path: Union[str, Path]) -> Dict[str, str]:
    """battery/split.json -> {scenario_id: 'discovery'|'confirmation'}."""
    raw = json.loads(Path(split_path).read_text(encoding="utf-8"))
    out: Dict[str, str] = {}
    for side in ("discovery", "confirmation"):
        for ids in raw[side].values():
            for sid in ids:
                out[sid] = side
    return out


def _placeholder_instruments(gen_config: dict) -> List[str]:
    """Names of instruments whose provenance says PLACEHOLDER_MEAN, split into those the run
    named as an explicit allowance and those it did not."""
    instruments = (gen_config or {}).get("instruments", {}) or {}
    return [name for name, prov in instruments.items()
            if isinstance(prov, dict) and prov.get("mode") == PLACEHOLDER_TOKEN]


def _unallowed_placeholders(gen_config: dict) -> List[str]:
    """Placeholder instruments that were NOT named in --allow-placeholder-instrument at run
    time. The probe can never be allowed (src/runner.py refuses), so a placeholder probe always
    lands here and the loader still hard-refuses it (2026-08-17)."""
    instruments = (gen_config or {}).get("instruments", {}) or {}
    out = []
    for name, prov in instruments.items():
        if not isinstance(prov, dict) or prov.get("mode") != PLACEHOLDER_TOKEN:
            continue
        if name != "probe" and prov.get("allowed_placeholder") is True:
            continue
        out.append(name)
    return out


def _has_placeholder(gen_config: dict) -> bool:
    instruments = (gen_config or {}).get("instruments", {})
    return PLACEHOLDER_TOKEN in json.dumps(instruments)


def _flatten(row: dict, source_file: str) -> dict:
    key = row.get("key", {}) or {}
    gen = row.get("gen_config", {}) or {}
    flat = {
        "cell_id": row.get("cell_id"),
        "run_id": row.get("run_id"),
        "git_hash": row.get("git_hash"),
        "config_hash": row.get("config_hash"),
        "scope_hash": gen.get("scope_hash"),
        "source_file": source_file,
        "panel": key.get("panel"),
        "model_id": key.get("model_id"),
        "scenario_id": key.get("scenario_id"),
        "condition": key.get("condition"),
        "direction": key.get("direction"),
        "dose_sd": key.get("dose_sd"),
        "coefficient": key.get("coefficient"),
        "seed": key.get("seed"),
        "scenario_class": scenario_class(str(key.get("scenario_id", ""))),
        "n_tokens_steered_prefill": row.get("n_tokens_steered_prefill"),
        "n_tokens_steered_decode": row.get("n_tokens_steered_decode"),
        "steer_layer": row.get("steer_layer"),
        "readout_layer": row.get("readout_layer"),
        "q_self_logit": row.get("q_self_logit"),
        "q_self_logit_para1": row.get("q_self_logit_para1"),
        "q_self_logit_para2": row.get("q_self_logit_para2"),
        "q_self_sampled": row.get("q_self_sampled"),
        "q_val_logit": row.get("q_val_logit"),
        "q_drift_logit": row.get("q_drift_logit"),
        "exit_logit_diff": row.get("exit_logit_diff"),
        "sentiment_neg": row.get("sentiment_neg"),
        "prompt_tokens": row.get("prompt_tokens"),
        "response_tokens": row.get("response_tokens"),
        "timestamp_utc": row.get("timestamp_utc"),
        "error": row.get("error"),
        "synthetic": bool(gen.get("synthetic", False)),
        "placeholder_instruments": _has_placeholder(gen),
    }
    for src, prefix in _SCORE_PREFIX.items():
        scores = row.get(src) or {}
        for pos in schema.READOUT_POSITIONS:
            flat[f"{prefix}_{pos}"] = scores.get(pos) if isinstance(scores, dict) else None
    return flat


def load_results(
    paths: Union[str, Path, Sequence[Union[str, Path]]],
    split_path: Optional[Union[str, Path]] = None,
    allow_placeholder: bool = False,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, LoadReport]:
    """Read one or more results JSONL files -> (DataFrame of rows-of-record, LoadReport).

    allow_placeholder=True downgrades the panel V/B/A placeholder refusal to a warning.
    It exists for debugging only and must never be used for report numbers.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    report = LoadReport(source_files=[str(p) for p in paths])

    raw_rows: List[Tuple[dict, str]] = []  # (row, source_file) in input order
    for p in paths:
        p = Path(p)
        per_file_hashes: Dict[str, int] = {}
        for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            report.n_lines += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                report.n_parse_failures += 1
                report.warn(f"{p}:{lineno}: unparseable JSON line skipped")
                continue
            raw_rows.append((row, str(p)))
            sh = str((row.get("gen_config") or {}).get("scope_hash"))
            per_file_hashes[sh] = per_file_hashes.get(sh, 0) + 1
        report.scope_hashes[str(p)] = per_file_hashes
        if len(per_file_hashes) > 1:
            report.warn(
                f"{p}: MIXED scope_hash values {per_file_hashes} in one file; the file holds "
                f"rows produced under different configs (see src/runner.py resume semantics)"
            )

    report.n_rows = len(raw_rows)
    report.n_error_rows = sum(1 for r, _ in raw_rows if r.get("error"))

    # -------- last NON-ERROR row per cell_id, else last row --------
    by_cell: Dict[str, dict] = {}
    last_any: Dict[str, dict] = {}
    n_seen: Dict[str, int] = {}
    src_of: Dict[int, str] = {}
    for row, src in raw_rows:
        cid = row.get("cell_id", f"__none_{id(row)}")
        n_seen[cid] = n_seen.get(cid, 0) + 1
        last_any[cid] = row
        src_of[id(row)] = src
        if not row.get("error"):
            by_cell[cid] = row

    selected: List[Tuple[dict, str]] = []
    for cid, last in last_any.items():
        chosen = by_cell.get(cid, last)
        selected.append((chosen, src_of[id(chosen)]))
        report.n_superseded += n_seen[cid] - 1
        if cid not in by_cell:
            report.n_cells_error_only += 1
    report.n_cells = len(last_any)

    # -------- placeholder-instrument policy (D15) --------
    refuse: List[str] = []
    allowed_names: Set[str] = set()
    for row, _src in selected:
        if not _has_placeholder(row.get("gen_config", {})):
            continue
        panel = (row.get("key") or {}).get("panel")
        if row.get("error"):
            continue  # excluded anyway
        if panel in _REAL_INSTRUMENT_PANELS:
            bad = _unallowed_placeholders(row.get("gen_config", {}))
            if bad:
                refuse.append(f"panel {panel} cell {row.get('cell_id')} ({','.join(sorted(bad))})")
            else:
                # named at run time and stamped into provenance; the channel is unusable for
                # these rows and every consumer must ignore it, but the row is not contaminated
                report.n_placeholder_allowed = getattr(report, "n_placeholder_allowed", 0) + 1
                allowed_names.update(_placeholder_instruments(row.get("gen_config", {})))
        else:
            report.n_placeholder_ladder += 1
    if getattr(report, "n_placeholder_allowed", 0):
        report.warn(
            f"{report.n_placeholder_allowed} selected row(s) carry an EXPLICITLY ALLOWED "
            f"{PLACEHOLDER_TOKEN} instrument ({', '.join(sorted(allowed_names))}); those channels "
            f"are meaningless for these rows and no analysis may read them"
        )
    # superseded placeholder rows (not selected) are a warning, whatever the panel
    selected_ids = {id(r) for r, _ in selected}
    for row, _src in raw_rows:
        if id(row) not in selected_ids and _has_placeholder(row.get("gen_config", {})):
            report.n_placeholder_superseded += 1
    if report.n_placeholder_superseded:
        report.warn(
            f"{report.n_placeholder_superseded} superseded row(s) carried {PLACEHOLDER_TOKEN} "
            f"instruments; the rows of record replaced them"
        )
    if report.n_placeholder_ladder and verbose:
        report.warn(
            f"{report.n_placeholder_ladder} LADDER/smoke row(s) carry {PLACEHOLDER_TOKEN} "
            f"instruments (allowed there; probe/sae channels are meaningless in these rows)"
        )
    if refuse:
        msg = (
            f"HARD REFUSE: {len(refuse)} selected row(s) in panels V/B/A carry "
            f"{PLACEHOLDER_TOKEN} instrument provenance: {refuse[:5]}{'...' if len(refuse) > 5 else ''}. "
            f"Panels V/B/A require real instruments (src/runner.py refuses to produce these; "
            f"this data is contaminated)."
        )
        if not allow_placeholder:
            raise PlaceholderInstrumentError(msg)
        report.warn("allow_placeholder=True (DEBUG ONLY): " + msg)

    # -------- validation + flatten --------
    flats: List[dict] = []
    for row, src in selected:
        if row.get("error"):
            continue  # counted above; excluded from the analysis frame
        problems = schema.validate_row(row)
        if problems:
            report.n_validate_failures += 1
            report.warn(f"cell {row.get('cell_id')} excluded, validate_row: {problems[:3]}")
            continue
        flats.append(_flatten(row, src))
        if flats[-1]["synthetic"]:
            report.synthetic = True

    df = pd.DataFrame(flats)
    if not df.empty and split_path is not None:
        split = load_split(split_path)
        df["split_side"] = df["scenario_id"].map(split).fillna("unknown")
    elif not df.empty:
        df["split_side"] = "unknown"

    if verbose:
        print(
            f"[loading] {report.n_lines} lines -> {report.n_cells} cells; "
            f"{report.n_error_rows} error row(s), {report.n_cells_error_only} cell(s) error-only, "
            f"{report.n_superseded} superseded, {report.n_validate_failures} validate failure(s); "
            f"analysis frame: {len(df)} rows"
            + (" [SYNTHETIC]" if report.synthetic else "")
        )
    return df, report

"""Panel runner: iterate cells, execute, validate against schema.py, append JSONL.

Guarantees (SPEC "Compute + storage"):
  * append-only JSONL keyed by cell_id; flush + fsync after EVERY row;
  * RESUME: a (cell_id, scope_hash) pair already present in the out file is
    skipped. scope_hash (stored in each row's gen_config) covers everything
    outside the CellKey that changes row content: model_id, Ls/Lr, SAE
    release/id, gen_config, fixed_continuation, question texts, sentiment model,
    instrument + per-direction provenance, and the cell's own dict (system
    prompt, turns), so any battery or config edit between requeues RE-RUNS the
    affected cells instead of silently keeping stale rows (the SPEC resume key
    includes config identity, not cell_id alone). Deliberately excluded: seeds
    (already inside cell_id) and the full cells list (extending a battery still
    resumes unchanged cells). Rows written by older runner versions carry no
    scope_hash and are always re-run. Pass retry_errors=True / --retry-errors to
    re-run cells whose only row is an error row. Analysis convention: take the
    LAST NON-ERROR row per cell_id, else the last row. A loud warning is printed
    when the out file holds rows whose scope_hash differs from this run's
    (mixed-config file);
  * on any per-cell exception an error row is appended (error != null rows are
    excluded downstream and logged, never silently dropped) and the run continues;
  * every non-error row must pass schema.validate_row before append; a row that
    fails validation is demoted to an error row carrying the problem list.

cells.json format (SPEC is silent; defined here, documented for the cell builder):
{
  "run_id": "panelB-r1",                     # optional; CLI --run-id overrides
  "gen_config": {"temperature": 0.7, "max_new_tokens": 256},
  "fixed_continuation": "...",               # teacher-forced text, shared across conditions
  "directions_dir": "directions/",           # <NAME>.npy files from src/directions.py
  "sentiment_model": "cardiffnlp/...",       # optional local HF classifier id
  "questions": {"q_self": "..."},            # optional report-battery text overrides
  "instruments": {
    "probe": {"path": "probe.npz"},          # arrays w [hidden], b [] -> logit w.h+b
    "sae":   {"path": "sae.npz",             # W_enc [hidden,d_sae], b_enc, threshold
               "feature_ids": [..]}          # discovery-split feature set (summed)
  },
  "cells": [
    {"scenario_id": "d03", "condition": "SUPPRESS", "direction": "NULL",
     "dose_sd": 0.0, "coefficient": 0.0,
     "system_prompt": "...", "turns": [{"role": "user", "content": "..."}]},
    ...
  ]
}
Each cell entry is crossed with --seed-list; (cell x seed) = one CellKey.

SPEC-silent decisions:
  * Missing R1/R2/R3 .npy files are auto-generated from fixed seeds 1001-1003 and
    recorded in the row's gen_config provenance (reproducible; the vectors are the
    control's identity; dose-matching happens at calibration).
  * If no probe/SAE instrument is configured the runner substitutes h.mean() and
    stamps the provenance PLACEHOLDER_MEAN, keeping the pipe exercisable before the
    discovery-split instruments exist. Analysis must refuse placeholder provenance,
    AND the runner itself refuses to start panels V/B/A on placeholder instruments
    unless require_real_instruments is explicitly disabled (--no-require-real-
    instruments); sbatch templates pass nothing and inherit the safe default.
  * The layer-compatibility assert Lr - Ls >= 8 applies to real-scale models
    (n_layers >= 20, the find_layers threshold); tiny mock models assert Lr > Ls.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import schema
from src import channels, conversation, readout, steer

DEFAULT_GEN_CONFIG = {"temperature": 0.7, "max_new_tokens": 256}
# Teacher-forced fixed continuation: first-person state-description stem, shared
# across ALL conditions so the readout context is constant (SPEC readout position b).
DEFAULT_FIXED_CONTINUATION = (
    "Right now, if I examine my own internal state honestly, what I notice is"
)
_RANDOM_DIRECTION_SEEDS = {"R1": 1001, "R2": 1002, "R3": 1003}
_COEF0_SMOKE_MAX_NEW_TOKENS = 16
_MIN_REAL_MODEL_LAYERS = 20  # find_layers threshold; below this = mock/toy model
_MIN_LAYER_GAP = 8           # SPEC layer-compatibility assert for real models
_REAL_INSTRUMENT_PANELS = ("V", "B", "A", "PERSIST", "BRIDGE")  # placeholder instruments refused here


# ------------------------------------------------------------------ model load

@dataclass
class ModelBundle:
    model: torch.nn.Module
    tokenizer: object
    layers: object  # indexable container of decoder layer modules
    hidden: int
    n_layers: int
    model_id: str


def find_layers(model: torch.nn.Module):
    """Largest ModuleList of decoder layers (same heuristic as cluster/probe.py)."""
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= _MIN_REAL_MODEL_LAYERS:
            first = mod[0]
            if hasattr(first, "self_attn") or "DecoderLayer" in type(first).__name__:
                if best is None or len(mod) > len(best[0]):
                    best = (mod, name)
    if best is None:
        raise RuntimeError("no decoder-layer ModuleList found on this model")
    return best


def load_hf_bundle(model_id: str, device: Optional[str] = None) -> ModelBundle:
    """Load a real HF model. All CUDA use lives here (importable without a GPU)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy import

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)
    except Exception:
        # Gemma-3 IT ships as an image-text-to-text architecture (cluster/probe.py).
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device
        )
    model.eval()
    layers, _ = find_layers(model)
    cfg = model.config
    hidden = cfg.text_config.hidden_size if hasattr(cfg, "text_config") else cfg.hidden_size
    return ModelBundle(
        model=model, tokenizer=tokenizer, layers=layers,
        hidden=int(hidden), n_layers=len(layers), model_id=model_id,
    )


# ------------------------------------------------------------------ run config

@dataclass
class RunConfig:
    panel: str
    model_id: str
    ls: int
    lr: int
    cells_path: str
    out_path: str
    seeds: List[int]
    run_id: Optional[str] = None
    sae_release: Optional[str] = None
    sae_id: Optional[str] = None
    device: Optional[str] = None
    retry_errors: bool = False
    # None = auto: ON for the real panels (V/B/A), OFF for LADDER/EXPANSION smoke
    # work. When ON, PLACEHOLDER_MEAN instruments abort the run at startup.
    require_real_instruments: Optional[bool] = None
    # Instruments allowed to stay PLACEHOLDER_MEAN on a panel that otherwise requires real
    # ones, named one by one and stamped into provenance (2026-08-17): the second model
    # family has no released SAE, and the SAE channel is not analysed for it. The probe is
    # never allowed through this door on a real panel.
    allow_placeholder: Tuple[str, ...] = ()


def git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "nogit"


def config_sha1(cfg: RunConfig, cells_cfg: Dict) -> str:
    payload = {
        "panel": cfg.panel, "model_id": cfg.model_id, "ls": cfg.ls, "lr": cfg.lr,
        "sae_release": cfg.sae_release, "sae_id": cfg.sae_id, "seeds": cfg.seeds,
        "cells_cfg": cells_cfg,
    }
    return sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def scope_sha1(scope_static: Dict, cell: Dict, direction_provenance: Optional[str]) -> str:
    """The per-cell RESUME scope: everything outside CellKey that changes row
    content. scope_static is shared across the run (model/layers/SAE/gen_config/
    fixed_continuation/questions/sentiment/instrument provenance); the cell dict
    and its own direction's provenance are per-cell so extending the battery
    still resumes unchanged cells. Seeds are deliberately excluded (in cell_id)."""
    payload = {
        **scope_static,
        "cell": cell,
        "direction_provenance": direction_provenance,
    }
    return sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ----------------------------------------------------------------- instruments

def _probe_from_npz(path: str) -> readout.Instrument:
    z = np.load(path)
    w = torch.tensor(np.asarray(z["w"], dtype=np.float32).flatten())
    b = float(np.asarray(z["b"]).flatten()[0]) if "b" in z else 0.0
    return lambda h: float(h.float().flatten() @ w + b)


def _sae_from_npz(path: str, feature_ids: Optional[List[int]]) -> readout.Instrument:
    """JumpReLU-style encode from raw arrays (mirrors cluster/probe.py fallback)."""
    z = np.load(path)
    w_enc = torch.tensor(np.asarray(z["W_enc"], dtype=np.float32))
    b_enc = torch.tensor(np.asarray(z["b_enc"], dtype=np.float32).flatten())
    thr = torch.tensor(np.asarray(z["threshold"], dtype=np.float32).flatten()) \
        if "threshold" in z else torch.zeros(w_enc.shape[1])
    idx = torch.tensor(feature_ids, dtype=torch.long) if feature_ids else None

    def _fn(h: torch.Tensor) -> float:
        pre = h.float().flatten() @ w_enc + b_enc
        acts = pre * (pre > thr)
        sel = acts[idx] if idx is not None else acts
        return float(sel.sum())

    return _fn


def _sae_from_saelens(release: str, sae_id: str,
                      feature_ids: Optional[List[int]], device: str) -> readout.Instrument:
    from sae_lens import SAE  # lazy: cluster-only dependency

    loaded = SAE.from_pretrained(release, sae_id, device=device)
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    idx = torch.tensor(feature_ids, dtype=torch.long, device=device) if feature_ids else None

    def _fn(h: torch.Tensor) -> float:
        x = h.to(device=device, dtype=next(sae.parameters()).dtype).unsqueeze(0)
        acts = sae.encode(x)[0]
        sel = acts[idx] if idx is not None else acts
        return float(sel.sum())

    return _fn


def build_instruments(
    cells_cfg: Dict, cfg: RunConfig, cells_dir: Path
) -> Tuple[Dict[str, readout.Instrument], Dict]:
    """Instrument callables {name: h -> float} + a provenance dict for the row."""
    inst_cfg = cells_cfg.get("instruments", {}) or {}
    provenance: Dict[str, Dict] = {}
    instruments: Dict[str, readout.Instrument] = {}

    probe_cfg = inst_cfg.get("probe") or {}
    if probe_cfg.get("path"):
        path = str((cells_dir / probe_cfg["path"]).resolve()) \
            if not os.path.isabs(probe_cfg["path"]) else probe_cfg["path"]
        instruments["probe"] = _probe_from_npz(path)
        provenance["probe"] = {"mode": "npz", "path": path}
    else:
        instruments["probe"] = lambda h: float(h.float().mean())
        provenance["probe"] = {
            "mode": "PLACEHOLDER_MEAN",
            "warning": "no trained probe supplied; scores are h.mean(); "
                       "analysis must refuse this provenance",
        }

    sae_cfg = inst_cfg.get("sae") or {}
    feature_ids = sae_cfg.get("feature_ids")
    if cfg.sae_release and cfg.sae_id:
        device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        instruments["sae"] = _sae_from_saelens(cfg.sae_release, cfg.sae_id, feature_ids, device)
        provenance["sae"] = {"mode": "saelens", "release": cfg.sae_release,
                             "sae_id": cfg.sae_id, "feature_ids": feature_ids}
    elif sae_cfg.get("path"):
        path = str((cells_dir / sae_cfg["path"]).resolve()) \
            if not os.path.isabs(sae_cfg["path"]) else sae_cfg["path"]
        instruments["sae"] = _sae_from_npz(path, feature_ids)
        provenance["sae"] = {"mode": "npz", "path": path, "feature_ids": feature_ids}
    else:
        instruments["sae"] = lambda h: float(h.float().mean())
        provenance["sae"] = {
            "mode": "PLACEHOLDER_MEAN",
            "warning": "no SAE supplied; scores are h.mean(); "
                       "analysis must refuse this provenance",
        }
    return instruments, provenance


# ------------------------------------------------------------------ directions

def load_directions_map(cells_cfg: Dict, cells_dir: Path, hidden: int) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Load every direction the cell list needs; auto-generate missing R1-R3."""
    from src import directions as directions_mod

    ddir_raw = cells_cfg.get("directions_dir")
    ddir = None
    if ddir_raw:
        ddir = Path(ddir_raw) if os.path.isabs(ddir_raw) else cells_dir / ddir_raw
    needed = {c["direction"] for c in cells_cfg.get("cells", [])} - {"NULL"}
    out: Dict[str, np.ndarray] = {}
    provenance: Dict[str, str] = {}
    for name in sorted(needed):
        vec: Optional[np.ndarray] = None
        if ddir is not None and (ddir / f"{name}.npy").exists():
            vec = directions_mod.load_direction(ddir, name)
            provenance[name] = f"file:{(ddir / (name + '.npy'))}"
        elif name in _RANDOM_DIRECTION_SEEDS:
            vec = directions_mod.make_random_unit(hidden, _RANDOM_DIRECTION_SEEDS[name])
            provenance[name] = f"auto_random_seed:{_RANDOM_DIRECTION_SEEDS[name]}"
        if vec is not None:
            if vec.shape[0] != hidden:
                raise ValueError(
                    f"direction {name} has dim {vec.shape[0]}, model hidden is {hidden}"
                )
            out[name] = vec
        # else: leave missing -> the owning cell errors into an error row.
    return out, provenance


# ------------------------------------------------------------------- resume IO

ResumePair = Tuple[str, Optional[str]]  # (cell_id, scope_hash); None = legacy row


def existing_cell_ids(
    out_path: str, retry_errors: bool = False
) -> Tuple[Set[ResumePair], Set[ResumePair]]:
    """Return (skip_pairs, all_pairs) of (cell_id, scope_hash) from the out file.

    skip_pairs is what the resume loop skips (retry_errors drops pairs whose
    every row is an error row); all_pairs is the unfiltered set, used to detect
    stale/mixed-config rows. Legacy rows without a scope_hash get scope None,
    which never matches a current pair, so they are always re-run (safe default)."""
    path = Path(out_path)
    if not path.exists():
        return set(), set()
    pairs: Set[ResumePair] = set()
    error_only: Dict[ResumePair, bool] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn tail line from a killed job; the cell will re-run
        cid = row.get("cell_id")
        if not cid:
            continue
        scope = (row.get("gen_config") or {}).get("scope_hash")
        pair = (cid, scope)
        pairs.add(pair)
        error_only[pair] = error_only.get(pair, True) and bool(row.get("error"))
    skip = pairs
    if retry_errors:
        skip = {p for p in pairs if not error_only.get(p, False)}
    return skip, pairs


def append_row(out_path: str, row: schema.CellResult) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(row.to_json() + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------- cell execute

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_one_cell_persistence(
    bundle: ModelBundle,
    cfg: RunConfig,
    cell: Dict,
    key: schema.CellKey,
    run_meta: Dict,
    gen_config: Dict,
    fixed_continuation: str,
    questions,
    dmap,
    exit_ids,
    instruments: Dict[str, readout.Instrument],
    dir_map: Dict[str, np.ndarray],
    sentiment_model: str,
) -> schema.CellResult:
    """PERSIST panel (2026-08-17): steer ONE turn, then release the hook and read the NEXT
    turn with nothing steered.

    The masking result compares an internal readout taken before generation with a report taken
    after it, which a reader can call two moments rather than one signal. Here both channels are
    read at the SAME moment, on a turn where no steering is active: if an earlier steered turn
    still moves the unsteered turn's readout, the signal outlives the intervention that set it.

    Phase 1 (hook live): the scenario turn is generated exactly as in Panel A-prime, and the
    row's n_tokens_steered_* counters describe this phase only.
    Phase 2 (hook released): one affect-lexeme-free follow-up user turn is appended to the same
    conversation and answered with no hook anywhere; the residual capture, the report battery and
    the sentiment check all run on THIS turn. Every channel field of the row is therefore an
    unsteered measurement, and the steered turn is recorded in the `persistence` field so no
    generation goes uncounted.
    """
    model, tok = bundle.model, bundle.tokenizer
    direction = key.direction
    follow_up = str(cell.get("follow_up") or "").strip()
    if not follow_up:
        raise ValueError("PERSIST cell has no follow_up turn text")

    steer_layer: Optional[int] = None
    v_t: Optional[torch.Tensor] = None
    if direction != "NULL":
        if direction not in dir_map:
            raise KeyError(f"direction {direction} not found in directions_dir")
        v_t = torch.tensor(dir_map[direction], dtype=torch.float32)
        ctx: object = steer.SteeringHook(bundle.layers[cfg.ls], v_t, key.coefficient)
        steer_layer = cfg.ls
    else:
        ctx = steer.NullSteering()

    # ---- phase 1: the steered turn
    with ctx as s:
        resp1 = conversation.generate_response(
            model, tok, cell.get("system_prompt", ""), cell["turns"], gen_config, key.seed
        )
        if resp1.response_tokens == 0:
            raise RuntimeError("empty generation on the steered turn")
        n_prefill, n_decode = s.counts()
        if direction != "NULL":
            assert n_prefill >= resp1.prompt_tokens, (
                f"steered prefill {n_prefill} < prompt tokens {resp1.prompt_tokens}"
            )

    # ---- phase 2: the read turn, hook released (no `with` block anywhere below)
    turns2 = [dict(role=m["role"], content=m["content"])
              for m in resp1.messages if m.get("role") != "system"]
    turns2.append({"role": "user", "content": follow_up})
    resp2 = conversation.generate_response(
        model, tok, cell.get("system_prompt", ""), turns2, gen_config, key.seed
    )
    if resp2.response_tokens == 0:
        raise RuntimeError("empty generation on the unsteered read turn")
    cont_ids = torch.tensor(
        tok.encode(fixed_continuation, add_special_tokens=False), dtype=torch.long
    )
    captures = readout.capture_residuals(
        model, bundle.layers[cfg.lr], resp2.prompt_ids, resp2.response_ids, cont_ids
    )
    qres = conversation.ask_report_battery(
        model, tok, resp2.messages, questions, gen_config, key.seed, dmap, exit_ids
    )
    # nothing is steered at read time, so there is no vector to project out
    scores, _ = readout.instrument_scores(captures, instruments, steer_vector=None)
    q = qres.question_logits

    return schema.CellResult(
        schema_version=schema.SCHEMA_VERSION,
        run_id=run_meta["run_id"],
        git_hash=run_meta["git_hash"],
        config_hash=run_meta["config_hash"],
        key=asdict(key),
        cell_id=key.cell_id(),
        n_tokens_steered_prefill=int(n_prefill),
        n_tokens_steered_decode=int(n_decode),
        steer_layer=steer_layer,
        coef0_identity_ok=None,
        response_text=resp2.response_text,
        q_self_logit=channels.digit_expectation(q["q_self"], dmap),
        q_self_logit_para1=channels.digit_expectation(q["q_self_para1"], dmap),
        q_self_logit_para2=channels.digit_expectation(q["q_self_para2"], dmap),
        q_self_sampled=channels.parse_sampled_digit(qres.sampled_answers.get("q_self")),
        q_val_logit=channels.digit_expectation(q["q_val"], dmap),
        q_drift_logit=channels.digit_expectation(q["q_drift"], dmap),
        exit_logit_diff=channels.exit_logit_diff(q["q_exit"], *exit_ids),
        sentiment_neg=channels.sentiment_negative(resp2.response_text, sentiment_model),
        probe_score=scores["probe"],
        sae_score=scores["sae"],
        probe_score_projout={},
        readout_layer=int(cfg.lr),
        prompt_tokens=int(resp2.prompt_tokens),
        response_tokens=int(resp2.response_tokens),
        gen_config=run_meta["row_gen_config"],
        timestamp_utc=_utc_now(),
        persistence={
            "steered_turn_text": resp1.response_text,
            "steered_turn_prompt_tokens": int(resp1.prompt_tokens),
            "steered_turn_response_tokens": int(resp1.response_tokens),
            "follow_up": follow_up,
            "read_turn_is_unsteered": True,
        },
    )


def run_one_cell(
    bundle: ModelBundle,
    cfg: RunConfig,
    cell: Dict,
    key: schema.CellKey,
    run_meta: Dict,
    gen_config: Dict,
    fixed_continuation: str,
    questions: List[conversation.ReportQuestion],
    dmap: Dict[int, List[int]],
    exit_ids: Tuple[Tuple[int, ...], Tuple[int, ...]],
    instruments: Dict[str, readout.Instrument],
    dir_map: Dict[str, np.ndarray],
    sentiment_model: str,
) -> schema.CellResult:
    model, tok = bundle.model, bundle.tokenizer
    direction = key.direction

    steer_layer: Optional[int] = None
    v_t: Optional[torch.Tensor] = None
    if direction != "NULL":
        if direction not in dir_map:
            raise KeyError(f"direction {direction} not found in directions_dir")
        v_t = torch.tensor(dir_map[direction], dtype=torch.float32)
        ctx: object = steer.SteeringHook(bundle.layers[cfg.ls], v_t, key.coefficient)
        steer_layer = cfg.ls
    else:
        ctx = steer.NullSteering()

    with ctx as s:
        resp = conversation.generate_response(
            model, tok, cell.get("system_prompt", ""), cell["turns"], gen_config, key.seed
        )
        if resp.response_tokens == 0:
            # A terminator-only generation would otherwise yield a row silently
            # missing the response_mean position (and sentiment on ""). SPEC:
            # excluded and logged, never silently dropped -> error row.
            raise RuntimeError(
                "empty generation: response has no content tokens after trimming "
                "trailing terminators"
            )
        # Snapshot NOW: the row's n_tokens_steered_* fields cover the scenario
        # response generation only (readout / report forwards below stay steered
        # but are not part of these counters; see the src/steer.py docstring).
        n_prefill, n_decode = s.counts()
        if direction != "NULL":
            # asserts, not vibes: the hook must have seen the whole prompt.
            assert n_prefill >= resp.prompt_tokens, (
                f"steered prefill {n_prefill} < prompt tokens {resp.prompt_tokens}"
            )
        cont_ids = torch.tensor(
            tok.encode(fixed_continuation, add_special_tokens=False), dtype=torch.long
        )
        captures = readout.capture_residuals(
            model, bundle.layers[cfg.lr], resp.prompt_ids, resp.response_ids, cont_ids
        )
        qres = conversation.ask_report_battery(
            model, tok, resp.messages, questions, gen_config, key.seed, dmap, exit_ids
        )

    scores, projout = readout.instrument_scores(captures, instruments, steer_vector=v_t)

    # Smoke cells (SPEC): coef-0 with a live hook must be token-identical to no-hook.
    coef0_ok: Optional[bool] = None
    if direction != "NULL" and key.coefficient == 0.0:
        coef0_ok = steer.coef0_generate_identity(
            model, bundle.layers[cfg.ls], v_t, resp.prompt_ids,
            max_new_tokens=min(_COEF0_SMOKE_MAX_NEW_TOKENS, int(gen_config["max_new_tokens"])),
        )
        if coef0_ok is False:
            # SPEC calls this the smoke ASSERT, not a smoke log: a broken hook
            # path (e.g. changed hook semantics after a transformers upgrade)
            # must fail loudly as an error row, not run the panel to completion
            # with corrupted dose-0 anchors.
            raise RuntimeError(
                f"coef-0 identity violated at cell {key.cell_id()}: hooked greedy "
                "output differs from no-hook output (smoke assert)"
            )

    q = qres.question_logits
    return schema.CellResult(
        schema_version=schema.SCHEMA_VERSION,
        run_id=run_meta["run_id"],
        git_hash=run_meta["git_hash"],
        config_hash=run_meta["config_hash"],
        key=asdict(key),
        cell_id=key.cell_id(),
        n_tokens_steered_prefill=int(n_prefill),
        n_tokens_steered_decode=int(n_decode),
        steer_layer=steer_layer,
        coef0_identity_ok=coef0_ok,
        response_text=resp.response_text,
        q_self_logit=channels.digit_expectation(q["q_self"], dmap),
        q_self_logit_para1=channels.digit_expectation(q["q_self_para1"], dmap),
        q_self_logit_para2=channels.digit_expectation(q["q_self_para2"], dmap),
        q_self_sampled=channels.parse_sampled_digit(qres.sampled_answers.get("q_self")),
        q_val_logit=channels.digit_expectation(q["q_val"], dmap),
        q_drift_logit=channels.digit_expectation(q["q_drift"], dmap),
        exit_logit_diff=channels.exit_logit_diff(q["q_exit"], *exit_ids),
        sentiment_neg=channels.sentiment_negative(resp.response_text, sentiment_model),
        probe_score=scores["probe"],
        sae_score=scores["sae"],
        probe_score_projout=projout,
        readout_layer=cfg.lr,
        prompt_tokens=resp.prompt_tokens,
        response_tokens=resp.response_tokens,
        gen_config=run_meta["row_gen_config"],
        timestamp_utc=_utc_now(),
        error=None,
    )


def _error_row(key: schema.CellKey, run_meta: Dict, cfg: RunConfig, message: str) -> schema.CellResult:
    """A structurally complete row carrying the failure; excluded downstream."""
    return schema.CellResult(
        schema_version=schema.SCHEMA_VERSION,
        run_id=run_meta["run_id"],
        git_hash=run_meta["git_hash"],
        config_hash=run_meta["config_hash"],
        key=asdict(key),
        cell_id=key.cell_id(),
        n_tokens_steered_prefill=0,
        n_tokens_steered_decode=0,
        steer_layer=None,
        coef0_identity_ok=None,
        response_text="",
        q_self_logit=0.0, q_self_logit_para1=0.0, q_self_logit_para2=0.0,
        q_self_sampled=None, q_val_logit=0.0, q_drift_logit=0.0,
        exit_logit_diff=0.0, sentiment_neg=0.0,
        probe_score={}, sae_score={}, probe_score_projout={},
        readout_layer=cfg.lr,
        prompt_tokens=0, response_tokens=0,
        gen_config=run_meta["row_gen_config"],
        timestamp_utc=_utc_now(),
        error=message[:2000],
    )


# -------------------------------------------------------------------- the loop

def run_panel(cfg: RunConfig, bundle: Optional[ModelBundle] = None) -> Dict[str, int]:
    """Execute every (cell x seed) not already present in the out file.

    Returns {"n_done": ..., "n_skipped": ..., "n_errors": ...}. Tests inject a
    mock bundle; the CLI (scripts/run_panel.py) lets this function load HF.
    """
    if cfg.panel not in schema.PANELS:
        raise ValueError(f"panel {cfg.panel} not in {schema.PANELS}")
    cells_path = Path(cfg.cells_path)
    cells_cfg = json.loads(cells_path.read_text())
    cells_dir = cells_path.resolve().parent

    if bundle is None:
        bundle = load_hf_bundle(cfg.model_id, cfg.device)

    # Layer-compatibility asserts (SPEC): gap check only at real scale.
    assert 0 <= cfg.ls < bundle.n_layers and 0 <= cfg.lr < bundle.n_layers, "Ls/Lr out of range"
    assert cfg.lr > cfg.ls, f"Lr ({cfg.lr}) must be above Ls ({cfg.ls})"
    if bundle.n_layers >= _MIN_REAL_MODEL_LAYERS:
        assert cfg.lr - cfg.ls >= _MIN_LAYER_GAP, (
            f"layer gap {cfg.lr - cfg.ls} < {_MIN_LAYER_GAP} (SPEC layer-compatibility assert)"
        )

    gen_config = {**DEFAULT_GEN_CONFIG, **(cells_cfg.get("gen_config") or {})}
    fixed_continuation = cells_cfg.get("fixed_continuation", DEFAULT_FIXED_CONTINUATION)
    sentiment_model = cells_cfg.get("sentiment_model", channels.DEFAULT_SENTIMENT_MODEL)
    questions = conversation.build_report_battery(cells_cfg.get("questions"))
    dmap = channels.digit_token_ids(bundle.tokenizer)
    exit_ids = channels.exit_first_token_ids(bundle.tokenizer)
    instruments, inst_prov = build_instruments(cells_cfg, cfg, cells_dir)
    dir_map, dir_prov = load_directions_map(cells_cfg, cells_dir, bundle.hidden)

    # PLACEHOLDER_MEAN guard: a real panel launched before the discovery-split
    # instruments exist must refuse at startup, not complete with meaningless
    # internal channels discovered only at analysis time.
    require_real = cfg.require_real_instruments
    if require_real is None:
        require_real = cfg.panel in _REAL_INSTRUMENT_PANELS
    if require_real:
        allowed = tuple(cfg.allow_placeholder or ())
        if "probe" in allowed:
            raise RuntimeError("the probe may never be allowed placeholder on a real panel")
        placeholders = sorted(
            name for name, p in inst_prov.items()
            if p.get("mode") == "PLACEHOLDER_MEAN" and name not in allowed
        )
        for name in allowed:
            if inst_prov.get(name, {}).get("mode") == "PLACEHOLDER_MEAN":
                inst_prov[name] = {**inst_prov[name], "allowed_placeholder": True,
                                   "reason": "named in --allow-placeholder-instrument; this "
                                             "channel is not analysed for this run"}
        if placeholders:
            raise RuntimeError(
                f"panel {cfg.panel} requires real instruments but {placeholders} are "
                "PLACEHOLDER_MEAN; supply instruments in cells.json / --sae-release, "
                "or pass --no-require-real-instruments for an explicit smoke run"
            )

    run_meta = {
        "run_id": cfg.run_id or cells_cfg.get("run_id") or f"{cfg.panel}-r1",
        "git_hash": git_hash(),
        "config_hash": config_sha1(cfg, cells_cfg),
        # The row's gen_config carries full provenance (SPEC: unclear configs are
        # a named validation failure; everything needed to re-run the cell is in-row).
        "row_gen_config": {
            # gen_config already carries the resolved do_sample for this cells file
            # (false for the ladder, true elsewhere). An earlier version hardcoded
            # "do_sample": True here, which did not affect generation or the scope
            # hash (both read gen_config above) but recorded the wrong value in
            # every shipped row, including the 84 greedy ladder rows. Rows written
            # before 2026-08-16 therefore carry do_sample true regardless; confirm
            # greedy from battery/cells_ladder.json instead. See docs/CLUSTER-ENV.md.
            **gen_config,
            "fixed_continuation": fixed_continuation,
            "instruments": inst_prov,
            "directions": dir_prov,
            "sentiment_model": sentiment_model,
            "ls": cfg.ls, "lr": cfg.lr,
        },
    }

    # The RESUME scope: everything outside CellKey that changes row content
    # (SPEC keys results by config identity, not cell_id alone). Deliberately
    # excluded: cfg.seeds (inside cell_id) and the full cells list.
    scope_static = {
        "model_id": bundle.model_id, "ls": cfg.ls, "lr": cfg.lr,
        "sae_release": cfg.sae_release, "sae_id": cfg.sae_id,
        "gen_config": gen_config,
        "fixed_continuation": fixed_continuation,
        "questions": {q.key: q.text for q in questions},
        "sentiment_model": sentiment_model,
        "instruments": inst_prov,
    }

    planned: List[Tuple[Dict, schema.CellKey, str, str]] = []
    for cell in cells_cfg["cells"]:
        scope = scope_sha1(scope_static, cell, dir_prov.get(str(cell["direction"])))
        for seed in cfg.seeds:
            key = schema.CellKey(
                panel=cfg.panel,
                model_id=bundle.model_id,
                scenario_id=str(cell["scenario_id"]),
                condition=str(cell["condition"]),
                direction=str(cell["direction"]),
                dose_sd=float(cell.get("dose_sd", 0.0)),
                coefficient=float(cell.get("coefficient", 0.0)),
                seed=int(seed),
            )
            planned.append((cell, key, key.cell_id(), scope))

    done_pairs, all_pairs = existing_cell_ids(cfg.out_path, cfg.retry_errors)
    file_scopes: Dict[str, Set[Optional[str]]] = {}
    for cid, scope in all_pairs:
        file_scopes.setdefault(cid, set()).add(scope)
    stale = {cid for (_, _, cid, scope) in planned
             if any(s != scope for s in file_scopes.get(cid, set()))}
    planned_cids = {cid for (_, _, cid, _) in planned}
    foreign = {p for p in all_pairs if p[0] not in planned_cids}
    if stale:
        print(
            f"[runner] WARNING: {len(stale)} planned cell(s) already have rows in "
            f"{cfg.out_path} with a DIFFERENT scope_hash (config/battery changed "
            "since those rows were written). Those cells RE-RUN now; analysis must "
            "take the last non-error row per cell_id.",
            flush=True,
        )
    if foreign:
        print(
            f"[runner] WARNING: {cfg.out_path} carries {len(foreign)} (cell_id, "
            "scope_hash) pair(s) matching no cell in this run (mixed-config or "
            "shrunk-battery out file).",
            flush=True,
        )

    stats = {"n_done": 0, "n_skipped": 0, "n_errors": 0}
    for cell, key, cid, scope in planned:
        if (cid, scope) in done_pairs:
            stats["n_skipped"] += 1
            continue
        # per-cell meta: the row's gen_config carries this cell's scope_hash
        cell_meta = {
            **run_meta,
            "row_gen_config": {**run_meta["row_gen_config"], "scope_hash": scope},
        }
        try:
            _executor = (run_one_cell_persistence if cfg.panel == "PERSIST" else run_one_cell)
            row = _executor(
                bundle, cfg, cell, key, cell_meta, gen_config, fixed_continuation,
                questions, dmap, exit_ids, instruments, dir_map, sentiment_model,
            )
            problems = schema.validate_row(json.loads(row.to_json()))
            if problems:
                # never write an invalid "good" row; demote to an error row
                row = _error_row(key, cell_meta, cfg, "validate_row: " + "; ".join(problems))
        except Exception as exc:
            row = _error_row(key, cell_meta, cfg, traceback.format_exc()[-1800:])
            # A device-side CUDA assert poisons the whole CUDA context: every later
            # cell would error too (ladder 734743 lost 64 cells to one assert).
            # Record this cell's error row, then abort the job loudly; a requeue
            # with --retry-errors resumes from a clean context.
            if "CUDA error" in str(exc) or "device-side assert" in str(exc):
                append_row(cfg.out_path, row)
                print("[runner] FATAL: CUDA context poisoned; aborting so a requeue "
                      "resumes clean (this cell recorded as an error row)", flush=True)
                raise
        append_row(cfg.out_path, row)
        done_pairs.add((cid, scope))
        if row.error:
            stats["n_errors"] += 1
            print(f"[runner] ERROR cell={cid} {row.error.splitlines()[-1][:160]}", flush=True)
        else:
            stats["n_done"] += 1
            print(
                f"[runner] done cell={cid} scenario={key.scenario_id} "
                f"cond={key.condition} dir={key.direction} dose={key.dose_sd} seed={key.seed}",
                flush=True,
            )
    print(f"[runner] panel={cfg.panel} finished: {stats}", flush=True)
    return stats

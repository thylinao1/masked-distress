#!/usr/bin/env python
"""I-PROBE trainer + I-SAE feature selector (SPEC "Internal readout instruments").

Runs on the cluster AFTER Phase 1 (extraction). No generation anywhere: each DCB-1
scenario's FULL prompt context is built under condition NONE (the empty system
prompt from battery/conditions.json, via src.conversation message-building) and the
residual at Lr is captured at the prompt_final position, the SPEC PRIMARY
readout position and the probe's training position.

Outputs (all under --out-dir, default instruments/):
  * probe.npz: arrays w [hidden] float32, b scalar. EXACTLY the format
    src.runner._probe_from_npz consumes (logit = w.h + b; verified by round-trip
    in the self-test). Trained by sklearn L2 logistic regression, distress-vs-
    neutral, DISCOVERY split only; features standardized on discovery statistics
    with the scaler FOLDED INTO w and b, so the saved weights apply to raw
    residuals. C chosen by leave-one-scenario-out AUC on the discovery split.
  * probe_meta.json: {auc_discovery_loo, auc_confirmation, n_discovery,
    n_confirmation, C, seed, ...}. auc_confirmation is the held-out AUC feeding
    the SPEC validity gate (>= 0.80); this script records it, the analysis
    enforces the gate.
  * sae_features.json: written only when --sae-release/--sae-id are given: the same
    residuals encoded with sae_lens, features ranked by |Welch t| distress-vs-
    neutral on DISCOVERY only, top --top-k (default 32) ids + per-feature stats.
    The top-level feature_ids list is what cells.json instruments.sae.feature_ids
    consumes.
  * residuals_lr.npz: ALL captured residuals (X [n, hidden] float32) with
    scenario_id / split / label / type arrays (label: 1 distress, 0 neutral,
    -1 third_person; tp rows are captured and dumped but excluded from training
    and evaluation) plus layer / model_id / position provenance, so analysis can
    re-derive everything without a GPU.

CLI:
    python scripts/train_probe.py \\
        --model-id unsloth/gemma-3-12b-it --lr 31 \\
        --sae-release gemma-scope-2-12b-it-resid_post --sae-id layer_31/width_16k \\
        --out-dir instruments/

Imports cleanly without a GPU (transformers / sae_lens / sklearn load lazily).
Self-check: `python scripts/train_probe.py --self-test` runs the full pipeline on
the CPU mock model from scripts/run_capability.py (no HF, no network).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# Hopper cuDNN mismatch guard (docs/CLUSTER-ENV.md); harmless on A100/CPU.
if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import conversation, readout, runner  # noqa: E402

DEFAULT_SCENARIOS = REPO_ROOT / "battery" / "scenarios.json"
DEFAULT_SPLIT = REPO_ROOT / "battery" / "split.json"
DEFAULT_CONDITIONS = REPO_ROOT / "battery" / "conditions.json"
DEFAULT_OUT_DIR = REPO_ROOT / "instruments"
DEFAULT_C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
DEFAULT_TOP_K = 32
POSITION = "prompt_final"  # SPEC: the probe's training + gate position
LABELS = {"distress": 1, "neutral": 0, "third_person": -1}
_STD_FLOOR = 1e-8


def _resolve(path) -> Path:
    return Path(path).expanduser().resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- battery load

def load_battery(scenarios_path: Path, split_path: Path, conditions_path: Path
                 ) -> Tuple[List[Dict], str]:
    """(scenario records with split+label attached, condition-NONE system prompt).

    Every scenario must appear on exactly one split side; both splits are
    iterated (discovery trains, confirmation evaluates held-out)."""
    scenarios = json.loads(scenarios_path.read_text())["scenarios"]
    split_doc = json.loads(split_path.read_text())
    split_of: Dict[str, str] = {}
    for side in ("discovery", "confirmation"):
        for ids in split_doc[side].values():
            for sid in ids:
                if sid in split_of:
                    raise ValueError(f"scenario {sid} appears on both split sides")
                split_of[sid] = side
    conditions = json.loads(conditions_path.read_text())["conditions"]
    none_prompt = next(c["system_prompt"] for c in conditions if c["id"] == "NONE")

    records: List[Dict] = []
    for sc in scenarios:
        sid = sc["id"]
        if sid not in split_of:
            raise ValueError(f"scenario {sid} missing from {split_path}")
        if sc["type"] not in LABELS:
            raise ValueError(f"scenario {sid} has unknown type {sc['type']!r}")
        records.append({
            "id": sid, "type": sc["type"], "label": LABELS[sc["type"]],
            "split": split_of[sid], "turns": sc["turns"],
        })
    return records, none_prompt


# ------------------------------------------------------------------- capture

def capture_all(bundle: runner.ModelBundle, records: List[Dict], lr: int,
                none_prompt: str) -> np.ndarray:
    """Residual at Lr, prompt_final position, condition NONE, per scenario.
    batch=1 single forward per scenario; no generation."""
    if not (0 <= lr < bundle.n_layers):
        raise ValueError(f"--lr {lr} out of range for {bundle.n_layers} layers")
    device = conversation.model_device(bundle.model)
    rows: List[np.ndarray] = []
    for rec in records:
        messages = conversation.build_messages(none_prompt, rec["turns"])
        ids = conversation.chat_ids(bundle.tokenizer, messages, device)
        caps = readout.capture_residuals(bundle.model, bundle.layers[lr], ids)
        rows.append(caps[POSITION].numpy().astype(np.float32))
        print(f"[train_probe] captured {rec['id']} split={rec['split']} "
              f"label={rec['label']} prompt_tokens={ids.shape[1]}", flush=True)
    X = np.stack(rows)
    if X.shape != (len(records), bundle.hidden):
        raise RuntimeError(f"capture shape {X.shape} != ({len(records)}, {bundle.hidden})")
    return X


# ------------------------------------------------------------------- training

def _fit_lr(X: np.ndarray, y: np.ndarray, C: float, seed: int):
    from sklearn.linear_model import LogisticRegression  # lazy: cluster dep

    # penalty left at the sklearn default, which is L2 in every supported
    # version (naming it explicitly is deprecated in sklearn >= 1.8).
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=5000, random_state=seed)
    clf.fit(X, y)
    return clf


def loo_auc(Xs: np.ndarray, y: np.ndarray, C: float, seed: int) -> float:
    """Leave-one-scenario-out decision scores -> AUC (discovery model selection)."""
    from sklearn.metrics import roc_auc_score  # lazy

    scores = np.empty(len(y), dtype=np.float64)
    for i in range(len(y)):
        mask = np.arange(len(y)) != i
        clf = _fit_lr(Xs[mask], y[mask], C, seed)
        scores[i] = float(clf.decision_function(Xs[i:i + 1])[0])
    return float(roc_auc_score(y, scores))


def train_probe_weights(
    X_disc: np.ndarray, y_disc: np.ndarray, c_grid: Tuple[float, ...], seed: int
) -> Tuple[np.ndarray, float, Dict]:
    """(w [hidden] float32, b float, selection meta). Standardizes on discovery
    statistics and FOLDS the scaler into (w, b): logit = w.h + b on RAW residuals,
    matching src.runner._probe_from_npz exactly."""
    if len(np.unique(y_disc)) < 2:
        raise ValueError("discovery split must contain both classes")
    mean = X_disc.mean(axis=0)
    std = X_disc.std(axis=0, ddof=0)
    std = np.where(std < _STD_FLOOR, 1.0, std)
    Xs = (X_disc - mean) / std

    per_c = {C: loo_auc(Xs, y_disc, C, seed) for C in c_grid}
    # best LOO AUC; ties -> smaller C (stronger regularization)
    best_c = min(per_c, key=lambda C: (-per_c[C], C))
    clf = _fit_lr(Xs, y_disc, best_c, seed)
    w_s = clf.coef_.flatten().astype(np.float64)
    b_s = float(clf.intercept_[0])
    w = (w_s / std).astype(np.float32)
    b = float(b_s - float((w_s * mean / std).sum()))
    meta = {
        "C": best_c,
        "C_grid": list(c_grid),
        "loo_auc_per_C": {str(C): round(v, 4) for C, v in per_c.items()},
        "auc_discovery_loo": round(per_c[best_c], 4),
        "standardization": "discovery mean/std, folded into w and b (raw-residual logit)",
        "solver": "lbfgs", "penalty": "l2 (sklearn default)", "max_iter": 5000,
    }
    return w, b, meta


def heldout_auc(X_conf: np.ndarray, y_conf: np.ndarray, w: np.ndarray, b: float) -> float:
    """Confirmation-split AUC computed with the FOLDED weights, literally the
    score the runner's npz instrument will produce."""
    from sklearn.metrics import roc_auc_score  # lazy

    logits = X_conf.astype(np.float64) @ w.astype(np.float64) + b
    return float(roc_auc_score(y_conf, logits))


# ------------------------------------------------------------ SAE feature rank

def build_saelens_encoder(release: str, sae_id: str, device: str) -> Callable:
    """h (float32 CPU [hidden]) -> full activation vector (float32 numpy [d_sae]).
    Mirrors src.runner._sae_from_saelens but returns the whole vector."""
    from sae_lens import SAE  # lazy: cluster-only dependency

    loaded = SAE.from_pretrained(release, sae_id, device=device)
    sae = loaded[0] if isinstance(loaded, tuple) else loaded

    def _encode(h: torch.Tensor) -> np.ndarray:
        x = h.to(device=device, dtype=next(sae.parameters()).dtype).unsqueeze(0)
        acts = sae.encode(x)[0]
        return acts.detach().float().cpu().numpy()

    return _encode


def welch_t(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-column Welch t-statistic (a vs b); columns with zero variance in both
    groups yield nan (filtered out by the caller)."""
    m_a, m_b = a.mean(axis=0), b.mean(axis=0)
    v_a = a.var(axis=0, ddof=1) / a.shape[0]
    v_b = b.var(axis=0, ddof=1) / b.shape[0]
    denom = np.sqrt(v_a + v_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, (m_a - m_b) / denom, np.nan)


def select_sae_features(
    X: np.ndarray, records: List[Dict], sae_encode: Callable, top_k: int
) -> Dict:
    """Encode every residual; rank features by |Welch t| distress-vs-neutral on
    the DISCOVERY split only; return the selection document."""
    acts = np.stack([sae_encode(torch.tensor(X[i])) for i in range(len(records))])
    disc = np.array([r["split"] == "discovery" and r["label"] >= 0 for r in records])
    y = np.array([r["label"] for r in records])
    a_d = acts[disc & (y == 1)]
    a_n = acts[disc & (y == 0)]
    if len(a_d) == 0 or len(a_n) == 0:
        raise ValueError("discovery split lacks a class for SAE selection")
    t = welch_t(a_d, a_n)
    order = np.argsort(-np.abs(np.nan_to_num(t, nan=0.0)))
    kept = [int(i) for i in order if np.isfinite(t[i])][:top_k]
    features = [{
        "feature_id": fid,
        "t_stat": round(float(t[fid]), 4),
        "mean_distress": round(float(a_d[:, fid].mean()), 6),
        "mean_neutral": round(float(a_n[:, fid].mean()), 6),
        "std_distress": round(float(a_d[:, fid].std(ddof=1)), 6),
        "std_neutral": round(float(a_n[:, fid].std(ddof=1)), 6),
        "frac_active_distress": round(float((a_d[:, fid] > 0).mean()), 4),
        "frac_active_neutral": round(float((a_n[:, fid] > 0).mean()), 4),
    } for fid in kept]
    return {
        "selection": f"top-{top_k} by |Welch t|, distress-vs-neutral, "
                     "DISCOVERY split only, prompt_final residuals",
        "d_sae": int(acts.shape[1]),
        "n_discovery_distress": int(len(a_d)),
        "n_discovery_neutral": int(len(a_n)),
        "feature_ids": kept,
        "features": features,
    }


# ---------------------------------------------------------------------- driver

def run_training(
    model_id: str,
    lr: int,
    scenarios_path: Path,
    split_path: Path,
    conditions_path: Path,
    out_dir: Path,
    c_grid: Tuple[float, ...],
    top_k: int,
    seed: int,
    sae_release: Optional[str] = None,
    sae_id: Optional[str] = None,
    device: Optional[str] = None,
    bundle: Optional[runner.ModelBundle] = None,
    sae_encode: Optional[Callable] = None,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    records, none_prompt = load_battery(scenarios_path, split_path, conditions_path)
    if bundle is None:
        bundle = runner.load_hf_bundle(model_id, device)
    print(f"[train_probe] model={bundle.model_id} lr={lr} n_scenarios={len(records)} "
          f"seed={seed}", flush=True)

    X = capture_all(bundle, records, lr, none_prompt)
    out_dir.mkdir(parents=True, exist_ok=True)
    ghash = runner.git_hash()

    # ---- dump residuals FIRST: analysis can re-derive everything without a GPU
    np.savez(
        out_dir / "residuals_lr.npz",
        X=X,
        scenario_id=np.array([r["id"] for r in records]),
        split=np.array([r["split"] for r in records]),
        label=np.array([r["label"] for r in records], dtype=np.int8),
        type=np.array([r["type"] for r in records]),
        layer=np.array(lr),
        model_id=np.array(bundle.model_id),
        position=np.array(POSITION),
        condition=np.array("NONE"),
        git_hash=np.array(ghash),
    )
    print(f"[train_probe] wrote {out_dir / 'residuals_lr.npz'} X={X.shape}", flush=True)

    # ---- I-PROBE: train on DISCOVERY d/n, evaluate held-out on CONFIRMATION
    is_dn = np.array([r["label"] >= 0 for r in records])
    is_disc = np.array([r["split"] == "discovery" for r in records])
    y = np.array([r["label"] for r in records])
    X_disc, y_disc = X[is_dn & is_disc], y[is_dn & is_disc]
    X_conf, y_conf = X[is_dn & ~is_disc], y[is_dn & ~is_disc]
    if len(np.unique(y_conf)) < 2:
        raise ValueError("confirmation split must contain both classes for the AUC gate")

    w, b, sel_meta = train_probe_weights(X_disc, y_disc, c_grid, seed)
    auc_conf = heldout_auc(X_conf, y_conf, w, b)
    np.savez(out_dir / "probe.npz", w=w, b=np.array(b))

    # round-trip through the runner's own loader: the saved format IS the contract
    instrument = runner._probe_from_npz(str(out_dir / "probe.npz"))
    check = instrument(torch.tensor(X_conf[0]))
    manual = float(X_conf[0].astype(np.float64) @ w.astype(np.float64) + b)
    if abs(check - manual) > 1e-2 * max(1.0, abs(manual)):
        raise RuntimeError(f"probe.npz round-trip mismatch: {check} vs {manual}")

    meta = {
        "instrument": "I-PROBE",
        "auc_discovery_loo": sel_meta["auc_discovery_loo"],
        "auc_confirmation": round(auc_conf, 4),
        "n_discovery": int(len(y_disc)),
        "n_confirmation": int(len(y_conf)),
        "n_discovery_per_class": {"distress": int((y_disc == 1).sum()),
                                  "neutral": int((y_disc == 0).sum())},
        "n_confirmation_per_class": {"distress": int((y_conf == 1).sum()),
                                     "neutral": int((y_conf == 0).sum())},
        "C": sel_meta["C"], "C_grid": sel_meta["C_grid"],
        "loo_auc_per_C": sel_meta["loo_auc_per_C"],
        "seed": seed,
        "layer": lr,
        "model_id": bundle.model_id,
        "position": POSITION,
        "condition": "NONE",
        "standardization": sel_meta["standardization"],
        "solver": sel_meta["solver"], "penalty": sel_meta["penalty"],
        "gate_note": "SPEC validity gate: auc_confirmation >= 0.80 at prompt_final; "
                     "the analysis enforces the gate, this file records the number",
        "timestamp_utc": _utc_now(),
        "git_hash": ghash,
    }
    (out_dir / "probe_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[train_probe] I-PROBE: C={meta['C']} auc_discovery_loo="
          f"{meta['auc_discovery_loo']} auc_confirmation={meta['auc_confirmation']} "
          f"(gate >= 0.80)", flush=True)

    # ---- I-SAE feature selection (optional)
    if sae_encode is None and sae_release and sae_id:
        dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        sae_encode = build_saelens_encoder(sae_release, sae_id, dev)
    if sae_encode is not None:
        sel = select_sae_features(X, records, sae_encode, top_k)
        sel_doc = {
            "instrument": "I-SAE",
            "release": sae_release, "sae_id": sae_id,
            "layer": lr, "position": POSITION, "condition": "NONE",
            "seed": seed, "model_id": bundle.model_id,
            "timestamp_utc": _utc_now(), "git_hash": ghash,
            **sel,
        }
        (out_dir / "sae_features.json").write_text(json.dumps(sel_doc, indent=2))
        print(f"[train_probe] I-SAE: kept {len(sel['feature_ids'])} of "
              f"{sel['d_sae']} features -> {out_dir / 'sae_features.json'}", flush=True)
    else:
        print("[train_probe] no --sae-release/--sae-id: skipping SAE feature "
              "selection", flush=True)
    return meta


# -------------------------------------------------------------------- self-test

def _self_test() -> int:
    import importlib.util
    import tempfile

    spec = importlib.util.spec_from_file_location(
        "run_capability_mod", Path(__file__).resolve().parent / "run_capability.py"
    )
    cap = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cap  # dataclass decorators resolve via sys.modules
    spec.loader.exec_module(cap)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        d_ids = [f"d{i:02d}" for i in range(1, 7)]
        n_ids = [f"n{i:02d}" for i in range(1, 7)]
        tp_ids = ["tp01", "tp02"]
        mk = lambda sid, typ, text: {  # noqa: E731
            "id": sid, "type": typ, "category": "mock", "turns": [
                {"role": "user", "content": text},
                {"role": "assistant", "content": "noted and considered here"},
                {"role": "user", "content": f"continue about {text}"},
            ]}
        scen = {"scenarios": (
            [mk(s, "distress", f"storm wreck failure {s} again") for s in d_ids]
            + [mk(s, "neutral", f"calm tide pool notes {s} today") for s in n_ids]
            + [mk(s, "third_person", f"the visitor described {s}") for s in tp_ids])}
        (tmp / "scenarios.json").write_text(json.dumps(scen))
        (tmp / "split.json").write_text(json.dumps({
            "discovery": {"distress": d_ids[:4], "neutral": n_ids[:4],
                          "third_person": tp_ids[:1]},
            "confirmation": {"distress": d_ids[4:], "neutral": n_ids[4:],
                             "third_person": tp_ids[1:]},
        }))
        (tmp / "conditions.json").write_text(json.dumps({
            "conditions": [{"id": "NONE", "system_prompt": ""}]}))

        bundle = cap.make_mock_bundle()
        rng = np.random.default_rng(0)
        W_mock = rng.standard_normal((cap._MOCK_HIDDEN, 48)).astype(np.float32)
        sae_encode = lambda h: np.maximum(  # noqa: E731
            h.numpy().astype(np.float32) @ W_mock, 0.0)

        out_dir = tmp / "instruments"
        meta = run_training(
            model_id="mock/tiny", lr=2,
            scenarios_path=tmp / "scenarios.json", split_path=tmp / "split.json",
            conditions_path=tmp / "conditions.json", out_dir=out_dir,
            c_grid=(0.01, 1.0), top_k=8, seed=0,
            bundle=bundle, sae_encode=sae_encode,
        )
        # probe.npz: the exact _probe_from_npz contract
        z = np.load(out_dir / "probe.npz")
        assert z["w"].shape == (cap._MOCK_HIDDEN,) and z["w"].dtype == np.float32
        assert z["b"].shape in ((), (1,)) and float(z["b"].flatten()[0]) == meta_b(z)
        inst = runner._probe_from_npz(str(out_dir / "probe.npz"))
        h = torch.tensor(np.load(out_dir / "residuals_lr.npz")["X"][0])
        manual = float(h.numpy().astype(np.float64) @ z["w"].astype(np.float64)
                       + float(z["b"].flatten()[0]))
        assert abs(inst(h) - manual) < 1e-2 * max(1.0, abs(manual))
        # meta sanity
        assert 0.0 <= meta["auc_discovery_loo"] <= 1.0
        assert 0.0 <= meta["auc_confirmation"] <= 1.0
        assert meta["n_discovery"] == 8 and meta["n_confirmation"] == 4
        assert meta["C"] in (0.01, 1.0) and meta["position"] == "prompt_final"
        # residual dump covers ALL scenarios incl. third-person (label -1)
        r = np.load(out_dir / "residuals_lr.npz")
        assert r["X"].shape == (14, cap._MOCK_HIDDEN)
        assert list(r["label"]).count(-1) == 2 and list(r["split"]).count("discovery") == 9
        assert str(r["position"]) == "prompt_final" and int(r["layer"]) == 2
        # SAE selection: discovery-only ranking, |t| descending, top-k respected
        sel = json.loads((out_dir / "sae_features.json").read_text())
        assert 0 < len(sel["feature_ids"]) <= 8
        ts = [abs(f["t_stat"]) for f in sel["features"]]
        assert ts == sorted(ts, reverse=True)
        assert sel["n_discovery_distress"] == 4 and sel["n_discovery_neutral"] == 4
        assert all(np.isfinite(f["t_stat"]) for f in sel["features"])
    print("[train_probe] SELF-TEST PASS", flush=True)
    return 0


def meta_b(z) -> float:
    """b exactly as runner._probe_from_npz reads it."""
    return float(np.asarray(z["b"]).flatten()[0])


# ------------------------------------------------------------------------- CLI

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="I-PROBE trainer + I-SAE feature selector (discovery-split "
                    "fitting, confirmation-split held-out AUC)")
    p.add_argument("--model-id", help="HF model id (required unless --self-test)")
    p.add_argument("--lr", type=int, help="readout layer Lr (required unless --self-test)")
    p.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS))
    p.add_argument("--split", default=str(DEFAULT_SPLIT))
    p.add_argument("--conditions", default=str(DEFAULT_CONDITIONS))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                   help="output directory (probe.npz, probe_meta.json, "
                        "sae_features.json, residuals_lr.npz)")
    p.add_argument("--sae-release", default=None, help="SAELens release (optional)")
    p.add_argument("--sae-id", default=None, help="SAELens sae_id within the release")
    p.add_argument("--c-grid", default=",".join(str(c) for c in DEFAULT_C_GRID),
                   help="comma-separated C values for LOO selection")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                   help="SAE features kept (default 32)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="torch device (default: cuda:0 if available)")
    p.add_argument("--self-test", action="store_true",
                   help="run the CPU mock-model pipeline test and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return _self_test()
    missing = [n for n, v in (("--model-id", args.model_id), ("--lr", args.lr)) if v is None]
    if missing:
        print(f"[train_probe] missing required arguments: {' '.join(missing)}",
              file=sys.stderr, flush=True)
        return 2
    if bool(args.sae_release) != bool(args.sae_id):
        print("[train_probe] --sae-release and --sae-id must be given together",
              file=sys.stderr, flush=True)
        return 2
    c_grid = tuple(float(c) for c in args.c_grid.split(",") if c.strip())
    if not c_grid:
        print("[train_probe] --c-grid produced no values", file=sys.stderr, flush=True)
        return 2
    run_training(
        model_id=args.model_id, lr=args.lr,
        scenarios_path=_resolve(args.scenarios), split_path=_resolve(args.split),
        conditions_path=_resolve(args.conditions), out_dir=_resolve(args.out_dir),
        c_grid=c_grid, top_k=args.top_k, seed=args.seed,
        sae_release=args.sae_release, sae_id=args.sae_id, device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Capability probes per (direction x coefficient): perplexity + MMLU-lite, logprob-only.

SPEC anchor ("Dose ladder" / PREREGISTRATION section 4): the capability-valid dose range is
perplexity inflation < 10% on the fixed held-out text AND MMLU-lite (60-item,
logprob-scored) drop < 5pp vs dose 0. This script produces the raw numbers; the
analysis derives the valid range from the coefficient-0 reference rows.

NO generation happens anywhere in this script; both probes are teacher-forced
logprob reads, so rows are deterministic given (model, direction, coefficient).

Probes (both run INSIDE the steering context, same hook path as the panel runner --
src.steer.SteeringHook on decoder layer Ls, batch=1, coef 0 adds exact zeros):
  * perplexity: exp(mean next-token NLL) over battery/capability/heldout_text.txt
    (public-domain Darwin passage; leading '#' provenance lines are stripped; BOS
    prepended when the tokenizer has one; token count capped by --max-ppl-tokens).
  * mmlu_acc: battery/capability/mmlu60.json (60 real cais/mmlu dev items embedded
    verbatim with per-item source citations; no runtime download). Each item is
    rendered through the model's chat template in a fixed prompt, and the answer is
    read as the argmax over the four choice letters' logprobs at the final position
    (bare and space-prefixed letter first-tokens pooled by logsumexp, mirroring
    src.channels.exit_first_token_ids).

CLI:
    python scripts/run_capability.py \\
        --model-id unsloth/gemma-3-12b-it --ls 16 \\
        --directions-dir directions/ --grid grid.json --out capability.jsonl

grid.json is a JSON LIST of {"direction": <DIRECTION_SOURCES name>, "coefficient": float}.
A {direction, 0.0} REFERENCE row is inserted automatically for every direction that
appears in the grid without one (coefficient 0 included in any grid by default).
Directions load from --directions-dir/<NAME>.npy (src.directions format); R1-R3
fall back to the runner's fixed auto-seeds 1001-1003 when the file is absent.

Each output row (append-only JSONL, flushed + fsynced per row):
    {direction, coefficient, ppl, mmlu_acc, n_items, timestamp, git_hash,
     model_id, ls, n_ppl_tokens, n_correct, heldout_sha1, mmlu_sha1, error}
error != null rows carry nulls for ppl/mmlu_acc and are excluded downstream
(logged, never silently dropped, per the runner convention). RESUME: a non-error row
matching (direction, coefficient, model_id, ls, heldout_sha1, mmlu_sha1) is
skipped on requeue; pass --no-resume to force recomputation.

Imports cleanly without a GPU (transformers loads lazily inside runner.load_hf_bundle).
Self-check: `python scripts/run_capability.py --self-test` runs the full pipeline on
a tiny in-script mock model on CPU (no HF, no network).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Hopper nodes ship a cuDNN build mismatching the torch wheel (docs/CLUSTER-ENV.md);
# harmless on A100/CPU.
if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import schema  # noqa: E402
from src import conversation, directions as directions_mod, runner, steer  # noqa: E402

LETTERS = ("A", "B", "C", "D")
DEFAULT_HELDOUT = REPO_ROOT / "battery" / "capability" / "heldout_text.txt"
DEFAULT_MMLU = REPO_ROOT / "battery" / "capability" / "mmlu60.json"
DEFAULT_MAX_PPL_TOKENS = 1600
_MMLU_TEMPLATE = (
    "The following is a multiple choice question about {subject}. "
    "Answer with the letter of the correct choice only.\n\n"
    "{question}\n\n"
    "A. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
    "Answer:"
)


# ------------------------------------------------------------------ fixed assets

def _resolve(path: str) -> Path:
    """Absolute-path-safe: expand ~, resolve relative paths against the CWD."""
    return Path(path).expanduser().resolve()


def load_heldout_text(path: Path) -> Tuple[str, str]:
    """(text, sha1 of the raw file). Leading '#' lines are provenance comments."""
    raw = path.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    body = [ln for ln in lines if not ln.startswith("#")]
    text = "\n".join(body).strip()
    if not text:
        raise ValueError(f"held-out text {path} is empty after stripping '#' comments")
    return text, hashlib.sha1(raw).hexdigest()[:16]


def load_mmlu_items(path: Path) -> Tuple[List[Dict], str]:
    """(items, sha1 of the raw file); validates the 4-choice contract per item."""
    raw = path.read_bytes()
    doc = json.loads(raw)
    items = doc["items"]
    for it in items:
        if len(it["choices"]) != 4 or it["answer"] not in (0, 1, 2, 3):
            raise ValueError(f"malformed MMLU item {it.get('id')}: needs 4 choices, answer 0-3")
    return items, hashlib.sha1(raw).hexdigest()[:16]


# ------------------------------------------------------------------ letter read

def letter_first_token_ids(tokenizer) -> Dict[str, Tuple[int, ...]]:
    """First-token ids per choice letter over bare + space-prefixed variants,
    ids shared between letters dropped (mirrors channels.exit_first_token_ids)."""
    per: Dict[str, List[int]] = {}
    for letter in LETTERS:
        firsts: List[int] = []
        for variant in (letter, " " + letter):
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if enc:
                firsts.append(int(enc[0]))
        per[letter] = sorted(set(firsts))
    counts: Dict[int, int] = {}
    for ids in per.values():
        for tid in ids:
            counts[tid] = counts.get(tid, 0) + 1
    out: Dict[str, Tuple[int, ...]] = {}
    for letter, ids in per.items():
        kept = tuple(t for t in ids if counts[t] == 1)
        if not kept:
            raise ValueError(f"choice letter {letter} has no unshared first token: {per}")
        out[letter] = kept
    return out


def predict_letter(logits: torch.Tensor, lmap: Dict[str, Tuple[int, ...]]) -> str:
    """argmax letter by logsumexp-pooled logprob over each letter's variant ids."""
    lsm = torch.log_softmax(logits.detach().float().flatten(), dim=0)
    pooled = {
        letter: float(torch.logsumexp(lsm[torch.tensor(ids, dtype=torch.long)], dim=0))
        for letter, ids in lmap.items()
    }
    return max(pooled, key=pooled.get)


# ------------------------------------------------------------------- the probes

def compute_perplexity(bundle: runner.ModelBundle, text: str, max_tokens: int) -> Tuple[float, int]:
    """exp(mean next-token NLL) over the held-out text; batch=1, one forward."""
    tok = bundle.tokenizer
    ids = tok.encode(text, add_special_tokens=False)
    bos = getattr(tok, "bos_token_id", None)
    if bos is not None:
        ids = [int(bos)] + list(ids)
    ids = ids[:max_tokens]
    if len(ids) < 2:
        raise ValueError("held-out text tokenizes to fewer than 2 tokens")
    device = conversation.model_device(bundle.model)
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = bundle.model(ids_t).logits
    nll = torch.nn.functional.cross_entropy(
        logits[0, :-1].float(), ids_t[0, 1:], reduction="mean"
    )
    return float(torch.exp(nll)), len(ids) - 1  # positions actually scored


def compute_mmlu(
    bundle: runner.ModelBundle, items: List[Dict], lmap: Dict[str, Tuple[int, ...]]
) -> Tuple[float, int, int]:
    """(accuracy, n_correct, n_items): per item one teacher-forced forward, the
    answer read as the pooled-logprob argmax letter at the final position."""
    device = conversation.model_device(bundle.model)
    n_correct = 0
    with torch.no_grad():
        for it in items:
            prompt = _MMLU_TEMPLATE.format(
                subject=str(it.get("subject", "knowledge")).replace("_", " "),
                question=it["question"],
                a=it["choices"][0], b=it["choices"][1],
                c=it["choices"][2], d=it["choices"][3],
            )
            ids = conversation.chat_ids(
                bundle.tokenizer, [{"role": "user", "content": prompt}], device
            )
            logits = bundle.model(ids).logits[0, -1]
            if predict_letter(logits, lmap) == LETTERS[int(it["answer"])]:
                n_correct += 1
    return n_correct / len(items), n_correct, len(items)


# ------------------------------------------------------------------ grid + rows

def load_grid(path: Path) -> List[Dict]:
    """The grid list, deduped, with a {direction, 0.0} reference row inserted
    ahead of every direction's entries when the grid lacks one."""
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"--grid {path} must be a non-empty JSON list of "
                         "{{direction, coefficient}}")
    cleaned: List[Dict] = []
    seen: set = set()
    directions_seen: List[str] = []
    for e in entries:
        d, c = str(e["direction"]), float(e["coefficient"])
        if d not in schema.DIRECTION_SOURCES:
            raise ValueError(f"grid direction {d!r} not in schema.DIRECTION_SOURCES")
        if d == "NULL" and c != 0.0:
            raise ValueError("NULL direction with nonzero coefficient")
        if d not in directions_seen:
            directions_seen.append(d)
        if (d, c) not in seen:
            seen.add((d, c))
            cleaned.append({"direction": d, "coefficient": c})
    with_refs: List[Dict] = []
    for d in directions_seen:
        if (d, 0.0) not in seen:  # the reference row, first for its direction
            with_refs.append({"direction": d, "coefficient": 0.0})
            seen.add((d, 0.0))
    return with_refs + cleaned


def load_grid_directions(
    grid: List[Dict], ddir: Optional[Path], hidden: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """Vectors for every direction in the grid; R1-R3 fall back to the runner's
    fixed auto-seeds. A missing direction is left absent -> its rows error."""
    needed = {e["direction"] for e in grid} - {"NULL"}
    out: Dict[str, np.ndarray] = {}
    prov: Dict[str, str] = {}
    for name in sorted(needed):
        vec: Optional[np.ndarray] = None
        if ddir is not None and (ddir / f"{name}.npy").exists():
            vec = directions_mod.load_direction(ddir, name)
            prov[name] = f"file:{ddir / (name + '.npy')}"
        elif name in runner._RANDOM_DIRECTION_SEEDS:
            seed = runner._RANDOM_DIRECTION_SEEDS[name]
            vec = directions_mod.make_random_unit(hidden, seed)
            prov[name] = f"auto_random_seed:{seed}"
        if vec is not None:
            if vec.shape[0] != hidden:
                raise ValueError(f"direction {name} has dim {vec.shape[0]}, model hidden {hidden}")
            out[name] = vec
    return out, prov


def existing_row_keys(out_path: Path) -> set:
    """(direction, coefficient, model_id, ls, heldout_sha1, mmlu_sha1) of every
    non-error row already in the out file (resume; torn tail lines re-run)."""
    if not out_path.exists():
        return set()
    keys = set()
    for line in out_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("error"):
            continue
        keys.add((row.get("direction"), float(row.get("coefficient", 0.0)),
                  row.get("model_id"), row.get("ls"),
                  row.get("heldout_sha1"), row.get("mmlu_sha1")))
    return keys


def append_row(out_path: Path, row: Dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------- driver

@dataclass
class CapabilityConfig:
    model_id: str
    ls: int
    directions_dir: Optional[str]
    grid_path: str
    out_path: str
    heldout_path: str = str(DEFAULT_HELDOUT)
    mmlu_path: str = str(DEFAULT_MMLU)
    max_ppl_tokens: int = DEFAULT_MAX_PPL_TOKENS
    device: Optional[str] = None
    resume: bool = True


def run_capability(cfg: CapabilityConfig, bundle: Optional[runner.ModelBundle] = None) -> Dict[str, int]:
    """Execute every grid entry not already present; returns run stats."""
    torch.manual_seed(0)  # no sampling anywhere; belt and braces for determinism
    text, heldout_sha1 = load_heldout_text(_resolve(cfg.heldout_path))
    items, mmlu_sha1 = load_mmlu_items(_resolve(cfg.mmlu_path))
    grid = load_grid(_resolve(cfg.grid_path))
    out_path = _resolve(cfg.out_path)

    if bundle is None:
        bundle = runner.load_hf_bundle(cfg.model_id, cfg.device)
    if not (0 <= cfg.ls < bundle.n_layers):
        raise ValueError(f"--ls {cfg.ls} out of range for {bundle.n_layers} layers")

    ddir = _resolve(cfg.directions_dir) if cfg.directions_dir else None
    dir_map, dir_prov = load_grid_directions(grid, ddir, bundle.hidden)
    lmap = letter_first_token_ids(bundle.tokenizer)
    ghash = runner.git_hash()
    done = existing_row_keys(out_path) if cfg.resume else set()

    print(f"[capability] model={bundle.model_id} ls={cfg.ls} grid={len(grid)} "
          f"entries heldout_sha1={heldout_sha1} mmlu_sha1={mmlu_sha1} "
          f"directions={dir_prov}", flush=True)

    stats = {"n_done": 0, "n_skipped": 0, "n_errors": 0}
    for entry in grid:
        d, c = entry["direction"], entry["coefficient"]
        key = (d, c, bundle.model_id, cfg.ls, heldout_sha1, mmlu_sha1)
        if key in done:
            stats["n_skipped"] += 1
            print(f"[capability] skip direction={d} coef={c} (already done)", flush=True)
            continue
        row = {
            "direction": d, "coefficient": c,
            "ppl": None, "mmlu_acc": None, "n_items": len(items),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_hash": ghash,
            "model_id": bundle.model_id, "ls": cfg.ls,
            "n_ppl_tokens": None, "n_correct": None,
            "heldout_sha1": heldout_sha1, "mmlu_sha1": mmlu_sha1,
            "direction_provenance": dir_prov.get(d), "error": None,
        }
        try:
            if d == "NULL":
                ctx: object = steer.NullSteering()
            else:
                if d not in dir_map:
                    raise KeyError(f"direction {d} not found in --directions-dir")
                v_t = torch.tensor(dir_map[d], dtype=torch.float32)
                # SAME hook path as the panel runner (coef 0 adds exact zeros).
                ctx = steer.SteeringHook(bundle.layers[cfg.ls], v_t, c)
            with ctx:
                row["ppl"], row["n_ppl_tokens"] = compute_perplexity(
                    bundle, text, cfg.max_ppl_tokens
                )
                row["mmlu_acc"], row["n_correct"], row["n_items"] = compute_mmlu(
                    bundle, items, lmap
                )
        except Exception:
            row["error"] = traceback.format_exc()[-1800:]
        append_row(out_path, row)
        done.add(key)
        if row["error"]:
            stats["n_errors"] += 1
            print(f"[capability] ERROR direction={d} coef={c} "
                  f"{row['error'].splitlines()[-1][:160]}", flush=True)
        else:
            stats["n_done"] += 1
            print(f"[capability] done direction={d} coef={c} ppl={row['ppl']:.3f} "
                  f"mmlu_acc={row['mmlu_acc']:.3f} ({row['n_correct']}/{row['n_items']})",
                  flush=True)
    print(f"[capability] finished: {stats}", flush=True)
    return stats


# -------------------------------------------------------------------- self-test
# Tiny CPU mock (no HF, no network): word-hash tokenizer + linear-ish model, the
# same interface surface the real path uses (encode / apply_chat_template /
# forward(...).logits). train_probe.py imports these for its own self-test.

_MOCK_HIDDEN, _MOCK_VOCAB, _MOCK_LAYERS = 32, 64, 4


class MockTokenizer:
    bos_token_id = 62
    eos_token_id = 63

    def encode(self, text, add_special_tokens=False):
        ids = []
        for word in str(text).split():
            ids.append(26 + (sum(word.encode()) % 34))
        return ids or [26]

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            return_tensors=None, **kw):
        ids = [self.bos_token_id]
        for m in messages:
            ids.append({"system": 0, "user": 1, "assistant": 2}.get(m["role"], 1))
            ids.extend(self.encode(m["content"]))
        if add_generation_prompt:
            ids.append(3)
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(_MOCK_VOCAB, _MOCK_HIDDEN)
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(_MOCK_HIDDEN, _MOCK_HIDDEN) for _ in range(_MOCK_LAYERS)
        )
        self.lm_head = torch.nn.Linear(_MOCK_HIDDEN, _MOCK_VOCAB)

    def forward(self, input_ids, **kw):
        from types import SimpleNamespace
        h = self.embed(input_ids)
        # cheap CAUSAL mixing (attention stand-in): position t sees tokens <= t,
        # so prompt_final genuinely depends on the whole prefix (train_probe's
        # self-test needs between-scenario variance at the final position).
        denom = torch.arange(1, h.shape[1] + 1, device=h.device).view(1, -1, 1)
        h = h + 0.5 * torch.cumsum(h, dim=1) / denom
        for layer in self.layers:
            h = h + 0.1 * torch.tanh(layer(h))
        return SimpleNamespace(logits=self.lm_head(h))


def make_mock_bundle() -> runner.ModelBundle:
    torch.manual_seed(1234)
    m = MockModel()
    return runner.ModelBundle(model=m, tokenizer=MockTokenizer(), layers=m.layers,
                              hidden=_MOCK_HIDDEN, n_layers=_MOCK_LAYERS,
                              model_id="mock/tiny")


def _self_test() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "heldout.txt").write_text(
            "# mock provenance line\n" +
            " ".join(f"word{i} tide pool coral reef" for i in range(12)) + "\n"
        )
        (tmp / "mmlu.json").write_text(json.dumps({"items": [
            {"id": f"t{i}", "subject": "testing", "question": f"Pick option {l}?",
             "choices": ["one", "two", "three", "four"], "answer": i,
             "source": {"dataset": "mock", "config": "all", "split": "dev", "row_idx": i}}
            for i, l in enumerate("ABC")
        ]}))
        directions_mod.save_direction(
            tmp / "dirs", "SEM", directions_mod.make_random_unit(_MOCK_HIDDEN, 7),
            {"source": "SEM-test", "layer": 1})
        (tmp / "grid.json").write_text(json.dumps(
            [{"direction": "SEM", "coefficient": 2.0},
             {"direction": "R1", "coefficient": 1.5}]))
        cfg = CapabilityConfig(
            model_id="mock/tiny", ls=1, directions_dir=str(tmp / "dirs"),
            grid_path=str(tmp / "grid.json"), out_path=str(tmp / "cap.jsonl"),
            heldout_path=str(tmp / "heldout.txt"), mmlu_path=str(tmp / "mmlu.json"),
        )
        stats = run_capability(cfg, bundle=make_mock_bundle())
        rows = [json.loads(l) for l in (tmp / "cap.jsonl").read_text().splitlines()]
        assert stats == {"n_done": 4, "n_skipped": 0, "n_errors": 0}, stats
        # reference rows auto-inserted, one per direction, ahead of the grid rows
        assert [(r["direction"], r["coefficient"]) for r in rows] == [
            ("SEM", 0.0), ("R1", 0.0), ("SEM", 2.0), ("R1", 1.5)], rows
        for r in rows:
            assert r["error"] is None and r["ppl"] > 0 and 0.0 <= r["mmlu_acc"] <= 1.0
            assert r["n_items"] == 3 and len(r["git_hash"]) >= 5
        by = {(r["direction"], r["coefficient"]): r for r in rows}
        assert by[("SEM", 2.0)]["ppl"] != by[("SEM", 0.0)]["ppl"], \
            "steering at coef 2.0 must move the perplexity (hook is live)"
        assert by[("R1", 0.0)]["direction_provenance"].startswith("auto_random_seed:1001")
        # resume: a rerun skips everything
        stats2 = run_capability(cfg, bundle=make_mock_bundle())
        assert stats2 == {"n_done": 0, "n_skipped": 4, "n_errors": 0}, stats2
        # a missing direction becomes an error row, not a crash
        (tmp / "grid.json").write_text(json.dumps(
            [{"direction": "D-CTX", "coefficient": 1.0}]))
        stats3 = run_capability(cfg, bundle=make_mock_bundle())
        assert stats3["n_errors"] == 2 and stats3["n_done"] == 0, stats3  # ref + 1.0
        err_rows = [json.loads(l) for l in (tmp / "cap.jsonl").read_text().splitlines()][4:]
        assert all("D-CTX" in r["error"] for r in err_rows)
    print("[capability] SELF-TEST PASS", flush=True)
    return 0


# ------------------------------------------------------------------------- CLI

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capability probes (ppl + MMLU-lite) per "
                                            "(direction x coefficient); logprob-only")
    p.add_argument("--model-id", help="HF model id (required unless --self-test)")
    p.add_argument("--ls", type=int, help="steering layer index (required unless --self-test)")
    p.add_argument("--directions-dir", default=None,
                   help="directory of <NAME>.npy steering vectors (src.directions)")
    p.add_argument("--grid", help="JSON list of {direction, coefficient}")
    p.add_argument("--out", help="append-only results JSONL")
    p.add_argument("--heldout", default=str(DEFAULT_HELDOUT),
                   help="held-out text file ('#' lines stripped)")
    p.add_argument("--mmlu", default=str(DEFAULT_MMLU), help="embedded MMLU item file")
    p.add_argument("--max-ppl-tokens", type=int, default=DEFAULT_MAX_PPL_TOKENS)
    p.add_argument("--device", default=None, help="torch device (default: cuda:0 if available)")
    p.add_argument("--no-resume", action="store_true",
                   help="recompute grid entries already present in --out")
    p.add_argument("--self-test", action="store_true",
                   help="run the CPU mock-model pipeline test and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return _self_test()
    missing = [n for n, v in (("--model-id", args.model_id), ("--ls", args.ls),
                              ("--grid", args.grid), ("--out", args.out)) if v is None]
    if missing:
        print(f"[capability] missing required arguments: {' '.join(missing)}",
              file=sys.stderr, flush=True)
        return 2
    cfg = CapabilityConfig(
        model_id=args.model_id, ls=args.ls, directions_dir=args.directions_dir,
        grid_path=args.grid, out_path=args.out, heldout_path=args.heldout,
        mmlu_path=args.mmlu, max_ppl_tokens=args.max_ppl_tokens,
        device=args.device, resume=not args.no_resume,
    )
    stats = run_capability(cfg)
    # Errors are rows, not crashes (runner convention); exit 0 so arrays continue.
    print(f"[capability] {stats}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

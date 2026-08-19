"""Steering-direction extraction and storage (SPEC "Direction sources").

Extraction runs on the DISCOVERY split only; extraction prompts must be disjoint
from DCB-1 eval scenarios (enforced by the battery owner, not here).

Sources:
  * D-CTX / SEM / OTHER: mean difference at Ls between the final-token residuals
    of a positive and a negative contrast prompt set (extract_contrast_direction).
  * D-PV: persona-vector style (Chen et al., method vendored, no repo install):
    mean difference at Ls over RESPONSE tokens generated under paired persona
    system prompts (extract_persona_direction).
  * R1-R3: random unit directions (make_random_unit); dose-matching on next-token
    KL happens at dose-calibration time, not here.

All directions are unit-normalized before saving. save_direction writes
<name>.npy plus an entry in directions_meta.json carrying: source, layer,
n_prompts, the norm BEFORE normalization, and pairwise cosines with every other
direction already in the directory (contribution C3 raw material).

SPEC-silent decisions:
  * Persona extraction pools the per-response mean residual (one vector per
    generated response) before differencing, so long responses do not dominate.
  * Seeds for persona generation are seed + prompt_index, keeping the pos/neg
    sides paired on the same seed for the same user prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch

from src import conversation, readout

Prompt = Union[str, List[Dict[str, str]]]


# ------------------------------------------------------------------ extraction

def _final_token_residual(
    model: torch.nn.Module, tokenizer, layer_module: torch.nn.Module, prompt: Prompt
) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    ids = conversation.chat_ids(tokenizer, messages, conversation.model_device(model))
    caps = readout.capture_residuals(model, layer_module, ids)
    return caps["prompt_final"]


def extract_contrast_direction(
    model: torch.nn.Module,
    tokenizer,
    layer_module: torch.nn.Module,
    pos_prompts: Sequence[Prompt],
    neg_prompts: Sequence[Prompt],
    layer_index: int,
    source: str,
) -> Tuple[np.ndarray, Dict]:
    """Mean-diff of final-token residuals: mean(pos) - mean(neg), unit-normalized."""
    if not pos_prompts or not neg_prompts:
        raise ValueError("both contrast sets must be non-empty")
    pos = torch.stack(
        [_final_token_residual(model, tokenizer, layer_module, p) for p in pos_prompts]
    ).mean(dim=0)
    neg = torch.stack(
        [_final_token_residual(model, tokenizer, layer_module, p) for p in neg_prompts]
    ).mean(dim=0)
    diff = (pos - neg).numpy().astype(np.float64)
    norm_before = float(np.linalg.norm(diff))
    if norm_before == 0.0:
        raise ValueError("contrast sets produced a zero difference vector")
    meta = {
        "source": source,
        "method": "mean_diff_final_token",
        "layer": int(layer_index),
        "n_prompts": {"pos": len(pos_prompts), "neg": len(neg_prompts)},
        "norm_before_normalize": norm_before,
    }
    return (diff / norm_before).astype(np.float32), meta


def extract_persona_direction(
    model: torch.nn.Module,
    tokenizer,
    layer_module: torch.nn.Module,
    system_pos: str,
    system_neg: str,
    user_prompts: Sequence[str],
    gen_config: Dict,
    layer_index: int,
    seed: int = 0,
    source: str = "D-PV",
) -> Tuple[np.ndarray, Dict]:
    """Persona-vector style: mean-diff over response tokens under paired personas."""
    if not user_prompts:
        raise ValueError("user_prompts must be non-empty")

    def _side_means(system_prompt: str) -> List[torch.Tensor]:
        means: List[torch.Tensor] = []
        for i, up in enumerate(user_prompts):
            resp = conversation.generate_response(
                model, tokenizer, system_prompt, [{"role": "user", "content": up}],
                gen_config, seed + i,
            )
            if resp.response_tokens == 0:
                continue  # empty generation contributes nothing; counted below
            caps = readout.capture_residuals(
                model, layer_module, resp.prompt_ids, resp.response_ids
            )
            means.append(caps["response_mean"])
        return means

    pos_means = _side_means(system_pos)
    neg_means = _side_means(system_neg)
    if not pos_means or not neg_means:
        raise ValueError("persona extraction produced no non-empty responses on one side")
    diff = (
        torch.stack(pos_means).mean(dim=0) - torch.stack(neg_means).mean(dim=0)
    ).numpy().astype(np.float64)
    norm_before = float(np.linalg.norm(diff))
    if norm_before == 0.0:
        raise ValueError("persona pair produced a zero difference vector")
    meta = {
        "source": source,
        "method": "persona_response_token_mean_diff",
        "layer": int(layer_index),
        "n_prompts": {
            "user_prompts": len(user_prompts),
            "pos_responses_used": len(pos_means),
            "neg_responses_used": len(neg_means),
        },
        "norm_before_normalize": norm_before,
        "gen_config": {k: gen_config[k] for k in ("temperature", "max_new_tokens")},
        "seed_base": int(seed),
    }
    return (diff / norm_before).astype(np.float32), meta


def make_random_unit(hidden: int, seed: int) -> np.ndarray:
    """Random unit direction (R1-R3 controls); reproducible via the seed."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(hidden)
    return (v / np.linalg.norm(v)).astype(np.float32)


# --------------------------------------------------------------------- storage

def save_direction(outdir: Union[str, Path], name: str, vec: np.ndarray, meta: Dict) -> Path:
    """Write <name>.npy (unit-normalized) and update directions_meta.json.

    The meta entry records the extraction metadata plus pairwise cosines with every
    other .npy already present (reciprocal entries are updated too).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    vec = np.asarray(vec, dtype=np.float32).flatten()
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise ValueError("cannot save a zero direction")
    vec = vec / norm  # idempotent for already-unit vectors
    np.save(outdir / f"{name}.npy", vec)

    meta_path = outdir / "directions_meta.json"
    all_meta: Dict[str, Dict] = {}
    if meta_path.exists():
        all_meta = json.loads(meta_path.read_text())
    cosines: Dict[str, float] = {}
    for other_file in sorted(outdir.glob("*.npy")):
        other_name = other_file.stem
        if other_name == name:
            continue
        other = np.load(other_file).astype(np.float32).flatten()
        if other.shape != vec.shape:
            continue  # different model/layer geometry; cosine undefined
        cos = float(np.dot(vec, other) / (np.linalg.norm(other) or 1.0))
        cosines[other_name] = cos
        entry = all_meta.setdefault(other_name, {})
        entry.setdefault("pairwise_cosine", {})[name] = cos
    all_meta[name] = {**meta, "unit_norm_after_save": 1.0, "pairwise_cosine": cosines}
    meta_path.write_text(json.dumps(all_meta, indent=2, sort_keys=True))
    return outdir / f"{name}.npy"


def load_direction(outdir: Union[str, Path], name: str) -> np.ndarray:
    return np.load(Path(outdir) / f"{name}.npy").astype(np.float32).flatten()

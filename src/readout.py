"""Residual capture at the readout layer Lr, at the three SPEC positions.

Positions (SPEC "Internal readout instruments"):
    prompt_final:   last token of the full prompt BEFORE generation (the position
                      right before the first sampled token; the chat template's
                      generation prompt is part of the prompt).
    teacher_forced: the fixed continuation text run through the model under the
                      same context; mean residual over the continuation's tokens.
                      The continuation string is a config value shared across
                      conditions (runner.DEFAULT_FIXED_CONTINUATION unless
                      cells.json overrides it).
    response_mean:  mean residual over the actually-generated response tokens
                      (SECONDARY channel; known confound under text-changing
                      manipulations, labelled by the analysis rather than here).

SPEC-silent decisions:
  * prompt_final is captured from the teacher-forced pass: in a causal LM the
    hidden state at position t depends only on positions <= t, so the prefix
    positions of the longer sequence are identical to a prompt-only pass. This
    saves one forward per cell. When there is no continuation we forward the
    prompt alone.
  * Captures are detached to float32 CPU tensors so instruments (numpy probes,
    SAE encoders) run device-independently.

Circularity control (SPEC): ``project_out`` removes span(v_steer) from the
residual BEFORE the instrument is applied; the dose-response must survive it.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

Instrument = Callable[[torch.Tensor], float]


def capture_residuals(
    model: torch.nn.Module,
    layer_module: torch.nn.Module,
    prompt_ids: torch.Tensor,
    response_ids: Optional[torch.Tensor] = None,
    continuation_ids: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Return {position: residual vector (float32, CPU)} at the given layer.

    prompt_ids:        [1, P] full prompt incl. generation prompt (device tensors).
    response_ids:      1-D generated response token ids (trailing specials trimmed).
    continuation_ids:  1-D fixed-continuation token ids (encoded without specials).

    Must be called INSIDE the active steering context for steered cells, because the
    readout reads the steered state (the runner guarantees this ordering).
    """
    assert prompt_ids.dim() == 2 and prompt_ids.shape[0] == 1, "batch=1 everywhere"
    p_len = prompt_ids.shape[1]
    store: Dict[str, torch.Tensor] = {}

    def _cap(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        assert hs.shape[0] == 1, "batch=1 everywhere"
        store["hs"] = hs[0].detach().float().cpu()

    out: Dict[str, torch.Tensor] = {}
    handle = layer_module.register_forward_hook(_cap)
    try:
        with torch.no_grad():
            if continuation_ids is not None and continuation_ids.numel() > 0:
                seq = torch.cat(
                    [prompt_ids, continuation_ids.view(1, -1).to(prompt_ids.device)], dim=1
                )
                model(seq)
                out["prompt_final"] = store["hs"][p_len - 1]
                out["teacher_forced"] = store["hs"][p_len:].mean(dim=0)
            else:
                model(prompt_ids)
                out["prompt_final"] = store["hs"][p_len - 1]
            if response_ids is not None and response_ids.numel() > 0:
                seq = torch.cat(
                    [prompt_ids, response_ids.view(1, -1).to(prompt_ids.device)], dim=1
                )
                model(seq)
                out["response_mean"] = store["hs"][p_len:].mean(dim=0)
    finally:
        handle.remove()
    return out


def project_out(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """h - (h.v_hat) v_hat : remove span(v) from the residual (circularity control)."""
    v = torch.as_tensor(v, dtype=torch.float32).flatten()
    v = v / v.norm()
    h = h.float()
    return h - (h @ v) * v


def apply_instrument(instrument: Instrument, captures: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Apply one instrument callable to every captured position."""
    return {pos: float(instrument(h)) for pos, h in captures.items()}


def instrument_scores(
    captures: Dict[str, torch.Tensor],
    instruments: Dict[str, Instrument],
    steer_vector: Optional[torch.Tensor] = None,
    projout_instrument: str = "probe",
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Score every instrument at every position; plus the projection-out variant.

    Returns (scores, projout) where scores = {instrument_name: {position: float}}
    and projout = {position: float} for the projout_instrument with span(v_steer)
    removed first, and {} when there is no active steering vector (NULL cells), which
    matches schema.CellResult.probe_score_projout ("{} if NULL").
    """
    scores = {name: apply_instrument(fn, captures) for name, fn in instruments.items()}
    projout: Dict[str, float] = {}
    if steer_vector is not None and projout_instrument in instruments:
        fn = instruments[projout_instrument]
        projout = {pos: float(fn(project_out(h, steer_vector))) for pos, h in captures.items()}
    return scores, projout

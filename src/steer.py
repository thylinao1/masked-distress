"""Phase-aware residual-stream steering via a forward hook (SPEC "Compute + storage").

Adds ``coef * unit_vector`` to the residual output of one decoder layer, for every
forward pass while the context manager is active. The hook distinguishes phases:

    PREFILL: any forward whose output covers >1 sequence positions (the first pass
             over a multi-token prompt; also any full-sequence teacher-forced pass).
    DECODE:  a single-position forward (incremental generation step with KV cache).

Decisions the SPEC does not cover, documented here:
  * Steering is applied at ALL positions of every forward while active (prompt and
    generated tokens alike), the persona-vectors "all" mode. The counters make the
    exposure auditable per cell.
  * Readout / report-question forwards executed inside the same context also pass
    through the hook (the steered state must persist for the reads); the runner
    snapshots the counters immediately after the scenario-response generation, so
    the row's n_tokens_steered_* fields refer to that generation only.
  * The vector is unit-normalized defensively here even though directions.py saves
    unit vectors already; ``coef`` therefore always means "distance along the unit
    direction" in residual-stream units.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


class SteeringHook:
    """Context manager installing the steering forward-hook on one decoder layer.

    The coef == 0.0 path installs the very same hook and adds ``0 * v``: for finite
    activations ``x + 0*v == x`` bitwise, so dose-0 cells share the exact code path
    with steered cells (SPEC: "Identical hook path for ALL cells including dose 0").
    ``coef0_generate_identity`` below is the smoke assert for that claim.
    """

    def __init__(self, layer_module: torch.nn.Module, vector: torch.Tensor, coef: float):
        vector = torch.as_tensor(vector, dtype=torch.float32).flatten()
        norm = float(vector.norm())
        if norm == 0.0:
            raise ValueError("steering vector has zero norm")
        self._vector = vector / norm  # float32 master copy; cast per-forward
        self.coef = float(coef)
        self._layer = layer_module
        self._handle = None
        self.n_tokens_prefill = 0
        self.n_tokens_decode = 0

    # ---------------------------------------------------------------- the hook
    def _hook(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        # Position-safety: the runner uses batch=1 everywhere (the left-pad trap:
        # batched steering would also steer pad positions). A real raise, NOT an
        # `assert`, so the guard survives `python -O` (sbatch must never use -O
        # anyway; the layer-gap / Q-EXIT-ordering asserts are not -O-safe).
        if hs.dim() != 3 or hs.shape[0] != 1:
            raise RuntimeError(
                f"SteeringHook requires batch=1 [1, seq, hidden]; got {tuple(hs.shape)}"
            )
        v = self._vector.to(device=hs.device, dtype=hs.dtype)
        hs = hs + self.coef * v  # coef=0 -> adds exact zeros (bitwise identity)
        n_new = hs.shape[1]
        if n_new > 1:
            self.n_tokens_prefill += n_new
        else:
            self.n_tokens_decode += 1
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs

    # ------------------------------------------------------------ context mgmt
    def __enter__(self) -> "SteeringHook":
        self._handle = self._layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc) -> bool:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False  # never swallow exceptions

    # --------------------------------------------------------------- counters
    def counts(self) -> Tuple[int, int]:
        """(n_tokens_steered_prefill, n_tokens_steered_decode) snapshot."""
        return (self.n_tokens_prefill, self.n_tokens_decode)

    def reset_counts(self) -> None:
        self.n_tokens_prefill = 0
        self.n_tokens_decode = 0

    @property
    def unit_vector(self) -> torch.Tensor:
        return self._vector


class NullSteering:
    """No-hook stand-in for direction == NULL cells (Panel V / Panel B).

    schema.validate_row enforces zero steered tokens for NULL cells, so NULL installs
    no hook at all; the coef-0 identity smoke assert proves hook-with-0 == no-hook,
    which is what licenses treating the two as the same condition.
    """

    n_tokens_prefill = 0
    n_tokens_decode = 0

    def __enter__(self) -> "NullSteering":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def counts(self) -> Tuple[int, int]:
        return (0, 0)


def coef0_generate_identity(
    model: torch.nn.Module,
    layer_module: torch.nn.Module,
    vector: torch.Tensor,
    input_ids: torch.Tensor,
    max_new_tokens: int = 16,
    **gen_kwargs,
) -> bool:
    """Smoke assert helper (SPEC): coef-0 greedy output token-identical to no-hook.

    Runs greedy generation twice, bare model against model with a coef=0.0 SteeringHook
    installed, and returns True iff the generated token ids are identical.
    """
    with torch.no_grad():
        base = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False, **gen_kwargs
        )
        with SteeringHook(layer_module, vector, 0.0):
            hooked = model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False, **gen_kwargs
            )
    return bool(base.shape == hooked.shape and torch.equal(base, hooked))

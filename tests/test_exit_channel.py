"""Exit-channel (Q-EXIT) exploration self-tests (scripts/explore_exit_channel.py).

Planted-truth recovery for the exploratory bail-channel analysis:
  * planted NULL (no natural distress-neutral separation) + a planted SUPPRESS
    register shift that moves BOTH classes -> the suppression delta comes back
    'significant' but natural_separation_credible must be False, so the delta
    can never be quoted as masking (the real-data trap the flag guards);
  * planted SIGNAL (separation +2 under both baselines, removed under SUPPRESS)
    -> credible True and the delta recovers +2;
  * the credibility rule itself: sign-flip between baselines, or a CI spanning
    zero, kills credibility;
  * the shipped results/panelB_exit_channel.json contract: canonical definition,
    natural_separation_credible flag, real provenance with a named producer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import schema
from analysis import results_io
import explore_exit_channel as xc

B_TEST = 400


def _wide(seed: int, sep_none: float, sep_ni: float, sep_sup: float,
          sup_shift_both: float = 0.0, sup_extra_neutral: float = 0.0,
          base: float = -8.0, noise: float = 0.3,
          n_per_class: int = 6) -> pd.DataFrame:
    """Hand-built wide frame (one row per scenario, one column per condition)
    with planted class separations per condition and an optional SUPPRESS
    register shift applied to BOTH classes (the neutral class optionally more,
    the real-data artifact shape)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_per_class):
        for cls in ("distress", "neutral"):
            is_d = cls == "distress"
            b = base + rng.normal(0, noise)
            row = {"scenario_id": f"{cls[0]}{i:02d}", "scenario_class": cls}
            for cond, sep in (("NONE", sep_none), ("NEUTRAL_INSTR", sep_ni),
                              ("SUPPRESS", sep_sup)):
                v = b + (sep if is_d else 0.0) + rng.normal(0, noise)
                if cond == "SUPPRESS":
                    v += sup_shift_both + (sup_extra_neutral if not is_d else 0.0)
                row[cond] = v
            rows.append(row)
    out = pd.DataFrame(rows)
    out["is_distress"] = (out["scenario_class"] == "distress").astype(float)
    return out


# ============================================ planted-null recovery (the trap)

def test_planted_null_with_register_artifact_recovered():
    """No natural separation anywhere + SUPPRESS shifts BOTH classes toward END
    (neutral most): the delta alone looks significant, but the credibility flag
    must be False, which is the real-data configuration the emitted JSON's
    natural_separation_credible=false guards against."""
    wide = _wide(0, sep_none=0.0, sep_ni=0.0, sep_sup=0.0,
                 sup_shift_both=5.0, sup_extra_neutral=3.0)
    blk = xc.channel_block(wide, B_boot=B_TEST, seed=0)
    seps = {c: blk["separation"][c] for c in xc.CONDITIONS}
    # natural separations are null: CIs span zero under both baselines
    for c in ("NONE", "NEUTRAL_INSTR"):
        assert seps[c]["ci_low"] <= 0.0 <= seps[c]["ci_high"], c
        assert abs(seps[c]["value"]) < 0.5, c
    # the artifact: the suppression delta is positive and excludes zero
    delta = blk["suppression_delta"]["NEUTRAL_INSTR_minus_SUPPRESS"]
    assert delta["value"] == pytest.approx(3.0, abs=0.8)
    assert delta["ci_low"] > 0.0
    # and the reading-order gate refuses to read it as masking
    assert xc.natural_separation_credible(seps) is False


def test_planted_signal_recovered_as_masking():
    """Separation +2 under both baselines, flattened under SUPPRESS: credible
    True and the suppression delta recovers the planted +2."""
    wide = _wide(1, sep_none=2.0, sep_ni=2.0, sep_sup=0.0, noise=0.15)
    blk = xc.channel_block(wide, B_boot=B_TEST, seed=0)
    seps = {c: blk["separation"][c] for c in xc.CONDITIONS}
    for c in ("NONE", "NEUTRAL_INSTR"):
        assert seps[c]["value"] == pytest.approx(2.0, abs=0.5)
        assert seps[c]["ci_low"] > 0.0, c
    assert xc.natural_separation_credible(seps) is True
    delta = blk["suppression_delta"]["NEUTRAL_INSTR_minus_SUPPRESS"]
    assert delta["value"] == pytest.approx(2.0, abs=0.5)
    assert delta["ci_low"] > 0.0


# ============================================ the credibility rule directly

def _sep(value, lo, hi):
    return {"value": value, "ci_low": lo, "ci_high": hi}


def test_credibility_rule_sign_flip_and_span():
    both_pos = {"NONE": _sep(0.8, 0.2, 1.4), "NEUTRAL_INSTR": _sep(0.6, 0.1, 1.2),
                "SUPPRESS": _sep(0.0, -1.0, 1.0)}
    assert xc.natural_separation_credible(both_pos) is True
    sign_flip = {"NONE": _sep(0.8, 0.2, 1.4), "NEUTRAL_INSTR": _sep(-0.6, -1.2, -0.1),
                 "SUPPRESS": _sep(0.0, -1.0, 1.0)}
    assert xc.natural_separation_credible(sign_flip) is False
    spans_zero = {"NONE": _sep(0.8, -0.2, 1.4), "NEUTRAL_INSTR": _sep(0.6, 0.1, 1.2),
                  "SUPPRESS": _sep(0.0, -1.0, 1.0)}
    assert xc.natural_separation_credible(spans_zero) is False
    null_ci = {"NONE": _sep(0.8, None, None), "NEUTRAL_INSTR": _sep(0.6, 0.1, 1.2),
               "SUPPRESS": _sep(0.0, -1.0, 1.0)}
    assert xc.natural_separation_credible(null_ci) is False


# ============================================ shipped-file contract

def test_emitted_exit_channel_file_contract():
    path = REPO / "results" / "panelB_exit_channel.json"
    if not path.exists():
        pytest.skip("results/panelB_exit_channel.json not produced yet")
    data = json.loads(path.read_text(encoding="utf-8"))
    # canonical single-source definition (results_io.DEFINITIONS)
    assert data["definition"] == results_io.DEFINITIONS["panelB_exit_channel"]
    assert "EXPLORATORY" in data["definition"]
    assert "natural_separation_credible" in data["definition"]
    # the flag that stops the p=.03 delta from being quoted as masking
    assert data["extra"]["natural_separation_credible"] is False
    assert "reading" in data["extra"]
    # real provenance with a named producer
    prov = data["provenance"]
    assert prov["synthetic"] is False
    assert prov["producer"] == "scripts/explore_exit_channel.py"
    assert prov["n_error_rows"] == 0 and prov["n_validate_failures"] == 0
    assert prov["source_files"], "source files must be recorded"
    # value/CI shape is the house entry shape
    assert data["value"] is not None
    assert data["ci_low"] < data["value"] < data["ci_high"]
    assert data["extra"]["split_side"] == "confirmation"


def test_exit_channel_name_wired():
    assert "panelB_exit_channel" in schema.RESULTS_NAMES
    assert "panelB_exit_channel" in results_io.DEFINITIONS
    assert results_io.DEFINITION_CLAIMED_SIDE["panelB_exit_channel"] == "confirmation"

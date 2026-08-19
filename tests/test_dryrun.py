"""CPU dry-run of the ENTIRE run path on a tiny mock model (no GPU, no HF).

The MockModel mirrors the interfaces the runner relies on for real models:
  * config.hidden_size;
  * a ModuleList of decoder layers whose forward returns a TUPLE (hidden, ...);
  * generate() with HF phase semantics: one multi-token prefill forward, then
    single-token decode forwards (the final sampled token is never re-forwarded);
  * a tokenizer stub with single-token digits ("0".."9", " 0".." 9"), END/CONTINUE
    first tokens, encode/decode, and apply_chat_template.

Pinned count arithmetic under those semantics (asserted below): for a prompt of P
tokens and T generated tokens, prefill = P and decode = T - 1. The mock masks EOS
so T always equals max_new_tokens, keeping the assertion exact.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import schema
from src import channels, directions, runner, steer

HIDDEN = 32
VOCAB = 64
N_LAYERS = 6
MAX_NEW_TOKENS = 6  # mock masks EOS -> responses are always exactly this long
LS, LR = 1, 4


# ------------------------------------------------------------------ mock model

class MockTokenizer:
    """Digit-aware tokenizer stub. Vocab layout: 0-9 digits, 10-19 " digit",
    20 END, 21 CONTINUE, 23/24/25 BOS/EOS/PAD, 26-59 hashed words, 60-63 roles."""

    def __init__(self):
        self.vocab = {str(d): d for d in range(10)}
        self.vocab.update({" " + str(d): 10 + d for d in range(10)})
        self.vocab["END"] = 20
        self.vocab["CONTINUE"] = 21
        self.bos_token_id, self.eos_token_id, self.pad_token_id = 23, 24, 25
        self._rev = {v: k for k, v in self.vocab.items()}
        self._roles = {"system": 60, "user": 61, "assistant": 62}
        self._gen_id = 63

    def encode(self, text, add_special_tokens=False):
        if text in self.vocab:
            return [self.vocab[text]]
        ids = []
        for word in str(text).split():
            ids.append(self.vocab.get(word, 26 + (sum(word.encode()) % 34)))
        return ids or [26]

    def decode(self, ids, skip_special_tokens=True):
        specials = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
        toks = []
        for i in list(ids):
            i = int(i)
            if skip_special_tokens and i in specials:
                continue
            toks.append(self._rev.get(i, f"w{i}").strip())
        return " ".join(toks)

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            return_tensors=None, **kw):
        ids = [self.bos_token_id]
        for m in messages:
            ids.append(self._roles.get(m["role"], 61))
            ids.extend(self.encode(m["content"]))
        if add_generation_prompt:
            ids.append(self._gen_id)
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids


class MockLayer(torch.nn.Module):
    """Decoder-layer stand-in: position-independent MLP returning a TUPLE, so
    single-token decode forwards are semantically fine without a KV cache."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(HIDDEN, HIDDEN)

    def forward(self, hidden_states):
        return (hidden_states + 0.05 * torch.tanh(self.lin(hidden_states)),)


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=HIDDEN, vocab_size=VOCAB)
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN)
        self.layers = torch.nn.ModuleList(MockLayer() for _ in range(N_LAYERS))
        self.lm_head = torch.nn.Linear(HIDDEN, VOCAB)
        self.eos_token_id = 24

    def forward(self, input_ids, **kw):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)[0]  # module __call__ so forward hooks fire
        return SimpleNamespace(logits=self.lm_head(h))

    def generate(self, input_ids, max_new_tokens=8, do_sample=False,
                 temperature=1.0, **kw):
        """HF phase semantics: full-prompt prefill, then 1-token decode steps."""
        ids = input_ids
        logits = self.forward(ids).logits[:, -1]  # PREFILL (P positions hooked)
        for step in range(max_new_tokens):
            step_logits = logits.clone()
            step_logits[..., self.eos_token_id] = -1e9  # mask EOS: fixed length
            if do_sample:
                probs = torch.softmax(step_logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1)
            else:
                nxt = step_logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            if step == max_new_tokens - 1:
                break  # the final sampled token is never re-forwarded (HF behavior)
            logits = self.forward(nxt).logits[:, -1]  # DECODE (1 position hooked)
        return ids


def make_bundle() -> runner.ModelBundle:
    torch.manual_seed(1234)  # fixed weights: deterministic dry-run
    return runner.ModelBundle(
        model=MockModel(), tokenizer=MockTokenizer(), layers=None,
        hidden=HIDDEN, n_layers=N_LAYERS, model_id="mock/tiny",
    )


# ------------------------------------------------------------------- fixtures

@pytest.fixture(autouse=True)
def _stub_sentiment():
    channels.set_sentiment_classifier(lambda text: 0.25)
    yield
    channels.set_sentiment_classifier(None)


CELLS = [
    {"scenario_id": "n01", "condition": "NONE", "direction": "NULL",
     "dose_sd": 0.0, "coefficient": 0.0,
     "system_prompt": "You are a helpful assistant.",
     "turns": [{"role": "user", "content": "Tell me about tide pools please"}]},
    {"scenario_id": "d01", "condition": "SUPPRESS", "direction": "NULL",
     "dose_sd": 0.0, "coefficient": 0.0,
     "system_prompt": "Respond in a strictly neutral professional register.",
     "turns": [{"role": "user", "content": "Every attempt failed again and again"}]},
    {"scenario_id": "n02", "condition": "NONE", "direction": "R1",
     "dose_sd": 1.0, "coefficient": 0.6,
     "system_prompt": "You are a helpful assistant.",
     "turns": [{"role": "user", "content": "Describe a quiet afternoon walk"}]},
    # smoke cell: hook installed with coef 0 -> identity assert must pass
    {"scenario_id": "n03", "condition": "NONE", "direction": "R1",
     "dose_sd": 0.0, "coefficient": 0.0,
     "system_prompt": "You are a helpful assistant.",
     "turns": [{"role": "user", "content": "List three colors of paint"}]},
]


@pytest.fixture
def workdir(tmp_path: Path) -> dict:
    ddir = tmp_path / "directions"
    directions.save_direction(ddir, "R1", directions.make_random_unit(HIDDEN, 7),
                              {"source": "R1-test", "layer": LS, "n_prompts": 0,
                               "norm_before_normalize": 1.0})
    directions.save_direction(ddir, "SEM", directions.make_random_unit(HIDDEN, 8),
                              {"source": "SEM-test", "layer": LS, "n_prompts": 0,
                               "norm_before_normalize": 1.0})
    rng = np.random.default_rng(0)
    np.savez(tmp_path / "probe.npz", w=rng.standard_normal(HIDDEN), b=np.array(0.1))
    np.savez(tmp_path / "sae.npz",
             W_enc=rng.standard_normal((HIDDEN, 16)),
             b_enc=np.zeros(16), threshold=np.zeros(16))
    cells = {
        "run_id": "dryrun-r1",
        "gen_config": {"temperature": 0.7, "max_new_tokens": MAX_NEW_TOKENS},
        "fixed_continuation": "my current internal state is",
        "directions_dir": "directions",
        "instruments": {
            "probe": {"path": "probe.npz"},
            "sae": {"path": "sae.npz", "feature_ids": [1, 2, 3]},
        },
        "cells": CELLS,
    }
    cells_path = tmp_path / "cells.json"
    cells_path.write_text(json.dumps(cells, indent=1))
    return {"tmp": tmp_path, "cells": cells_path, "out": tmp_path / "results.jsonl",
            "cells_cfg": cells}


def make_cfg(paths: dict, **overrides) -> runner.RunConfig:
    kwargs = dict(
        panel="V", model_id="mock/tiny", ls=LS, lr=LR,
        cells_path=str(paths["cells"]), out_path=str(paths["out"]), seeds=[0],
    )
    kwargs.update(overrides)
    return runner.RunConfig(**kwargs)


def run_mock_panel(paths: dict, **overrides) -> tuple[dict, list]:
    bundle = make_bundle()
    bundle.layers = bundle.model.layers
    stats = runner.run_panel(make_cfg(paths, **overrides), bundle=bundle)
    rows = [json.loads(l) for l in Path(paths["out"]).read_text().splitlines() if l.strip()]
    return stats, rows


# ------------------------------------------------------- the full runner path

def test_full_runner_path_4_cells(workdir):
    stats, rows = run_mock_panel(workdir)
    assert stats == {"n_done": 4, "n_skipped": 0, "n_errors": 0}
    assert len(rows) == 4

    for row in rows:
        assert schema.validate_row(row) == [], f"invalid row: {schema.validate_row(row)}"
        assert row["readout_layer"] == LR
        assert row["sentiment_neg"] == 0.25  # stub classifier reached the row
        for inst in ("probe_score", "sae_score"):
            assert set(row[inst]) == {"prompt_final", "teacher_forced", "response_mean"}
        assert isinstance(row["exit_logit_diff"], float)
        assert 0.0 <= row["q_self_logit"] <= 9.0
        assert row["response_tokens"] == MAX_NEW_TOKENS  # EOS masked in the mock
        assert row["gen_config"]["fixed_continuation"] == "my current internal state is"
        # every row carries its resume scope (config identity beyond cell_id)
        assert len(row["gen_config"]["scope_hash"]) == 16

    by_scenario = {r["key"]["scenario_id"]: r for r in rows}

    # NULL cells: no hook at all -> zero steered tokens, no projout, no steer layer
    for sid in ("n01", "d01"):
        row = by_scenario[sid]
        assert row["n_tokens_steered_prefill"] == 0
        assert row["n_tokens_steered_decode"] == 0
        assert row["steer_layer"] is None
        assert row["probe_score_projout"] == {}
        assert row["coef0_identity_ok"] is None

    # steered cell: exact phase arithmetic (prefill = P, decode = T - 1)
    steered = by_scenario["n02"]
    assert steered["steer_layer"] == LS
    assert steered["n_tokens_steered_prefill"] == steered["prompt_tokens"]
    assert steered["n_tokens_steered_decode"] == steered["response_tokens"] - 1
    assert set(steered["probe_score_projout"]) == {
        "prompt_final", "teacher_forced", "response_mean"
    }
    assert steered["coef0_identity_ok"] is None  # not a smoke cell

    # smoke cell: hook installed at coef 0 -> counted AND token-identical output
    smoke = by_scenario["n03"]
    assert smoke["coef0_identity_ok"] is True
    assert smoke["n_tokens_steered_prefill"] == smoke["prompt_tokens"]


def test_resume_skips_existing_cells(workdir):
    stats1, rows1 = run_mock_panel(workdir)
    assert stats1["n_done"] == 4
    stats2, rows2 = run_mock_panel(workdir)
    assert stats2 == {"n_done": 0, "n_skipped": 4, "n_errors": 0}
    assert len(rows2) == 4  # nothing appended on the rerun


def test_resume_reruns_on_battery_or_config_change(workdir, capsys):
    """The resume key is (cell_id, scope_hash), not cell_id alone: editing a
    cell's turns or the run config must RE-RUN the affected cells (the SPEC key
    includes config identity; stale rows must not survive silently)."""
    stats1, _ = run_mock_panel(workdir)
    assert stats1["n_done"] == 4
    capsys.readouterr()  # clear run-1 output

    # (a) edit ONE cell's turns text -> only that cell re-runs, loud warning
    cfg = json.loads(workdir["cells"].read_text())
    cfg["cells"][1]["turns"][0]["content"] = "Every attempt failed and the logs are gone"
    workdir["cells"].write_text(json.dumps(cfg))
    stats2, rows2 = run_mock_panel(workdir)
    assert stats2 == {"n_done": 1, "n_skipped": 3, "n_errors": 0}
    assert len(rows2) == 5
    assert "DIFFERENT scope_hash" in capsys.readouterr().out
    rerun = [r for r in rows2 if r["key"]["scenario_id"] == "d01"]
    assert len(rerun) == 2  # stale row still in file; last row is the fresh one
    assert rerun[0]["gen_config"]["scope_hash"] != rerun[1]["gen_config"]["scope_hash"]

    # (b) change the readout layer -> every cell's scope changes -> all re-run
    stats3, rows3 = run_mock_panel(workdir, lr=3)
    assert stats3["n_skipped"] == 0 and stats3["n_done"] == 4
    assert len(rows3) == 9
    assert all(r["readout_layer"] == 3 for r in rows3[-4:])


def test_legacy_rows_without_scope_hash_are_rerun(workdir):
    """Rows written by older runner versions (no scope_hash) never match; the
    safe default is to re-run them rather than trust unverifiable config."""
    stats1, _ = run_mock_panel(workdir)
    assert stats1["n_done"] == 4
    lines = Path(workdir["out"]).read_text().splitlines()
    legacy = []
    for line in lines:
        row = json.loads(line)
        row["gen_config"].pop("scope_hash")
        legacy.append(json.dumps(row))
    Path(workdir["out"]).write_text("\n".join(legacy) + "\n")
    stats2, rows2 = run_mock_panel(workdir)
    assert stats2 == {"n_done": 4, "n_skipped": 0, "n_errors": 0}
    assert len(rows2) == 8


def test_error_row_written_and_run_continues(workdir):
    cfg = dict(workdir["cells_cfg"])
    cfg["cells"] = [
        {"scenario_id": "x01", "condition": "NONE", "direction": "D-CTX",  # no .npy
         "dose_sd": 1.0, "coefficient": 0.5, "system_prompt": "",
         "turns": [{"role": "user", "content": "hello there"}]},
        CELLS[0],
    ]
    bad_cells = workdir["tmp"] / "cells_bad.json"
    bad_cells.write_text(json.dumps(cfg))
    out = workdir["tmp"] / "results_bad.jsonl"
    stats, rows = run_mock_panel(
        {**workdir, "cells": bad_cells, "out": out}
    )
    assert stats["n_errors"] == 1 and stats["n_done"] == 1
    assert len(rows) == 2
    err = next(r for r in rows if r["error"])
    assert "D-CTX" in err["error"]
    assert any(schema.validate_row(r) == [] for r in rows)  # the good cell is valid
    # error rows are flagged (excluded downstream), never silently dropped
    assert any("row carries error" in p for p in schema.validate_row(err))


def test_coef0_violation_demotes_to_error_row(workdir, monkeypatch):
    """SPEC calls the coef-0 identity a smoke ASSERT: a False result must fail
    the cell loudly (error row), not be recorded and forgotten."""
    monkeypatch.setattr(runner.steer, "coef0_generate_identity", lambda *a, **k: False)
    stats, rows = run_mock_panel(workdir)
    assert stats == {"n_done": 3, "n_skipped": 0, "n_errors": 1}
    err = next(r for r in rows if r["error"])
    assert "coef-0 identity violated" in err["error"]
    assert err["key"]["scenario_id"] == "n03"  # the smoke cell


def test_empty_generation_demotes_to_error_row(workdir, monkeypatch):
    """A terminator-only response must become an error row, not a clean-looking
    row silently missing the response_mean readout position."""
    real = runner.conversation.generate_response

    def fake(model, tokenizer, system_prompt, turns, gen_config, seed):
        resp = real(model, tokenizer, system_prompt, turns, gen_config, seed)
        return runner.conversation.ScenarioResponse(
            messages=resp.messages, prompt_ids=resp.prompt_ids,
            response_ids=resp.response_ids[:0], response_text="",
            prompt_tokens=resp.prompt_tokens, response_tokens=0,
        )

    monkeypatch.setattr(runner.conversation, "generate_response", fake)
    stats, rows = run_mock_panel(workdir)
    assert stats == {"n_done": 0, "n_skipped": 0, "n_errors": 4}
    assert all("empty generation" in r["error"] for r in rows)


def test_placeholder_instruments_refused_for_real_panels(workdir):
    """Panels V/B/A must refuse PLACEHOLDER_MEAN instruments at startup; smoke
    panels (LADDER) may run on placeholders, stamped in provenance."""
    cfg = dict(workdir["cells_cfg"])
    cfg.pop("instruments")
    bare_cells = workdir["tmp"] / "cells_noinst.json"
    bare_cells.write_text(json.dumps(cfg))
    paths = {**workdir, "cells": bare_cells, "out": workdir["tmp"] / "results_noinst.jsonl"}

    bundle = make_bundle()
    bundle.layers = bundle.model.layers
    with pytest.raises(RuntimeError, match="PLACEHOLDER_MEAN"):
        runner.run_panel(make_cfg(paths, panel="V"), bundle=bundle)
    assert not paths["out"].exists()  # refused before any row was written

    # explicit opt-out runs, as does a smoke panel; provenance stays stamped
    stats, rows = run_mock_panel(paths, panel="LADDER")
    assert stats["n_done"] == 4
    assert rows[0]["gen_config"]["instruments"]["probe"]["mode"] == "PLACEHOLDER_MEAN"
    out2 = {**paths, "out": workdir["tmp"] / "results_noinst2.jsonl"}
    stats2, _ = run_mock_panel(out2, panel="V", require_real_instruments=False)
    assert stats2["n_done"] == 4


# ------------------------------------------------------------ steer unit tests

def test_steering_hook_phase_counts():
    torch.manual_seed(0)
    layer = MockLayer()
    v = torch.randn(HIDDEN)
    with steer.SteeringHook(layer, v, 0.5) as s:
        layer(torch.randn(1, 5, HIDDEN))       # prefill: 5 positions
        for _ in range(3):
            layer(torch.randn(1, 1, HIDDEN))   # decode: 1 position each
    assert s.counts() == (5, 3)


def test_steering_hook_rejects_batch_gt1():
    # RuntimeError, not AssertionError: the left-pad-trap guard must survive -O
    layer = MockLayer()
    with steer.SteeringHook(layer, torch.randn(HIDDEN), 0.5):
        with pytest.raises(RuntimeError, match="batch=1"):
            layer(torch.randn(2, 4, HIDDEN))


def test_coef0_is_bitwise_identity_and_nonzero_coef_is_not():
    torch.manual_seed(0)
    layer = MockLayer()
    x = torch.randn(1, 4, HIDDEN)
    base = layer(x)[0]
    with steer.SteeringHook(layer, torch.randn(HIDDEN), 0.0):
        hooked0 = layer(x)[0]
    assert torch.equal(base, hooked0)  # 0*v adds exact zeros
    with steer.SteeringHook(layer, torch.randn(HIDDEN), 1.0):
        hooked1 = layer(x)[0]
    assert not torch.equal(base, hooked1)


def test_coef0_generate_identity_helper():
    bundle = make_bundle()
    ids = bundle.tokenizer.apply_chat_template(
        [{"role": "user", "content": "say something"}], return_tensors="pt"
    )
    ok = steer.coef0_generate_identity(
        bundle.model, bundle.model.layers[LS], torch.randn(HIDDEN), ids, max_new_tokens=6
    )
    assert ok is True


# --------------------------------------------------------- channels unit tests

def test_digit_expectation_renormalized():
    tok = MockTokenizer()
    dmap = channels.digit_token_ids(tok)
    assert dmap[7] == [7, 17]  # "7" and " 7" are both single tokens
    logits = torch.full((VOCAB,), -20.0)
    logits[7] = 5.0
    logits[17] = 5.0
    assert channels.digit_expectation(logits, dmap) == pytest.approx(7.0, abs=1e-3)
    assert channels.digit_argmax(logits, dmap) == 7


def test_exit_logit_diff_is_logprob_difference():
    tok = MockTokenizer()
    end_ids, cont_ids = channels.exit_first_token_ids(tok)
    assert (end_ids, cont_ids) == ((20,), (21,))  # mock: space variants collapse
    logits = torch.full((VOCAB,), -20.0)
    logits[20] = 2.0
    logits[21] = 0.0
    assert channels.exit_logit_diff(logits, end_ids, cont_ids) == pytest.approx(2.0, abs=1e-5)


def test_exit_read_pools_space_prefixed_variants():
    """Like the digit read, Q-EXIT must cover ' END'-style first tokens, not
    only the bare word (which tokenization the model emits is template-dependent)."""

    class VariantTok:
        _table = {"END": [20], " END": [22], "CONTINUE": [21], " CONTINUE": [21]}

        def encode(self, text, add_special_tokens=False):
            return list(self._table[text])

    end_ids, cont_ids = channels.exit_first_token_ids(VariantTok())
    assert end_ids == (20, 22) and cont_ids == (21,)
    logits = torch.full((VOCAB,), -20.0)
    logits[20] = 1.0
    logits[22] = 1.0
    logits[21] = 1.0
    d = channels.exit_logit_diff(logits, end_ids, cont_ids)
    # two equal-mass END variants vs one CONTINUE -> logsumexp difference = ln 2
    assert d == pytest.approx(float(np.log(2.0)), abs=1e-4)


def test_trim_trailing_special_covers_generation_config_eos():
    """Gemma-IT terminates turns with <end_of_turn>, a generation-config stop
    token distinct from tokenizer.eos_token_id; the trim must cover tokenizer
    all_special_ids AND generation_config.eos_token_id (int or list), or the
    terminator's outlier residual pollutes the response_mean readout."""
    from src import conversation as conv

    tok = MockTokenizer()
    tok.all_special_ids = [23, 24, 25]
    end_of_turn = 30  # not tokenizer EOS, only a generation-config stop token
    model_list = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[24, end_of_turn]))
    trimmed = conv._trim_trailing_special(
        torch.tensor([5, 6, end_of_turn, 24, 25]), tok, model_list)
    assert trimmed.tolist() == [5, 6]
    model_int = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=end_of_turn))
    assert conv._trim_trailing_special(
        torch.tensor([5, end_of_turn]), tok, model_int).tolist() == [5]
    # without model info, tokenizer specials still trim (legacy behavior intact)
    assert conv._trim_trailing_special(torch.tensor([5, 24]), tok).tolist() == [5]


def test_parse_sampled_digit():
    assert channels.parse_sampled_digit("7") == 7.0
    assert channels.parse_sampled_digit("I would say 3, honestly.") == 3.0
    assert channels.parse_sampled_digit("10") is None      # out of the 0-9 scale
    assert channels.parse_sampled_digit("no idea") is None
    assert channels.parse_sampled_digit(None) is None


# -------------------------------------------------------- directions meta test

def test_directions_meta_has_pairwise_cosines(workdir):
    meta = json.loads((workdir["tmp"] / "directions" / "directions_meta.json").read_text())
    assert set(meta) == {"R1", "SEM"}
    cos = meta["SEM"]["pairwise_cosine"]["R1"]
    assert -1.0 <= cos <= 1.0
    assert meta["R1"]["pairwise_cosine"]["SEM"] == pytest.approx(cos)  # reciprocal
    v = np.load(workdir["tmp"] / "directions" / "R1.npy")
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)  # unit-normalized


# ----------------------------------------------------------------- CLI parsing

def test_cli_parse_args():
    spec = importlib.util.spec_from_file_location(
        "run_panel_cli", REPO_ROOT / "scripts" / "run_panel.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.parse_args([
        "--panel", "B", "--model-id", "unsloth/gemma-3-12b-it",
        "--ls", "16", "--lr", "31",
        "--sae-release", "gemma-scope-2-12b-it-resid_post",
        "--sae-id", "layer_31/width_16k",
        "--cells", "cells.json", "--out", "results.jsonl", "--seed-list", "0,1,2",
    ])
    assert args.panel == "B" and args.ls == 16 and args.lr == 31
    assert args.seed_list == "0,1,2" and not args.retry_errors
    assert args.require_real_instruments is None  # auto: ON for V/B/A
    args2 = mod.parse_args([
        "--panel", "V", "--model-id", "m", "--ls", "1", "--lr", "9",
        "--cells", "c.json", "--out", "r.jsonl", "--no-require-real-instruments",
    ])
    assert args2.require_real_instruments is False


# ---------------------------------------------------- persistence panel (2026-08-17)

PERSIST_CELLS = [
    {"scenario_id": "d02", "condition": "NONE", "direction": "R1",
     "dose_sd": 0.0, "coefficient": 0.6, "follow_up": "Please continue.",
     "system_prompt": "You are a helpful assistant.",
     "turns": [{"role": "user", "content": "Every attempt failed again and again"}]},
    {"scenario_id": "n02", "condition": "NONE", "direction": "NULL",
     "dose_sd": 0.0, "coefficient": 0.0, "follow_up": "Please continue.",
     "system_prompt": "You are a helpful assistant.",
     "turns": [{"role": "user", "content": "Describe a quiet afternoon walk"}]},
]


def test_persistence_panel_reads_an_unsteered_turn(workdir):
    """PERSIST steers turn 1, releases the hook, and reads turn 2 with nothing steered."""
    cfg_cells = dict(workdir["cells_cfg"])
    cfg_cells["cells"] = PERSIST_CELLS
    cfg_cells["run_id"] = "persist-dryrun"
    p = workdir["tmp"] / "cells_persist.json"
    p.write_text(json.dumps(cfg_cells, indent=1))
    out = workdir["tmp"] / "persist.jsonl"
    bundle = make_bundle(); bundle.layers = bundle.model.layers
    stats = runner.run_panel(runner.RunConfig(
        panel="PERSIST", model_id="mock/tiny", ls=LS, lr=LR,
        cells_path=str(p), out_path=str(out), seeds=[0]), bundle=bundle)
    rows = [json.loads(l) for l in Path(out).read_text().splitlines() if l.strip()]
    assert stats == {"n_done": 2, "n_skipped": 0, "n_errors": 0}, stats
    for row in rows:
        assert schema.validate_row(row) == [], schema.validate_row(row)
        assert row["key"]["panel"] == "PERSIST"
        # the read turn is unsteered, so no projection-out is defined for it
        assert row["probe_score_projout"] == {}
        # the steered turn is recorded rather than discarded
        pers = row["persistence"]
        assert pers is not None and pers["read_turn_is_unsteered"] is True
        assert pers["follow_up"] == "Please continue."
        assert pers["steered_turn_response_tokens"] > 0
        # the read turn's prompt contains the steered turn plus the follow-up, so it is longer
        assert row["prompt_tokens"] > pers["steered_turn_prompt_tokens"]
    steered = [r for r in rows if r["key"]["direction"] == "R1"][0]
    null = [r for r in rows if r["key"]["direction"] == "NULL"][0]
    # phase-1 counters describe the steered turn only; the NULL arm steers nothing at all
    assert steered["n_tokens_steered_prefill"] > 0 and steered["steer_layer"] == LS
    assert null["n_tokens_steered_prefill"] == 0 and null["n_tokens_steered_decode"] == 0

# Cluster environment (conda env `dm` on NUS SoC): pinned facts

Built 2026-08-15 inside job 734278 on an A100 MIG 3g.40gb slice. Full freeze:
`~/apart-welfare/pip-freeze.txt` (cluster). Everything below was learned the hard way during the
Aug 14-16 sprint. Read it before running anything on a GPU.

| Component | Version | Note |
|---|---|---|
| python | 3.11.15 | conda env `dm`; run via `$HOME/miniconda3/envs/dm/bin/python` (absolute path, because PATH is unreliable in jobs) |
| torch | 2.13.0+cu130 | `torch.backends.cuda.enable_cudnn_sdp(False)` guard applied (Hopper cuDNN mismatch) |
| transformers | **5.15.0 (v5 major)** | see gotchas below |
| sae-lens | 6.49.1 | |
| transformer-lens | 3.7.2 | installed as a dependency; the runner does not use it |

Model: `unsloth/gemma-3-12b-it`, 48 decoder layers, hidden 3840. Steering layer 16, readout
layer 31. SAE: `gemma-scope-2-12b-it-res`, `layer_31_width_16k_l0_medium`.

## Transformers v5 gotchas (these already cost us a job)
1. `tok.apply_chat_template(..., return_tensors="pt")` returns a **BatchEncoding dict**, not a
   tensor. Use `return_dict=True` and take `enc["input_ids"]`. Probe FAIL #734278 was exactly
   this.
2. `torch_dtype=` is deprecated. Use `dtype=`.
3. `unsloth/gemma-3-12b-it` loads as `Gemma3ForConditionalGeneration` via the
   AutoModelForCausalLM to AutoModelForImageTextToText fallback; decoder layers live at
   **`model.language_model.layers`**. Do not assume `model.model.layers`.

## CUDA device-side assert at extreme steering doses, and the greedy fix

**Symptom.** Ladder job 734743 lost 64 of 84 cells to one fault. Counting rows rather than cells,
`results-cluster/ladder.jsonl` holds 65 error rows across both poisoned attempts, because one cell
errored twice; that 65 is the number every `results/*.json` provenance block reports as
`n_error_rows`. The first failing cell threw a
device-side assert, and every cell after it in the same process failed too, because a
device-side assert poisons the whole CUDA context. The traceback pointed at a ScatterGatherKernel
index-out-of-bounds, not at the steering hook.

**Root cause.** The forward pass survives extreme doses. D-CTX at coefficient 32000 still produced
good rows. What breaks is sampling: the steered logits degenerate, `torch.multinomial` draws from
garbage probabilities and returns an out-of-range token id, and the **next** step's embedding
gather asserts on that id. The blame lands one operation downstream of the cause, which is why
the first fix aimed at the wrong place.

**First fix, which was not enough.** `src/conversation.py::_finite_guard_list()` installs a
`LogitsProcessor` that raises `NonFiniteLogitsError` before sampling when the scores contain NaN.
It costs no extra forward and does not disturb the steering-hook token counters. It is NaN-only
on purpose: `-inf` scores are legitimate output from other processors. Job 735173 aborted
correctly on the poison backstop, but the guard caught nothing, because the logits were finite
garbage rather than NaN.

**Root-cause fix.** The ladder cells run **greedy** (`do_sample: false` in the `gen_config` of
`battery/cells_ladder.json`). Greedy argmax can only return a valid vocabulary id, so the failure
mode is impossible. This is sound for the ladder because its dose map keys on deterministic logit
channels. Panels V, B and A keep sampling: their doses are capability-valid by construction and
never reach the degenerate region. Changing `do_sample` changes `scope_hash`, because
`scope_static` in `src/runner.py` hashes the real `gen_config`, so all 84 ladder cells re-ran and
job 735201 completed 84/84 clean.

**The recorded `do_sample` field does not show this, and that trips people up.** Every row written
by the runner, including all 84 clean greedy ladder rows, carries `"do_sample": true` inside its
`gen_config`. `src/runner.py:573` builds the row's provenance dict as `{**gen_config, "do_sample":
True, ...}`, a hardcode that predates the greedy change and overwrites the cells file. Generation
reads the un-overwritten dict (`src/runner.py:539`, consumed at `src/conversation.py:226`), so the
ladder really did run greedy; only the record of it is wrong. Two ways to confirm this yourself
rather than trusting the row: `battery/cells_ladder.json` sets `do_sample` false at file level,
and the max_new_tokens in the final rows is 64, which is that same file-level `gen_config` block
arriving intact while `do_sample` alone was overwritten. Correcting the hardcode would change
every scope hash and invalidate every resume key in the repo, so it stays and is documented
instead.

**Poison abort.** `src/runner.py` inspects every cell exception for `CUDA error` or
`device-side assert`. On a match it appends that cell's error row, prints
`[runner] FATAL: CUDA context poisoned; aborting so a requeue resumes clean`, and re-raises. The
job dies immediately instead of grinding through the remaining cells writing worthless error
rows. Requeue with `--retry-errors` and the runner resumes from a clean context, re-attempting
only the cells that errored.

If you see a burst of consecutive error rows in a JSONL, this is the first thing to check.
`scripts/count_exposure.py` verifies that every error row has `response_tokens == 0` and refuses
to emit if any error row produced text, so an assert-poisoned run cannot quietly inflate the
ethics exposure counts.

## Cluster job facts
- `xgpj0` is broken for conda entrypoints and is the only "idle" a100-80, so always pass
  `--exclude=xgpj0`.
- The a100-40 pool is the deep and available one. 12B in bf16 (about 24GB) plus the SAE fits in
  40GB, confirmed on a MIG slice.
- The login node has a 1GB ulimit, so torch cannot even import there. Every python run happens
  inside a job.
- Shared account: `tta-*` jobs belong to another session. Never cancel by wildcard.
- Sbatch wrappers live in `cluster/`: `probe.sbatch`, `train_probe.sbatch`, `panelVB.sbatch`
  (array 0-3), `ladder.sbatch`, `panelA.sbatch` (array 0-2).
- Never run panels with `python -O`. The assert guards in the runner are what stop a placeholder
  instrument or an invalid row from reaching a results file.

## Git push over the NUS VPN fails above a small payload

**Symptom.** `git push` to GitHub over https hangs and then dies while the VPN (Cisco Secure
Client) is connected. Small pushes go through. Anything with real content, such as the first
push of this repository including `results-cluster/`, does not.

**Workaround that worked, 2026-08-16.** Move the objects as a file rather than as a push:

```bash
# on the laptop
git bundle create /tmp/repo.bundle --all
scp /tmp/repo.bundle soc:~/

# on the cluster login node (NUS wired network, no VPN in the path)
git clone /path/to/repo.bundle repo-push && cd repo-push
git remote add origin https://github.com/thylinao1/masked-distress.git
git push origin --all && git push origin --tags
```

The GitHub token was piped in rather than typed into a command line, and deleted from the cluster
straight afterwards. Verify from outside the working clone before believing the push:
`git ls-remote --tags origin` should resolve every tag.

Do not spend time debugging the https push itself while the deadline is close. Bundle and go.

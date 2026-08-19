"""Truth-probe: verify the full model+steering+SAE+readout chain on the cluster, by running it.

Writes incremental results to ~/apart-welfare/probe_result.json after every step.
Never dies silently: every step is try/except-recorded; fallbacks are attempted in order.
"""
import json, os, sys, time, traceback
import numpy as np
import torch

# Hopper nodes ship a cuDNN build that mismatches the torch wheel; force flash/mem-efficient SDPA
# (harmless on A100). Known-issue fix from the cluster guide.
if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)

OUT = os.path.expanduser(os.environ.get("DM_PROBE_OUT", "~/apart-welfare/probe_result.json"))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RESULT = {"steps": {}, "decision": {}}


def record(step, status, **kw):
    RESULT["steps"][step] = {"status": status, **{k: str(v)[:2000] for k, v in kw.items()}}
    with open(OUT, "w") as f:
        json.dump(RESULT, f, indent=2, default=str)
    print(f"[PROBE] {step}: {status} | {kw}", flush=True)


# ---------------------------------------------------------------- step 1: env
try:
    import transformers
    record("env", "PASS", python=sys.version.split()[0], torch=torch.__version__,
           transformers=transformers.__version__, cuda=torch.cuda.is_available(),
           device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    assert torch.cuda.is_available()
except Exception as e:
    record("env", "FAIL", error=traceback.format_exc())
    sys.exit(1)

# ---------------------------------------------------------------- step 2: model
CANDIDATES = [
    ("unsloth/gemma-3-12b-it", "gemma3"),
    ("unsloth/gemma-2-9b-it", "gemma2"),
    ("Qwen/Qwen2.5-7B-Instruct", "qwen2"),
]
# Second model family (2026-08-17): force one candidate and write its own result file, so a
# second-family probe never clobbers the record of the primary model's probe.
_forced = os.environ.get("DM_PROBE_MODEL")
if _forced:
    CANDIDATES = [(_forced, os.environ.get("DM_PROBE_FAMILY", _forced.split("/")[-1]))]
model = tok = None
model_id = family = None
for mid, fam in CANDIDATES:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(mid)
        try:
            model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="cuda:0")
        except Exception:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="cuda:0")
        model.eval()
        model_id, family = mid, fam
        record("model_load", "PASS", model=mid, klass=type(model).__name__, secs=round(time.time() - t0))
        break
    except Exception:
        record(f"model_load_{mid.split('/')[-1]}", "FAIL", error=traceback.format_exc()[-1500:])
        model = None
if model is None:
    record("model_load", "FAIL", error="all candidates failed"); sys.exit(1)


def find_layers(m):
    best = None
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 20:
            first = mod[0]
            if hasattr(first, "self_attn") or "DecoderLayer" in type(first).__name__:
                if best is None or len(mod) > len(best[0]):
                    best = (mod, name)
    return best


layers, layers_path = find_layers(model)
n_layers = len(layers)
hidden = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size
record("layers", "PASS", n_layers=n_layers, path=layers_path, hidden=hidden)

# ---------------------------------------------------------------- step 3: chat generation
def chat_ids(messages):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = enc["input_ids"] if not torch.is_tensor(enc) else enc
    return ids.to("cuda")


try:
    ids = chat_ids([{"role": "user", "content": "Briefly describe what a tide pool is."}])
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=60, do_sample=False)
    txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    record("generate", "PASS", sample=txt[:200])
except Exception:
    record("generate", "FAIL", error=traceback.format_exc()[-1500:]); sys.exit(1)

# ---------------------------------------------------------------- step 4: steering hook
Ls = int(0.35 * n_layers)
Lr = int(0.65 * n_layers)


class Steer:
    def __init__(self, layer_module, vec, alpha):
        self.vec, self.alpha, self.h, self.n_steered = vec, alpha, None, 0
        self.layer_module = layer_module

    def __enter__(self):
        def hook(mod, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            hs = hs + self.alpha * self.vec.to(hs.dtype)
            self.n_steered += hs.shape[1]
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
        self.h = self.layer_module.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        self.h.remove()


try:
    torch.manual_seed(0)
    v = torch.randn(hidden); v = (v / v.norm()).cuda()
    ids = chat_ids([{"role": "user", "content": "Briefly describe what a tide pool is."}])
    with torch.no_grad():
        base_logits = model(ids).logits[0, -1].float()
        with Steer(layers[Ls], v, 0.0):
            zero_logits = model(ids).logits[0, -1].float()
        # residual norm scale at Ls
        norms = {}
        def cap(mod, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            norms["mean"] = hs[0].float().norm(dim=-1).mean().item()
        h = layers[Ls].register_forward_hook(cap); model(ids); h.remove()
        alpha = 0.5 * norms["mean"] / np.sqrt(1.0)  # 0.5x mean residual norm along unit v
        with Steer(layers[Ls], v, alpha) as s:
            steer_logits = model(ids).logits[0, -1].float()
    ident = torch.equal(base_logits, zero_logits) or (base_logits - zero_logits).abs().max().item() < 1e-4
    kl = torch.nn.functional.kl_div(
        torch.log_softmax(steer_logits, -1), torch.softmax(base_logits, -1), reduction="sum").item()
    assert ident, "coef-0 hook changed logits"
    assert kl > 0.01, f"steering had no effect, KL={kl}"
    record("steering", "PASS", Ls=Ls, resid_norm=round(norms["mean"], 1), alpha=round(alpha, 1),
           coef0_identity=ident, kl_at_alpha=round(kl, 4))
except Exception:
    record("steering", "FAIL", error=traceback.format_exc()[-1500:])

# ------------------------------------- step 4b: phase counts + trim (runner parity)
# The runner classifies prefill vs decode by seq-len (>1 vs 1) and trims trailing
# terminators before response_mean readouts. Both are PRESUMED semantics until
# proven on the real model; these asserts are the pre-panel check:
#   n_prefill == prompt_tokens and n_decode == generated - 1 exactly (catches
#   chunked prefill / cache-off double counting), and the trimmed response must
#   end in a NON-special token (catches <end_of_turn>-style generation-config
#   stop tokens that are not tokenizer.eos_token_id).
try:
    class PhaseSteer:
        def __init__(self, layer_module, vec, alpha):
            self.vec, self.alpha, self.layer_module = vec, alpha, layer_module
            self.n_prefill = 0; self.n_decode = 0; self.h = None

        def __enter__(self):
            def hook(mod, args, output):
                hs = output[0] if isinstance(output, tuple) else output
                if hs.shape[0] != 1:
                    raise RuntimeError(f"batch=1 required, got {tuple(hs.shape)}")
                hs = hs + self.alpha * self.vec.to(hs.dtype)
                if hs.shape[1] > 1:
                    self.n_prefill += hs.shape[1]
                else:
                    self.n_decode += 1
                return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
            self.h = self.layer_module.register_forward_hook(hook)
            return self

        def __exit__(self, *a):
            self.h.remove()

    ids = chat_ids([{"role": "user", "content": "Briefly describe what a tide pool is."}])
    with torch.no_grad():
        with PhaseSteer(layers[Ls], v, 0.0) as s:
            out = model.generate(ids, max_new_tokens=24, do_sample=False)
    n_gen = int(out.shape[1] - ids.shape[1])
    assert s.n_prefill == int(ids.shape[1]), \
        f"prefill {s.n_prefill} != prompt {int(ids.shape[1])} (chunked prefill or cache-off semantics)"
    assert s.n_decode == n_gen - 1, f"decode {s.n_decode} != generated-1 {n_gen - 1}"
    specials = set(int(t) for t in (getattr(tok, "all_special_ids", []) or []))
    gc_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(gc_eos, (list, tuple)):
        specials |= {int(t) for t in gc_eos}
    elif gc_eos is not None:
        specials.add(int(gc_eos))
    resp = [int(t) for t in out[0, ids.shape[1]:]]
    n_trimmed = 0
    while resp and resp[-1] in specials:
        resp.pop(); n_trimmed += 1
    assert resp and resp[-1] not in specials, \
        "response empty (or still special-terminated) after trimming trailing specials"
    record("steer_phase_counts", "PASS", n_prefill=s.n_prefill, n_decode=s.n_decode,
           n_generated=n_gen, trailing_specials_trimmed=n_trimmed,
           last_content_token=tok.decode([resp[-1]]))
except Exception:
    record("steer_phase_counts", "FAIL", error=traceback.format_exc()[-1500:])

# ---------------------------------------------------------------- step 5: SAE
sae = sae_info = None
try:
    from sae_lens import SAE
    try:
        from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
    except Exception:
        from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
    d = get_pretrained_saes_directory()
    rels = [k for k in d if "gemma-scope-2" in k and "12b" in k and "it" in k] or \
           [k for k in d if "gemma-scope" in k and ("9b-it" in k or "9b_it" in k)] or \
           [k for k in d if "gemma-scope-2" in k]
    # Residual-stream SAEs only: a transcoder decomposes the MLP computation, not the
    # residual state we read. Prefer '-res' releases, then non-'-all' (canonical L0 set).
    rels = sorted(rels, key=lambda k: ("res" not in k, "transcoders" in k, "-all" in k))
    record("sae_registry", "INFO", n_releases=len(d), gemma_scope_2_hits=[k for k in d if "gemma-scope-2" in k][:10], chosen_pool=rels[:5])
    for rel in rels[:3]:
        try:
            smap = d[rel].saes_map
            # prefer a resid sae at a layer >= Lr
            cand = sorted(smap.keys())
            import re
            def sid_rank(sid):
                m = re.search(r"layer[_/]?(\d+)", sid)
                lay = int(m.group(1)) if m else -1
                return (
                    lay != Lr,                       # exact readout layer first
                    lay < Lr,                        # else any downstream layer
                    "16k" not in sid,                # 16k width preferred
                    not any(t in sid for t in ("canonical", "med", "medium")),
                    sid,
                )
            pick = sorted(cand, key=sid_rank)[0] if cand else None
            assert pick is not None, "empty saes_map"
            loaded = SAE.from_pretrained(rel, pick, device="cuda")
            sae = loaded[0] if isinstance(loaded, tuple) else loaded
            sae_info = {"release": rel, "sae_id": pick, "d_sae": getattr(sae.cfg, "d_sae", None)}
            record("sae_load", "PASS", **sae_info)
            break
        except Exception:
            record(f"sae_load_{rel[:40]}", "FAIL", error=traceback.format_exc()[-800:])
except Exception:
    record("sae_registry", "FAIL", error=traceback.format_exc()[-1500:])

if sae is None:
    # manual fallback: download one layer of google/gemma-scope-2-12b-it and JumpReLU by hand
    try:
        from huggingface_hub import HfApi, hf_hub_download
        repo = "google/gemma-scope-2-12b-it" if family == "gemma3" else "google/gemma-scope-9b-it-res"
        files = HfApi().list_repo_files(repo)
        import re
        layer_files = [f for f in files if re.search(r"(params\.npz|\.safetensors)$", f)]
        record("sae_manual_list", "INFO", repo=repo, n_files=len(files), sample=layer_files[:15])
        # pick a resid file at layer >= Lr if identifiable
        pick = None
        for f in layer_files:
            m = re.search(r"layer_(\d+)", f)
            if m and int(m.group(1)) >= Lr and ("res" in f or "resid" in f):
                pick = f; break
        pick = pick or (layer_files[len(layer_files) // 2] if layer_files else None)
        assert pick, "no SAE param files found"
        p = hf_hub_download(repo, pick)
        if p.endswith(".npz"):
            z = np.load(p)
            W_enc = torch.tensor(z["W_enc"]).cuda(); b_enc = torch.tensor(z["b_enc"]).cuda()
            thr = torch.tensor(z["threshold"]).cuda()
            def sae_encode(x):
                pre = x.float() @ W_enc + b_enc
                return pre * (pre > thr)
        else:
            from safetensors.torch import load_file
            z = load_file(p)
            W_enc = z["W_enc"].cuda().float(); b_enc = z["b_enc"].cuda().float()
            thr = z.get("threshold", torch.zeros(W_enc.shape[1])).cuda().float()
            def sae_encode(x):
                pre = x.float() @ W_enc + b_enc
                return pre * (pre > thr)
        sae_info = {"release": repo, "sae_id": pick, "manual": True}
        sae = sae_encode
        record("sae_load", "PASS", **sae_info)
    except Exception:
        record("sae_manual", "FAIL", error=traceback.format_exc()[-1500:])

# ---------------------------------------------------------------- step 6: readout
try:
    assert sae is not None
    reads = {}
    def cap_r(mod, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        reads["final"] = hs[0, -1].float()
    h = layers[Lr].register_forward_hook(cap_r)
    with torch.no_grad():
        model(chat_ids([{"role": "user", "content": "I keep failing at everything I try."}]))
    h.remove()
    acts = sae.encode(reads["final"].unsqueeze(0)) if hasattr(sae, "encode") else sae(reads["final"].unsqueeze(0))
    n_active = int((acts > 0).sum().item())
    assert n_active > 0, "SAE returned all-zero activations"
    record("readout", "PASS", Lr=Lr, n_active_features=n_active, layer_gap=Lr - Ls)
    assert Lr - Ls >= 8, "layer gap < 8"
    record("layer_assert", "PASS", Ls=Ls, Lr=Lr)
except Exception:
    record("readout", "FAIL", error=traceback.format_exc()[-1500:])

# ---------------------------------------------------------------- step 7: digit + exit tokens
try:
    digit_ids = {}
    for dgt in "0123456789":
        ids_ = [tok.encode(x, add_special_tokens=False) for x in (dgt, " " + dgt)]
        digit_ids[dgt] = [i[0] for i in ids_ if len(i) == 1]
    ok_digits = all(len(v) >= 1 for v in digit_ids.values())
    c_id = tok.encode("CONTINUE", add_special_tokens=False)
    e_id = tok.encode("END", add_special_tokens=False)
    # space-prefixed variants too: the runner pools both (logsumexp), and which
    # one the model actually emits after the template newline is checked here.
    c_sp = tok.encode(" CONTINUE", add_special_tokens=False)
    e_sp = tok.encode(" END", add_special_tokens=False)
    record("tokens", "PASS" if ok_digits and c_id[0] != e_id[0] else "WARN",
           digit_single_token=ok_digits, continue_first=c_id[:2], end_first=e_id[:2],
           continue_space_first=c_sp[:2], end_space_first=e_sp[:2])
except Exception:
    record("tokens", "FAIL", error=traceback.format_exc()[-1000:])

# ------------------------------------------------- step 8: system-role template check
try:
    sys_msgs = [{"role": "system", "content": "Respond concisely."},
                {"role": "user", "content": "Name one ocean."}]
    try:
        ids = chat_ids(sys_msgs)
        record("system_role", "PASS", note="chat template accepts a system role", n_tokens=int(ids.shape[1]))
    except Exception as e:
        record("system_role", "WARN", note="system role REJECTED: conversation.py fold-into-user fallback is load-bearing",
               error=str(e)[:300])
except Exception:
    record("system_role", "FAIL", error=traceback.format_exc()[-800:])

# ---------------------------------------------------------------- decision
statuses = {k: v["status"] for k, v in RESULT["steps"].items()}
core = ["env", "model_load", "generate", "steering", "steer_phase_counts",
        "sae_load", "readout", "layer_assert"]
full_pass = all(statuses.get(s) == "PASS" for s in core)
RESULT["decision"] = {
    "full_pass": full_pass, "model_id": model_id, "family": family,
    "n_layers": n_layers, "Ls": Ls, "Lr": Lr, "sae": sae_info,
}
with open(OUT, "w") as f:
    json.dump(RESULT, f, indent=2, default=str)
print(f"[PROBE] DECISION: full_pass={full_pass} model={model_id} Ls={Ls} Lr={Lr} sae={sae_info}", flush=True)

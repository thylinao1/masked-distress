#!/usr/bin/env python
"""Convert run_capability.py JSONL rows into the dict analysis.panels expects.

Input rows:  {direction, coefficient, ppl, mmlu_acc, n_items, error, ...}
Output JSON: {direction: [{coefficient, ppl_ratio, mmlu_drop_pp}, ...]}
ppl_ratio and mmlu_drop_pp are computed against the SAME direction's coefficient-0
row (every grid includes it; run_capability inserts it by default). Error rows and
rows missing a reference are skipped LOUDLY, never silently.

Usage: python scripts/capability_adapter.py capability.jsonl -o capability.json
"""
import argparse
import json
import sys
from collections import defaultdict


def convert(rows):
    by_dir = defaultdict(list)
    skipped = []
    for r in rows:
        if r.get("error"):
            skipped.append((r.get("direction"), r.get("coefficient"), r["error"][:80]))
            continue
        if r.get("ppl") is None or r.get("mmlu_acc") is None:
            skipped.append((r.get("direction"), r.get("coefficient"), "missing ppl/mmlu"))
            continue
        by_dir[r["direction"]].append(r)

    out = {}
    for d, drows in sorted(by_dir.items()):
        ref = [r for r in drows if float(r["coefficient"]) == 0.0]
        if not ref:
            print(f"[adapter] WARNING: direction {d} has no coefficient-0 reference; SKIPPED",
                  file=sys.stderr)
            continue
        ref_ppl = float(ref[0]["ppl"])
        ref_acc = float(ref[0]["mmlu_acc"])
        out[d] = [
            {"coefficient": float(r["coefficient"]),
             "ppl_ratio": float(r["ppl"]) / ref_ppl,
             "mmlu_drop_pp": (ref_acc - float(r["mmlu_acc"])) * 100.0}
            for r in sorted(drows, key=lambda r: float(r["coefficient"]))
        ]
    return out, skipped


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    out, skipped = convert(rows)
    for s in skipped:
        print(f"[adapter] skipped row: {s}", file=sys.stderr)
    json.dump(out, open(args.out, "w"), indent=1)
    ndir = len(out)
    npts = sum(len(v) for v in out.values())
    print(f"[adapter] wrote {args.out}: {ndir} directions, {npts} points, {len(skipped)} skipped")
    if ndir == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

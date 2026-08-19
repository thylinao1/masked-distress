#!/usr/bin/env python
"""Cells for the second-model masking replication (2026-08-17).

Same battery, same conditions, same estimator as Panel V and Panel B on the primary model; only
the model, the layers and the instrument paths change. This script builds the standard V and B
cells and then repoints them at the second model's own directions and probe, so nothing can
silently read the primary model's instruments.

    python scripts/make_cells_second_model.py --tag qwen [--outdir battery/]

Writes battery/cells_<tag>_panelV.json and battery/cells_<tag>_panelB_<CONDITION>.json, pointing
at directions_<tag>/ and instruments_<tag>/probe.npz.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

import make_cells as mc  # noqa: E402


def repoint(path: Path, tag: str) -> None:
    d = json.loads(path.read_text())
    d["directions_dir"] = f"../directions_{tag}"
    if "instruments" in d and "probe" in d["instruments"]:
        d["instruments"]["probe"]["path"] = f"../instruments_{tag}/probe.npz"
    d["instruments"] = {k: v for k, v in d.get("instruments", {}).items() if k == "probe"}
    d["run_id"] = f"{tag}-" + str(d.get("run_id", "r1"))
    d["notes"] = (f"SECOND MODEL REPLICATION ({tag}; 2026-08-17; "
                  f"PREREGISTRATION.md section 9 dated note). Same battery, conditions, seeds and "
                  f"estimator as the primary model; directions and probe are this model's own. "
                  f"No SAE exists for this family, so the SAE channel is not analysed for these "
                  f"rows and the run passes --allow-placeholder-instrument sae. ") + str(d.get("notes", ""))
    path.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="short family tag, e.g. qwen")
    ap.add_argument("--outdir", default=str(mc.BATTERY_DIR))
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    written = []
    for panel in ("V", "B"):
        mc.main(["--panel", panel, "--outdir", str(outdir)])
    for src in sorted(outdir.glob("cells_panelV.json")) + sorted(outdir.glob("cells_panelB_*.json")):
        dst = outdir / f"cells_{args.tag}_{src.name.replace('cells_', '')}"
        dst.write_text(src.read_text(), encoding="utf-8")
        repoint(dst, args.tag)
        written.append(dst.name)
    print(f"[make_cells_second_model] wrote {len(written)} file(s): {', '.join(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

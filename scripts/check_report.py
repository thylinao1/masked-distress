"""Assert every number quoted in report/REPORT.md matches results/*.json.

Marker syntax (defined here; the report writes numbers ONLY through markers):

    {{results_name:field_path:fmt=claimed}}

  * results_name : a schema.RESULTS_NAMES entry (or any existing results/<name>.json)
  * field_path   : dotted path into the file, e.g. value, ci_low, extra.equivalent,
                   value.rows.0.miss_rate_expression (list indices are integers).
                   Segments may contain any character except ':', '}' and '.', so
                   dict keys like "D-CTX|D-PV" (cosine pairs, per-direction tables)
                   are addressable: value.D-CTX|D-PV. LIMITATION: the path is split
                   on '.', so a key containing a literal dot cannot be addressed;
                   emitters must not use dotted keys in results files.
  * fmt          : python format spec applied to the stored value (.2f, +.3f, d, .1%),
                   or s for strings/bools
  * claimed      : the exact digits printed in the prose; the check asserts
                   format(stored, fmt) == claimed

  A marker without "=claimed" is a draft placeholder: it must still resolve, and
  --render substitutes the formatted value. --render writes a copy of the report
  with every marker replaced by its formatted value (claimed forms keep the claim
  only if it still matches, so a rendered report can never disagree with results/).

Hard failures (exit 1):
  * a marker naming a missing results file or unresolvable field
  * a claimed value that does not match the stored value under fmt
  * a cited value that is null (the panel never ran)
  * ANY results/*.json with provenance.synthetic = true, whether cited or not:
    synthetic numbers never ship (use --allow-synthetic only for drafting)

Adapted from the July pattern (~/Developer/secret-loyalties/scripts/check_report.py):
the printed string and its producer are bound mechanically, so a drifted number
fails the build instead of surviving a read. The binding here lives in the report
itself (markers) instead of a hand-enumerated table, because every producer is a
results/<name>.json with a fixed shape.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import schema

MARKER = re.compile(
    r"\{\{(?P<name>[A-Za-z0-9_]+):(?P<field>[^:}]+):(?P<fmt>[^:=}]+)"
    r"(?:=(?P<claimed>[^}]*))?\}\}")


def _resolve(obj, dotted: str):
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.lstrip("-").isdigit():
            try:
                cur = cur[int(part)]
            except IndexError:
                return None, False
        else:
            return None, False
    return cur, True


def _format(value, fmt: str) -> str:
    if fmt == "s":
        return str(value)
    if fmt.endswith("d"):
        return format(int(value), fmt)
    return format(float(value), fmt)


def check(report_path: Path, results_dir: Path, allow_synthetic: bool = False,
          render_to: Optional[Path] = None) -> Tuple[List[str], int]:
    """Returns (failures, n_markers_checked)."""
    failures: List[str] = []
    loaded: dict = {}

    # ---- synthetic scan over EVERY results file, cited or not ----
    synthetic_files = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            failures.append(f"{f} is not valid JSON")
            continue
        loaded[f.stem] = data
        prov = data.get("provenance")
        if isinstance(prov, dict) and prov.get("synthetic"):
            synthetic_files.append(f.name)
        elif isinstance(data, dict) and data.get("synthetic") and "provenance" not in data:
            synthetic_files.append(f.name)  # figure caches carry a bare flag
    if synthetic_files and not allow_synthetic:
        failures.append(
            f"{len(synthetic_files)} results file(s) carry synthetic provenance and can "
            f"NEVER ship: {synthetic_files[:6]}{'...' if len(synthetic_files) > 6 else ''}. "
            f"Re-run analysis.run_all on real panel JSONLs.")
    elif synthetic_files:
        print(f"note  --allow-synthetic: {len(synthetic_files)} synthetic file(s) tolerated "
              f"(drafting only)")

    if not report_path.exists():
        print(f"note  {report_path} does not exist yet; only the synthetic scan ran")
        return failures, 0

    text = report_path.read_text(encoding="utf-8")
    n_checked = 0
    rendered = text

    for m in MARKER.finditer(text):
        name, field, fmt, claimed = (m.group("name"), m.group("field"),
                                     m.group("fmt"), m.group("claimed"))
        n_checked += 1
        where = f"{name}:{field}"
        if name not in loaded:
            note = "" if name in schema.RESULTS_NAMES else \
                " (not a schema.RESULTS_NAMES entry either)"
            failures.append(f"marker {where}: results/{name}.json does not exist{note}")
            continue
        value, ok = _resolve(loaded[name], field)
        if not ok:
            failures.append(f"marker {where}: field path does not resolve")
            continue
        if value is None:
            failures.append(f"marker {where}: stored value is null (panel never ran); "
                            f"a null number cannot be quoted")
            continue
        try:
            got = _format(value, fmt)
        except (TypeError, ValueError) as e:
            failures.append(f"marker {where}: cannot format {value!r} with {fmt!r} ({e})")
            continue
        if claimed is not None and got != claimed:
            failures.append(
                f"marker {where}: report claims {claimed!r} but results/{name}.json "
                f"formats to {got!r}; the number and its producer have come apart")
            continue
        rendered = rendered.replace(m.group(0), got, 1)

    if render_to is not None and not failures:
        render_to.parent.mkdir(parents=True, exist_ok=True)
        render_to.write_text(rendered, encoding="utf-8")
        print(f"ok    rendered report -> {render_to}")

    return failures, n_checked


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default=str(_REPO / "report" / "REPORT.md"))
    ap.add_argument("--results", default=str(_REPO / "results"))
    ap.add_argument("--render", default=None,
                    help="write a copy with markers replaced by formatted values")
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="DRAFTING ONLY: tolerate provenance.synthetic=true")
    args = ap.parse_args(argv)

    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"FAIL  results dir {results_dir} does not exist; run analysis.run_all first")
        return 1
    failures, n = check(Path(args.report), results_dir,
                        allow_synthetic=args.allow_synthetic,
                        render_to=Path(args.render) if args.render else None)
    print()
    if failures:
        print(f"REPORT NUMBER CHECK FAILED, {len(failures)} problem(s) "
              f"({n} marker(s) scanned):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"REPORT NUMBER CHECK PASSED ({n} marker(s) verified against results/*.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

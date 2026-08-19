"""Build the submission PDF from report/report.tex, with two gates around the build.

The July pipeline (~/Developer/secret-loyalties/scripts/build_pdf.py) converted REPORT.md
to LaTeX mechanically. That converter is not ported here: this sprint's report.tex is
maintained by hand, so the risk it removes is different. What can go wrong here is DRIFT,
the .md and the .tex disagreeing about which results file feeds which claim, because
scripts/check_report.py only verifies REPORT.md.

TWO TeX COMMANDS CARRY A RESULTS NAME, and both feed the drift check:

  \\pend{name}          a number that has NOT landed. Typesets as a conspicuous bracketed
                       name. A submitted PDF must have zero of these.
  \\pnum{name}{value}   a number that HAS landed. Typesets as the value alone, so nothing
                       marker-like reaches the page, while the results name it came from
                       stays readable in the source.

The second command exists because of what finalization does to this check. Once the
numbers are filled in, a .tex with no results names left in it would trivially "agree"
with a .md with no markers left in it, and the drift check would pass by being empty on
both sides, at exactly the moment there is a real PDF to get wrong. Keeping the name
beside the filled value means the parity check stays load-bearing on the finished report:
report.tex and REPORT.md must still cite the same 18 results files, and REPORT.md's
markers are in check_report.py's claimed form, so the printed digits are asserted against
results/*.json as well.

So this script checks three things and then builds:

  1. Every \\pend{name} and \\pnum{name}{value} in report.tex names a schema.RESULTS_NAMES
     entry (or is an explicitly labelled "TBD ..." hand-filled number, one of the short
     table aliases listed in TABLE_ALIASES, or an existing results/<name>.json that is not
     a RESULTS_NAMES entry, such as a figure cache; those are excluded from the parity set
     on this side exactly as they already are on the REPORT.md side). A name matching none
     of those is a typo and fails.
  2. The set of results files referenced by report.tex equals the set referenced by
     REPORT.md's check_report markers. A results file quoted in one and not the other is
     drift and fails.
  3. tectonic builds, and the TeX log carries zero overfull boxes wider than 1pt. An
     overfull hbox is text standing in the margin of a graded PDF.

It also counts surviving \\pend placeholders. A submitted PDF must have zero: run
scripts/check_report.py --render to read the real values, write them into both files as
claimed markers (.md) and \\pnum calls (.tex), then rebuild.

Run: python report/build_pdf.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import schema  # noqa: E402

TEX = _REPO / "report" / "report.tex"
MD = _REPO / "report" / "REPORT.md"
PDF = _REPO / "report" / "report.pdf"
RESULTS = _REPO / "results"

# Short names used inside the two wide result tables, where the full results name would
# not fit the column. Both resolve to one results file.
TABLE_ALIASES = {"cos": "cosine_matrix", "ct": "countermeasure_table"}

PEND = re.compile(r"\\pend\{([^}]*)\}")
PNUM = re.compile(r"\\pnum\{([^}]*)\}\{")
PNUM_VAL = re.compile(r"\\pnum\{([^}]*)\}\{((?:[^{}]|\{[^{}]*\})*)\}")
CLAIM = re.compile(r"\{\{(?P<name>[A-Za-z0-9_]+):[^:}]+:[^:=}]+=(?P<claimed>[^}]*)\}\}")
MARKER = re.compile(r"\{\{(?P<name>[A-Za-z0-9_]+):")
OVERFULL = re.compile(r"Overfull \\hbox \((?P<pt>[0-9.]+)pt too wide\)")


def _unescape(name: str) -> str:
    return name.replace("\\_", "_").replace("\\%", "%").strip()


def _detex(value: str) -> str:
    """Strip the LaTeX spelling from a printed number so it can be compared."""
    for a, b in (("\\%", "%"), ("$-$", "-"), ("$+$", "+"), ("{,}", ","),
                 ("\\allowbreak{}", ""), ("$\\geq$", ">="), ("\\_", "_")):
        value = value.replace(a, b)
    return value.strip()


def collect() -> tuple[list[str], set[str], set[str], int]:
    """Returns (problems, tex_results_names, md_results_names, n_placeholders)."""
    problems: list[str] = []
    tex = TEX.read_text(encoding="utf-8")
    md = MD.read_text(encoding="utf-8")

    tex_names: set[str] = set()
    n_placeholders = len(PEND.findall(tex))
    for cmd, raw in ([("pend", r) for r in PEND.findall(tex)]
                     + [("pnum", r) for r in PNUM.findall(tex)]):
        name = _unescape(raw)
        if name.startswith("TBD"):
            continue  # hand-filled number with no results/*.json producer
        name = TABLE_ALIASES.get(name, name)
        if name in schema.RESULTS_NAMES:
            tex_names.add(name)
            continue
        if (RESULTS / f"{name}.json").exists():
            # An existing results file that is not a canonical endpoint (a figure cache,
            # say). REPORT.md's side of the parity set already filters these out, so this
            # side must too, or a name quoted in both files would read as drift.
            continue
        problems.append(
            f"report.tex: \\{cmd}{{{raw}}} is not a schema.RESULTS_NAMES entry, not a "
            f"results/*.json file, and not a TBD placeholder")

    md_names = {m.group("name") for m in MARKER.finditer(md)}
    md_names = {n for n in md_names if n in schema.RESULTS_NAMES}

    # A filled \pnum carries the printed digits, and check_report.py never reads this
    # file, so a value retyped into the .tex would reach the PDF unchecked while
    # REPORT.md still passed. Assert that every printed value is one REPORT.md claims
    # for that same results name; a claim REPORT.md does not make cannot be printed.
    claims: dict = {}
    for m in CLAIM.finditer(md):
        claims.setdefault(m.group("name"), set()).add(m.group("claimed"))
    for raw_name, raw_val in PNUM_VAL.findall(tex):
        name = TABLE_ALIASES.get(_unescape(raw_name), _unescape(raw_name))
        if name not in claims:
            continue  # a figure cache or a TBD; the name check above already ruled on it
        val = _detex(raw_val)
        if val not in claims[name]:
            problems.append(
                f"report.tex prints {val!r} for {name}, which REPORT.md never claims "
                f"(it claims {sorted(claims[name])[:8]}): a retyped number")

    only_tex = sorted(tex_names - md_names)
    only_md = sorted(md_names - tex_names)
    if only_tex:
        problems.append(f"quoted in report.tex but not in REPORT.md (unverified): {only_tex}")
    if only_md:
        problems.append(f"quoted in REPORT.md but not in report.tex (dropped claim): {only_md}")

    return problems, tex_names, md_names, n_placeholders


def check_boxes(log: Path) -> list[str]:
    if not log.exists():
        return ["no TeX log was written, so box warnings could not be checked"]
    bad = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = OVERFULL.search(line)
        if m and float(m.group("pt")) > 1.0:
            bad.append(line.strip())
    return bad


def main() -> int:
    problems, tex_names, md_names, n_pend = collect()
    print(f"note  {len(tex_names)} results file(s) cited by report.tex, "
          f"{len(md_names)} by REPORT.md")
    if problems:
        print("FAIL  report.tex and REPORT.md have come apart:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("ok    report.tex and REPORT.md cite the same results files")

    if shutil.which("tectonic") is None:
        print("FAIL  tectonic is not on PATH, and it is the only TeX engine installed here")
        return 1

    proc = subprocess.run(
        ["tectonic", "--keep-logs", "--outdir", str(TEX.parent), str(TEX)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        print("FAIL  tectonic exited nonzero")
        print(proc.stderr[-4000:])
        return 1

    bad = check_boxes(TEX.with_suffix(".log"))
    if bad:
        print(f"FAIL  {len(bad)} overfull box(es) wider than 1pt; text is in the margin:")
        for line in bad[:12]:
            print(f"  - {line}")
        return 1
    print("ok    zero overfull boxes")

    print(f"ok    wrote {PDF}")
    if n_pend:
        print(f"WARN  {n_pend} \\pend placeholder(s) still in the PDF. A submitted PDF "
              f"must have zero: run scripts/check_report.py --render, write the values "
              f"into REPORT.md and report.tex, then rebuild.")
    else:
        print("ok    zero \\pend placeholders; every number is a filled \\pnum whose "
              "results name is still checked above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

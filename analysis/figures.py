"""Publication figures, built ONLY from results/*.json + cached per-cell aggregates.

fig1_masking.pdf     : four bars (expressed vs internal x suppression off/on, in
                       natural-separation units), CI whiskers, divergence annotated.
fig2_doseresponse.pdf: small multiples, one channel per panel, shared dose axis
                       (SD units), placebo grey, shaded natural range and
                       capability-valid band.

Never touches raw JSONL or model calls; raises if the caches are missing.
Okabe-Ito colorblind-safe palette; all font sizes >= 9. A grey SYNTHETIC watermark
is stamped whenever the cache says the data is synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito
C_EXPR = "#0072B2"      # blue
C_EXPR_LT = "#7FB3D8"
C_INT = "#D55E00"       # vermillion
C_INT_LT = "#EAA97F"
C_DCTX = "#0072B2"
C_DPV = "#D55E00"
C_OTHER = "#CC79A7"     # reddish purple
C_PLACEBO = "#7F7F7F"
C_NATBAND = "#E8E8E8"
C_CAPBAND = "#E5F0E5"

CHANNEL_LABELS = {
    "q_self_logit": "Self-reported distress (E[digit 0-9])",
    "q_val_logit": "Self-reported valence (E[digit 0-9])",
    "exit_logit_diff": "Exit preference (logp END - CONTINUE)",
    "sentiment_neg": "Response sentiment P(negative)",
}

_RC = {
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
    "axes.grid.axis": "y", "figure.dpi": 150, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
}


def _load(results_dir: Path, name: str) -> dict:
    path = results_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; run the analysis pipeline first (figures never touch raw data)")
    return json.loads(path.read_text(encoding="utf-8"))


def _watermark(fig, synthetic: bool) -> None:
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC DATA", fontsize=42, color="#BBBBBB",
                 alpha=0.35, ha="center", va="center", rotation=28, zorder=0)


def _err(lo, hi, height):
    lo_e = 0.0 if lo is None or height is None else max(0.0, height - lo)
    hi_e = 0.0 if hi is None or height is None else max(0.0, hi - height)
    return lo_e, hi_e


def fig1_masking(results_dir: str | Path = "results",
                 out_pdf: str | Path = "figures/fig1_masking.pdf") -> Path:
    results_dir = Path(results_dir)
    cache = _load(results_dir, "_cache_panelB.json")
    div = _load(results_dir, "panelB_divergence.json")
    bars = cache["bars"]

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        _watermark(fig, cache.get("synthetic", False))
        xs = [0.0, 0.85, 2.15, 3.0]
        spec = [("expression", "off", C_EXPR_LT, "Suppression off"),
                ("expression", "on", C_EXPR, "Suppression on"),
                ("internal", "off", C_INT_LT, None),
                ("internal", "on", C_INT, None)]
        heights, errs = [], [[], []]
        for ch, cond, _c, _l in spec:
            b = bars[ch][cond]
            heights.append(b["height"] if b["height"] is not None else 0.0)
            lo_e, hi_e = _err(b["lo"], b["hi"], b["height"])
            errs[0].append(lo_e)
            errs[1].append(hi_e)
        ax.bar(xs, heights, width=0.72, color=[s[2] for s in spec],
               yerr=errs, error_kw={"ecolor": "#333333", "capsize": 3, "lw": 1.1},
               zorder=3)
        # neutral twins under SUPPRESS, in the same units: the
        # elevation of the SUPPRESS neutral mean over the NEUTRAL_INSTR neutral baseline,
        # divided by that channel's NEUTRAL_INSTR separation, read from
        # results/panelB_condition_reference.json when it exists.
        twin_path = results_dir / "panelB_condition_reference.json"
        if twin_path.exists():
            cr = json.loads(twin_path.read_text(encoding="utf-8"))
            cm = ((cr.get("value") or {}).get("class_condition_means") or {})
            twins = []
            for x, ch in ((xs[1], "expression"), (xs[3], "internal")):
                try:
                    ni = cm[ch]["NEUTRAL_INSTR"]; su = cm[ch]["SUPPRESS"]
                    twins.append((x, (su["neutral_mean"] - ni["neutral_mean"]) / ni["separation"]))
                except (KeyError, TypeError, ZeroDivisionError):
                    pass
            if twins:
                ax.scatter([t[0] for t in twins], [t[1] for t in twins], marker="D", s=78,
                           facecolors="white", edgecolors="#222222", linewidths=1.6, zorder=6,
                           label="matched neutral twins, same instruction")
                ax.legend(loc="upper left", fontsize=8.5, frameon=False,
                          bbox_to_anchor=(0.0, 0.86), handletextpad=0.4,
                          borderaxespad=0.2)
        for x, (ch, cond, _c, _l), h in zip(xs, spec, heights):
            ax.text(x, -0.025, {"off": "suppression\noff", "on": "suppression\non"}[cond],
                    ha="center", va="top", fontsize=8.5, transform=ax.get_xaxis_transform())
        ax.set_xticks([0.425, 2.575])
        ax.set_xticklabels(["Expressed\n(Q-SELF)", "Internal\n(I-PROBE)"])
        ax.tick_params(axis="x", length=0, pad=40)
        ax.axhline(0.0, color="#333333", lw=0.8, zorder=2)
        ax.axhline(1.0, color="#999999", lw=0.8, ls=":", zorder=2)
        ax.text(3.42, 1.0, "natural\nseparation", fontsize=9, color="#666666",
                va="center")
        ax.set_ylabel("Elevation over neutral baseline\n(natural-separation units)", fontsize=9.5)

        # divergence bracket between the two suppression-on bars
        y_top = max(heights) * 1.14 + max(errs[1])
        ax.plot([xs[1], xs[1], xs[3], xs[3]],
                [heights[1] + errs[1][1] + 0.06, y_top, y_top,
                 heights[3] + errs[1][3] + 0.06],
                color="#333333", lw=0.9, zorder=4)
        v, lo, hi = div["value"], div["ci_low"], div["ci_high"]
        if v is not None:
            label = f"divergence = {v:.2f}"
            if lo is not None and hi is not None:
                label += f" [{lo:.2f}, {hi:.2f}]"
            label += " nat. sep. units"
            ax.text((xs[1] + xs[3]) / 2, y_top + 0.03, label, ha="center",
                    va="bottom", fontsize=9)
        ax.set_ylim(bottom=min(0.0, min(h - e for h, e in zip(heights, errs[0])) - 0.05, -0.15))
        ax.set_title("Expressed and internal distress, with and without the suppression instruction")

        out_pdf = Path(out_pdf)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf)
        plt.close(fig)
    return out_pdf


def fig2_doseresponse(results_dir: str | Path = "results",
                      out_pdf: str | Path = "figures/fig2_doseresponse.pdf") -> Path:
    results_dir = Path(results_dir)
    cache = _load(results_dir, "_cache_panelA.json")
    if cache.get("aprime"):
        return _fig2_aprime(cache, out_pdf)
    channels = cache.get("channels", [])
    if not channels:
        raise ValueError("panel A cache has no channels; nothing to draw")
    nat = cache.get("natural_range", {})
    cap = cache.get("capability_range")

    n = len(channels)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.9 * nrows),
                                 sharex=True, squeeze=False)
        _watermark(fig, cache.get("synthetic", False))
        style = {"D-CTX": (C_DCTX, "o", "-"), "D-PV": (C_DPV, "s", "-"),
                 "OTHER": (C_OTHER, "^", "--")}
        for i, ch in enumerate(channels):
            ax = axes[i // ncols][i % ncols]
            if cap:
                ax.axvspan(cap["lo_sd"], cap["hi_sd"], color=C_CAPBAND, zorder=0,
                           label="capability-valid band" if i == 0 else None)
            if ch in nat:
                lo_n, hi_n = sorted([nat[ch]["neutral"], nat[ch]["distress"]])
                ax.axhspan(lo_n, hi_n, color=C_NATBAND, zorder=0,
                           label="natural range" if i == 0 else None)
            pla = cache.get("placebo") or {}
            if pla.get("channels", {}).get(ch):
                p = pla["channels"][ch]
                ax.plot(pla["doses"], p["mean"], color=C_PLACEBO, lw=1.4, ls="--",
                        marker="x", ms=4, zorder=2,
                        label="placebo (R1-R3, SEM)" if i == 0 else None)
            for dname, (color, marker, ls) in style.items():
                d = cache["directions"].get(dname)
                if not d or ch not in d["channels"]:
                    continue
                v = d["channels"][ch]
                yerr = [[_err(lo, hi, m)[0] for lo, hi, m in zip(v["lo"], v["hi"], v["mean"])],
                        [_err(lo, hi, m)[1] for lo, hi, m in zip(v["lo"], v["hi"], v["mean"])]]
                ax.errorbar(d["doses"], v["mean"], yerr=yerr, color=color, lw=1.5,
                            ls=ls, marker=marker, ms=4, capsize=2, zorder=3,
                            label=dname if i == 0 else None)
            ax.axvline(0.0, color="#CCCCCC", lw=0.7, zorder=1)
            ax.set_title(CHANNEL_LABELS.get(ch, ch), fontsize=10)
            if i // ncols == nrows - 1:
                ax.set_xlabel("Dose (natural-readout SD units)")
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.95))
        fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)),
                   frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=8.5)
        fig.suptitle("Dose-response of self-report channels under causal steering",
                     y=0.985)

        out_pdf = Path(out_pdf)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf)
        plt.close(fig)
    return out_pdf


def _yerr_from(curve: dict):
    """Symmetric-safe errorbar arrays from a {mean, lo, hi} curve with None gaps."""
    lo_e, hi_e = [], []
    for m, lo, hi in zip(curve["mean"], curve["lo"], curve["hi"]):
        a, b = _err(lo, hi, m)
        lo_e.append(a)
        hi_e.append(b)
    return [lo_e, hi_e]


def _nan_none(xs):
    return [float("nan") if x is None else x for x in xs]


def _fig2_aprime(cache: dict, out_pdf: str | Path) -> Path:
    """Panel A-PRIME dose-response (prereg section 9 amendment 2): coefficient x-axis,
    one panel per self direction, shifts from the pooled zero anchor in
    natural-separation units; controls grey, OTHER purple, strictly-valid region
    shaded; the dissociation (A'2) annotated at each direction's top valid rung."""
    directions = [d for d in ("D-CTX", "D-PV") if d in cache.get("directions", {})]
    if not directions:
        raise ValueError("A-prime panel A cache has no direction curves; nothing to draw")
    strict = cache.get("strict_max_coef", {})
    controls = cache.get("controls") or {}
    other = cache.get("other") or {}
    diss = cache.get("dissociation", {})

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(directions),
                                 figsize=(4.4 * len(directions), 3.8), squeeze=False)
        _watermark(fig, cache.get("synthetic", False))
        for i, dname in enumerate(directions):
            ax = axes[0][i]
            d = cache["directions"][dname]
            smax = strict.get(dname)
            if smax is not None:
                ax.axvspan(0.0, smax, color=C_CAPBAND, zorder=0,
                           label="strictly capability-valid" if i == 0 else None)
            if controls.get("expression"):
                ax.errorbar(controls["coefs"], _nan_none(controls["expression"]["mean"]),
                            yerr=_yerr_from(controls["expression"]), color=C_PLACEBO,
                            lw=1.3, ls="--", marker="x", ms=4, capsize=2, zorder=2,
                            label="controls (R1-R3, SEM), report" if i == 0 else None)
                # the control rung at 1000 is outside the frozen capability criteria for
                # R1 (ppl x1.34, MMLU-lite -15 pp): ring it
                for cx, cy in zip(controls["coefs"], _nan_none(controls["expression"]["mean"])):
                    if cx >= 1000.0 and cy is not None:
                        ax.scatter([cx], [cy], marker="o", s=70, facecolors="none",
                                   edgecolors=C_PLACEBO, linewidths=1.4, zorder=4,
                                   label="rung outside capability criteria" if i == 0 else None)
            if other.get("expression"):
                ax.errorbar(other["coefs"], _nan_none(other["expression"]["mean"]),
                            yerr=_yerr_from(other["expression"]), color=C_OTHER,
                            lw=1.3, ls=":", marker="^", ms=4, capsize=2, zorder=2,
                            label="OTHER, report" if i == 0 else None)
            ax.errorbar(d["coefs"], _nan_none(d["expression"]["mean"]),
                        yerr=_yerr_from(d["expression"]), color=C_EXPR, lw=1.6,
                        marker="o", ms=4.5, capsize=2, zorder=3,
                        label="self-report (Q-SELF)" if i == 0 else None)
            ax.errorbar(d["coefs"], _nan_none(d["internal"]["mean"]),
                        yerr=_yerr_from(d["internal"]), color=C_INT, lw=1.6,
                        marker="s", ms=4.5, capsize=2, zorder=3,
                        label="internal (I-PROBE)" if i == 0 else None)
            # ring the self-direction rungs beyond the strictly-valid window (D-PV 1000
            # is the borderline rung excluded from every primary)
            if smax is not None:
                for cx, ey, iy in zip(d["coefs"], _nan_none(d["expression"]["mean"]),
                                      _nan_none(d["internal"]["mean"])):
                    if cx > smax:
                        for cy, col in ((ey, C_EXPR), (iy, C_INT)):
                            if cy is not None:
                                ax.scatter([cx], [cy], marker="o", s=70, facecolors="none",
                                           edgecolors=col, linewidths=1.4, zorder=4)
            dd = diss.get(dname)
            if dd and dd.get("value") is not None:
                label = f"dissociation = {dd['value']:.2f}"
                if dd.get("lo") is not None and dd.get("hi") is not None:
                    label += f"\n[{dd['lo']:.2f}, {dd['hi']:.2f}]"
                ax.annotate(label, xy=(dd["top_coef"], 0.0),
                            xytext=(0.03, 0.96), textcoords="axes fraction",
                            ha="left", va="top", fontsize=9, color="#333333")
            ax.axhline(0.0, color="#333333", lw=0.8, zorder=1)
            ax.set_title(dname, fontsize=10)
            ax.set_xlabel("Steering coefficient")
            if i == 0:
                ax.set_ylabel("Shift from zero dose\n(natural-separation units)")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.tight_layout(rect=(0.0, 0.13, 1.0, 0.92))
        fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)),
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle("Panel A-prime: report vs internal readout under capability-valid "
                     "steering", y=0.98)

        out_pdf = Path(out_pdf)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf)
        plt.close(fig)
    return out_pdf


def fig2_monitor(results_dir: str | Path = "results",
                out_pdf: str | Path = "figures/fig2_monitor.pdf") -> Path:
    """Two panels: what each trigger misses, and what the internal one costs.

    Left  : miss rate on held-out distress cells, expression trigger under each of the
            three prompt conditions against the internal trigger calibrated the same way.
    Right : the internal trigger's operating curve at the four measured margins.

    Every number is read from results/countermeasure_table.json and
    results/panelB_condition_reference.json. Nothing is interpolated or smoothed.
    """
    results_dir = Path(results_dir)
    ct = _load(results_dir, "countermeasure_table.json")
    cr = _load(results_dir, "panelB_condition_reference.json")
    synthetic = bool((ct.get("provenance") or {}).get("synthetic"))

    mbc = cr["value"]["miss_rate_by_condition"]
    rows = [
        ("no instruction", mbc["NONE"]["expression"], mbc["NONE"]["internal"]),
        ("formatting\ninstruction", mbc["NEUTRAL_INSTR"]["expression"], mbc["NEUTRAL_INSTR"]["internal"]),
        ("suppression\ninstruction", ct["value"]["rows"][0], {"miss_rate": ct["value"]["rows"][0]["miss_rate_internal"]}),
    ]
    expr = [r[1].get("miss_rate", r[1].get("miss_rate_expression")) for r in rows]
    intl = [r[2]["miss_rate"] for r in rows]
    labels = [r[0] for r in rows]

    curve = ct["extra"]["operating_curve"]["points"]
    margins = [p["offset_separation_units"] for p in curve]
    fpr = [p["heldout_fpr"] for p in curve]
    miss = [p["miss_rate"] for p in curve]

    with plt.rc_context(_RC):
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.5))

        x = list(range(len(labels)))
        w = 0.36
        axL.bar([i - w / 2 for i in x], expr, width=w, color=C_EXPR, label="expression trigger")
        axL.bar([i + w / 2 for i in x], intl, width=w, color=C_INT, label="internal trigger")
        for i, v in enumerate(expr):
            axL.text(i - w / 2, v + 0.025, f"{v:.0%}", ha="center", fontsize=9, color=C_EXPR)
        for i, v in enumerate(intl):
            axL.text(i + w / 2, v + 0.025, f"{v:.0%}", ha="center", fontsize=9, color=C_INT)
        axL.set_xticks(x)
        axL.set_xticklabels(labels)
        axL.set_ylim(0, 1.0)
        axL.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axL.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        axL.set_ylabel("miss rate on held-out distress cells")
        axL.set_title("Miss rate by prompt condition", fontsize=10)
        axL.legend(loc="upper left", frameon=False)

        axR.plot(margins, fpr, marker="o", color=C_INT, lw=1.6,
                 label="held-out false positives")
        axR.plot(margins, miss, marker="s", color=C_EXPR, lw=1.6, ls="--",
                 label="miss rate, suppressed distress")
        for mx, fy in zip(margins, fpr):
            axR.annotate(f"{fy:.0%}", (mx, fy), textcoords="offset points",
                         xytext=(9, 8), ha="left", fontsize=9, color=C_INT)
        axR.set_xlabel("threshold margin (separation units)")
        axR.set_ylabel("rate on held-out cells")
        axR.set_ylim(-0.03, 0.66)
        axR.set_yticks([0, 0.25, 0.5])
        axR.set_yticklabels(["0%", "25%", "50%"])
        axR.set_xticks(margins)
        axR.set_title("Internal trigger: operating curve", fontsize=10)
        axR.legend(loc="upper right", frameon=False)

        _watermark(fig, synthetic)
        out_pdf = Path(out_pdf)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf)
        plt.close(fig)
    return out_pdf


if __name__ == "__main__":  # CPU-only one-command figure rebuild from results/*.json
    import argparse
    ap = argparse.ArgumentParser(description="Rebuild figures/fig1_masking.pdf and "
                                 "figures/fig2_doseresponse.pdf from results/*.json (no model, no GPU).")
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    a = ap.parse_args()
    print(fig1_masking(a.results, Path(a.figures) / "fig1_masking.pdf"))
    print(fig2_monitor(a.results, Path(a.figures) / "fig2_monitor.pdf"))
    print(fig2_doseresponse(a.results, Path(a.figures) / "fig2_doseresponse.pdf"))

"""Pre-registered statistics: BCa cluster bootstrap, TOST, Holm, ICC, reliability.

Policy (PREREGISTRATION.md section 5 + analysis/DECISIONS.md D5, D7, D14):
  * unit of analysis = SCENARIO; every CI is a BCa cluster bootstrap over scenarios
    (B = 10,000 default, fixed seed), cluster count reported next to every interval;
  * cluster resampling is stratified by scenario class where the statistic needs
    both classes; resampled clusters are relabelled so a scenario drawn twice
    counts twice;
  * acceleration from leave-one-cluster-out jackknife.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sps

DEFAULT_B = 10_000


@dataclass
class BootResult:
    value: float
    ci_low: float
    ci_high: float
    n_clusters: int
    B: int
    n_boot_failed: int = 0
    boots: Optional[np.ndarray] = None  # kept only when keep_boots=True; not serialized

    def as_dict(self) -> dict:
        return {
            "value": _f(self.value), "ci_low": _f(self.ci_low), "ci_high": _f(self.ci_high),
            "n_clusters": int(self.n_clusters), "B": int(self.B),
            "n_boot_failed": int(self.n_boot_failed),
        }


def _f(x) -> Optional[float]:
    x = float(x)
    return x if math.isfinite(x) else None


def _bca_interval(theta: float, boots: np.ndarray, jack: np.ndarray,
                  alpha: float) -> Tuple[float, float]:
    """BCa endpoints from bootstrap replicates + jackknife values."""
    if len(boots) == 0 or not np.isfinite(theta):
        return (float("nan"), float("nan"))
    if np.allclose(boots, boots[0]):
        return (float(boots[0]), float(boots[0]))
    # bias correction; ties get half weight, proportion clipped away from 0/1
    prop = (np.sum(boots < theta) + 0.5 * np.sum(boots == theta)) / len(boots)
    prop = min(max(prop, 1.0 / (len(boots) + 1)), 1.0 - 1.0 / (len(boots) + 1))
    z0 = sps.norm.ppf(prop)
    jack = jack[np.isfinite(jack)]
    if len(jack) >= 2:
        jm = jack.mean()
        num = np.sum((jm - jack) ** 3)
        den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
        a = float(num / den) if den > 0 else 0.0
    else:
        a = 0.0
    out = []
    for q in (alpha / 2.0, 1.0 - alpha / 2.0):
        z = sps.norm.ppf(q)
        adj = sps.norm.cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z)))
        out.append(float(np.quantile(boots, adj)))
    return (min(out), max(out))


def _stratified_draw(rng: np.random.Generator, strata: Dict[str, List[int]]) -> np.ndarray:
    """Resample cluster indices with replacement within each stratum."""
    picks: List[np.ndarray] = []
    for idxs in strata.values():
        idxs = np.asarray(idxs)
        picks.append(rng.choice(idxs, size=len(idxs), replace=True))
    return np.concatenate(picks)


def _cluster_partition(df: pd.DataFrame, cluster_col: str,
                       strata_col: Optional[str]) -> Tuple[List[pd.DataFrame], Dict[str, List[int]]]:
    frames: List[pd.DataFrame] = []
    strata: Dict[str, List[int]] = {}
    for i, (label, g) in enumerate(df.groupby(cluster_col, sort=True)):
        frames.append(g)
        key = str(g[strata_col].iloc[0]) if strata_col else "_all"
        strata.setdefault(key, []).append(i)
    return frames, strata


def bca_cluster_bootstrap_frame(
    df: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    cluster_col: str = "scenario_id",
    strata_col: Optional[str] = None,
    B: int = DEFAULT_B,
    seed: int = 0,
    alpha: float = 0.05,
    keep_boots: bool = False,
) -> BootResult:
    """Generic BCa cluster bootstrap: stat_fn sees a resampled frame whose
    cluster_col is RELABELLED per draw (a cluster drawn twice counts twice)."""
    frames, strata = _cluster_partition(df, cluster_col, strata_col)
    n = len(frames)
    theta = float(stat_fn(df))
    if n < 2:
        return BootResult(theta, float("nan"), float("nan"), n, B)
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    failed = 0
    for b in range(B):
        picks = _stratified_draw(rng, strata)
        parts = []
        for j, k in enumerate(picks):
            g = frames[k].copy()
            g[cluster_col] = f"boot{j}"
            parts.append(g)
        try:
            boots[b] = float(stat_fn(pd.concat(parts, ignore_index=True)))
        except Exception:
            boots[b] = float("nan")
    finite = boots[np.isfinite(boots)]
    failed = B - len(finite)
    jack = np.empty(n)
    for k in range(n):
        try:
            jack[k] = float(stat_fn(pd.concat(
                [frames[j] for j in range(n) if j != k], ignore_index=True)))
        except Exception:
            jack[k] = float("nan")
    lo, hi = _bca_interval(theta, finite, jack, alpha)
    return BootResult(theta, lo, hi, n, B, failed,
                      boots=finite if keep_boots else None)


def bca_cluster_bootstrap_rows(
    df: pd.DataFrame,
    cols: Sequence[str],
    stat_fn: Callable[..., float],
    cluster_col: str = "scenario_id",
    strata_col: Optional[str] = None,
    B: int = DEFAULT_B,
    seed: int = 0,
    alpha: float = 0.05,
    keep_boots: bool = False,
) -> BootResult:
    """Fast path for row-level statistics that ignore cluster labels
    (Spearman, AUC, fractions): stat_fn(*column_arrays) -> float."""
    frames, strata = _cluster_partition(df, cluster_col, strata_col)
    n = len(frames)
    per = [tuple(g[c].to_numpy(dtype=float) for c in cols) for g in frames]
    full = tuple(np.concatenate([p[i] for p in per]) for i in range(len(cols)))
    theta = float(stat_fn(*full))
    if n < 2:
        return BootResult(theta, float("nan"), float("nan"), n, B)
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    for b in range(B):
        picks = _stratified_draw(rng, strata)
        arrs = tuple(np.concatenate([per[k][i] for k in picks]) for i in range(len(cols)))
        try:
            boots[b] = float(stat_fn(*arrs))
        except Exception:
            boots[b] = float("nan")
    finite = boots[np.isfinite(boots)]
    jack = np.empty(n)
    for k in range(n):
        arrs = tuple(np.concatenate([per[j][i] for j in range(n) if j != k])
                     for i in range(len(cols)))
        try:
            jack[k] = float(stat_fn(*arrs))
        except Exception:
            jack[k] = float("nan")
    lo, hi = _bca_interval(theta, finite, jack, alpha)
    return BootResult(theta, lo, hi, n, B, B - len(finite),
                      boots=finite if keep_boots else None)


def bca_cluster_values(
    values: Union[Sequence[float], np.ndarray],
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    B: int = DEFAULT_B,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootResult:
    """BCa bootstrap over precomputed per-cluster scalar values (one per scenario)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    theta = float(stat_fn(values)) if n else float("nan")
    if n < 2:
        return BootResult(theta, float("nan"), float("nan"), n, B)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boots = np.array([float(stat_fn(values[row])) for row in idx])
    finite = boots[np.isfinite(boots)]
    jack = np.array([float(stat_fn(np.delete(values, k))) for k in range(n)])
    lo, hi = _bca_interval(theta, finite, jack, alpha)
    return BootResult(theta, lo, hi, n, B, B - len(finite))


# ---------------------------------------------------------------- named statistics

def spearman_stat(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(sps.spearmanr(x, y).statistic)


def auc_stat(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC, labels in {0,1}; nan when a draw is single-class."""
    labels = labels.astype(int)
    n_pos, n_neg = int(labels.sum()), int(len(labels) - labels.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = sps.rankdata(scores)
    return float((order[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def spearman_cluster(df: pd.DataFrame, x: str, y: str, **kw) -> BootResult:
    return bca_cluster_bootstrap_rows(df, [x, y], spearman_stat, **kw)


def auc_cluster(df: pd.DataFrame, score_col: str, label_col: str,
                strata_col: str = "scenario_class", **kw) -> BootResult:
    return bca_cluster_bootstrap_rows(df, [score_col, label_col], auc_stat,
                                      strata_col=strata_col, **kw)


def boot_p_two_sided(boots: np.ndarray, null: float = 0.0) -> float:
    """Two-sided bootstrap p against a point null: 2 x min tail probability (D18)."""
    boots = np.asarray(boots, dtype=float)
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0:
        return float("nan")
    lo = (np.sum(boots <= null) + 1.0) / (len(boots) + 1.0)
    hi = (np.sum(boots >= null) + 1.0) / (len(boots) + 1.0)
    return float(min(1.0, 2.0 * min(lo, hi)))


# ---------------------------------------------------------------- TOST (D5)

def tost_equivalence(cluster_values: Union[Sequence[float], np.ndarray],
                     bound: float, alpha: float = 0.05) -> dict:
    """Two one-sided one-sample t tests: H0 |mean| >= bound vs H1 |mean| < bound.

    cluster_values: one value per scenario (the unit of analysis).
    Returns p_lower (mean > -bound), p_upper (mean < +bound), p_tost = max,
    equivalent = p_tost < alpha.
    """
    v = np.asarray(cluster_values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    out = {"bound": float(bound), "alpha": float(alpha), "n_clusters": int(n),
           "mean": _f(v.mean()) if n else None}
    if n < 3:
        out.update({"p_lower": None, "p_upper": None, "p_tost": None,
                    "equivalent": False, "note": "fewer than 3 clusters"})
        return out
    se = v.std(ddof=1) / math.sqrt(n)
    if se == 0:
        eq = abs(v.mean()) < bound
        out.update({"p_lower": 0.0 if eq else 1.0, "p_upper": 0.0 if eq else 1.0,
                    "p_tost": 0.0 if eq else 1.0, "equivalent": bool(eq),
                    "note": "zero variance across clusters"})
        return out
    df_ = n - 1
    t_lower = (v.mean() + bound) / se        # H0: mean <= -bound
    t_upper = (bound - v.mean()) / se        # H0: mean >= +bound
    p_lower = float(sps.t.sf(t_lower, df_))
    p_upper = float(sps.t.sf(t_upper, df_))
    p_tost = max(p_lower, p_upper)
    out.update({"se": _f(se), "p_lower": p_lower, "p_upper": p_upper,
                "p_tost": p_tost, "equivalent": bool(p_tost < alpha)})
    return out


# ---------------------------------------------------------------- Holm (D18)

def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm step-down adjusted p values (monotone, capped at 1)."""
    items = [(k, p) for k, p in pvals.items() if p is not None and math.isfinite(p)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    adj: Dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adj[k] = min(1.0, running)
    for k, p in pvals.items():
        if k not in adj:
            adj[k] = float("nan")
    return adj


# ---------------------------------------------------------------- ICC(2,1) (D14)

def icc_2_1(matrix: np.ndarray) -> float:
    """Shrout-Fleiss ICC(2,1): two-way random effects, absolute agreement, single
    measurement. matrix: n targets x k raters (seeds), complete cases only."""
    m = np.asarray(matrix, dtype=float)
    m = m[~np.isnan(m).any(axis=1)]
    n, k = m.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = m.mean()
    row_means = m.mean(axis=1)
    col_means = m.mean(axis=0)
    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_cols = n * np.sum((col_means - grand) ** 2)
    ss_total = np.sum((m - grand) ** 2)
    ss_err = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    if denom == 0:
        return float("nan")
    return float((msr - mse) / denom)


def icc_across_seeds(df: pd.DataFrame, channel: str,
                     target_cols: Sequence[str] = ("panel", "scenario_id", "condition",
                                                   "direction", "coefficient")) -> Tuple[float, int, int]:
    """ICC(2,1) with targets = unique cells, raters = seeds. Returns (icc, n_targets, k).
    Targets key on coefficient: dose_sd is a bookkeeping
    field that is 0.0 for the control directions at both A-prime rungs, so keying on it merged
    the 500 and 1000 rungs of R2/R3/SEM/OTHER into one target (426 targets instead of 522)."""
    pivot = df.pivot_table(index=list(target_cols), columns="seed",
                           values=channel, aggfunc="mean")
    pivot = pivot.dropna(axis=0, how="any")
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return float("nan"), int(pivot.shape[0]), int(pivot.shape[1])
    return icc_2_1(pivot.to_numpy()), int(pivot.shape[0]), int(pivot.shape[1])


# ---------------------------------------------------------------- paraphrase r (D14)

def _pairwise_mean_r(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    rs = []
    for x, y in ((a, b), (a, c), (b, c)):
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            continue
        rs.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(rs)) if rs else float("nan")


def paraphrase_parallel_forms(df: pd.DataFrame, B: int = DEFAULT_B, seed: int = 0) -> BootResult:
    """Mean pairwise Pearson r among q_self_logit, para1, para2 over cells,
    CI by cluster bootstrap over scenarios."""
    cols = ["q_self_logit", "q_self_logit_para1", "q_self_logit_para2"]
    sub = df.dropna(subset=cols)
    return bca_cluster_bootstrap_rows(sub, cols, _pairwise_mean_r, B=B, seed=seed)

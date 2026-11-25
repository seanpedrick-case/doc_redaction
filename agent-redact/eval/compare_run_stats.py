#!/usr/bin/env python3
"""Statistically compare eval runs recorded in ``eval_run_summary.csv``.

Reads the append-only summary CSV written by ``eval_run_summary_csv.py`` (one row
per run), groups rows by task (``--task-col``, default ``document_name``) and by
run configuration (``--config-cols``, default ``model_name,ocr_method`` — e.g.
compare models), then tests whether configurations differ significantly on each
metric.

Recommended tests (chosen for small-n, paired-by-task, bounded [0,1] scores
with ceiling effects such as many perfect 1.0s):

* **Continuous metrics** (default ``gold_geometry_f1/recall/precision`` and
  ``ocr_gold_f1/recall/precision``), two configs:
  - **paired permutation (randomization) test** on the mean per-task difference
    — primary; exact for <= 16 tasks, Monte Carlo otherwise; no distribution
    assumptions and handles ties naturally;
  - **Wilcoxon signed-rank** (exact for small n) and **sign test** (exact
    binomial) as alternatives robust to magnitude outliers;
  - **bootstrap percentile CI** of the mean difference (resampling tasks) plus
    a probability-of-superiority effect size.
* **Continuous metrics, k >= 3 configs**: **Friedman test** per metric with
  post-hoc pairwise tests (all p-values FDR-adjusted within the metric).
* **Binary metrics** (default ``overall_pass``, ``policy_strict_pass``):
  **McNemar's exact test** on discordant task pairs (or a paired permutation
  test on pass-rate differences when a task/config has repeated runs);
  **Cochran's Q** for k >= 3 configs.
* Each of the six metrics is a separate test family; primary p-values are
  adjusted with **Benjamini–Hochberg FDR** within each metric
  (``--fdr-scope metric``, default) or across all tests (``--fdr-scope all``).

Design notes:

* Runs of the same (task, config) are aggregated first (mean for continuous
  metrics, pass-rate for binary) so each task contributes **one paired
  observation per config** — this avoids pseudoreplication from repeated runs.
* Rows with a truthy ``<metric>_skipped`` column or an unparseable score are
  dropped per metric; paired tests use only **complete task pairs** (tasks
  observed under both configs) and report ``n_pairs``.
* scipy is used when available (Wilcoxon / Friedman / binomial exact); exact
  stdlib fallbacks (sign test, permutation test, McNemar, chi-square tail via
  incomplete gamma) keep the script working without scipy.

Example::

    python agent-redact/eval/compare_run_stats.py \\
      --csv agent-redact/eval/outputs/eval_run_summary.csv \\
      --config-cols model_name,ocr_method \\
      --out agent-redact/eval/outputs/run_stats_comparison

Writes ``<out>.csv`` and ``<out>.json`` and prints a console table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

try:  # optional — exact stdlib fallbacks below are used when unavailable
    from scipy import stats as _sp
except Exception:  # pragma: no cover
    _sp = None

_EVAL_DIR = Path(__file__).resolve().parent

DEFAULT_TASK_COL = "document_name"
DEFAULT_CONFIG_COLS = ("model_name", "ocr_method")
DEFAULT_METRICS = (
    "gold_geometry_f1",
    "gold_geometry_recall",
    "gold_geometry_precision",
    "ocr_gold_f1",
    "ocr_gold_recall",
    "ocr_gold_precision",
)
DEFAULT_BINARY = ("overall_pass", "policy_strict_pass")

#: Exhaustive sign-flip enumeration is used up to this many pairs (2^n masks).
EXHAUSTIVE_PERM_MAX_N = 16

OUTPUT_COLUMNS = (
    "metric",
    "kind",
    "scope",
    "config_a",
    "config_b",
    "n_pairs",
    "mean_a",
    "mean_b",
    "mean_diff",
    "effect_prob_superiority",
    "ci_low",
    "ci_high",
    "p_permutation",
    "p_wilcoxon",
    "p_sign",
    "p_mcnemar",
    "discordant_b",
    "discordant_c",
    "omnibus_p",
    "primary_test",
    "primary_p",
    "q_value",
    "significant_at_alpha",
    "method",
)


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _parse_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _is_truthy(raw: str | None) -> bool:
    return bool(raw) and raw.strip().lower() in {"1", "true", "yes", "y", "t", "pass"}


def _parse_binary(raw: str | None) -> bool | None:
    if raw is None:
        return None
    text = raw.strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "t", "pass"}:
        return True
    if text in {"0", "false", "no", "n", "f", "fail"}:
        return False
    return None


def _config_label(row: dict[str, str], config_cols: Sequence[str]) -> str:
    return " | ".join((row.get(c) or "").strip() or "-" for c in config_cols)


def default_csv_path() -> Path:
    custom = (os.environ.get("AGENT_EVAL_RUN_SUMMARY_CSV_PATH") or "").strip()
    if custom:
        path = Path(custom)
        if not path.is_absolute():
            path = (_EVAL_DIR.parent.parent / custom).resolve()
        return path
    return (_EVAL_DIR / "outputs" / "eval_run_summary.csv").resolve()


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------- #
# Aggregation: rows -> table[task][config]
# --------------------------------------------------------------------------- #
def build_task_config_table(
    rows: list[dict[str, str]],
    *,
    task_col: str,
    config_cols: Sequence[str],
    metrics: Sequence[str],
    binary_cols: Sequence[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Aggregate runs into ``table[task][config]`` with per-metric value lists."""
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        task = (row.get(task_col) or "").strip() or "(unknown)"
        config = _config_label(row, config_cols)
        cell = table.setdefault(task, {}).setdefault(
            config,
            {
                "n_runs": 0,
                "metrics": {m: [] for m in metrics},
                "binary": {b: [] for b in binary_cols},
            },
        )
        cell["n_runs"] += 1
        for metric in metrics:
            if _is_truthy(row.get(f"{metric}_skipped")):
                continue
            value = _parse_float(row.get(metric))
            if value is not None:
                cell["metrics"][metric].append(value)
        for col in binary_cols:
            if _is_truthy(row.get(f"{col}_skipped")):
                continue
            value = _parse_binary(row.get(col))
            if value is not None:
                cell["binary"][col].append(value)
    return table


def _aggregate(values: Sequence[float], how: str) -> float | None:
    if not values:
        return None
    if how == "median":
        ordered = sorted(values)
        n = len(ordered)
        mid = n // 2
        return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    return sum(values) / len(values)


def _config_value(
    cell: dict[str, Any] | None, kind: str, key: str, how: str
) -> float | None:
    if not cell:
        return None
    raw = cell[kind].get(key) or []
    return _aggregate([float(v) for v in raw], how)


def paired_series(
    table: dict[str, dict[str, dict[str, Any]]],
    *,
    kind: str,
    key: str,
    config_a: str,
    config_b: str,
    how: str,
) -> tuple[list[str], list[float], list[float]]:
    """Complete pairs (task, value_a, value_b) across the two configs."""
    tasks: list[str] = []
    va: list[float] = []
    vb: list[float] = []
    for task in sorted(table):
        xa = _config_value(table[task].get(config_a), kind, key, how)
        xb = _config_value(table[task].get(config_b), kind, key, how)
        if xa is None or xb is None:
            continue
        tasks.append(task)
        va.append(xa)
        vb.append(xb)
    return tasks, va, vb


# --------------------------------------------------------------------------- #
# Statistical tests (stdlib-exact; scipy when available)
# --------------------------------------------------------------------------- #
def sign_test_p(diffs: Sequence[float]) -> float:
    """Two-sided exact sign test on non-zero differences (X ~ Bin(n, 0.5))."""
    nonzero = [d for d in diffs if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    k = sum(1 for d in nonzero if d > 0)
    pmf = [math.comb(n, i) / (2.0**n) for i in range(n + 1)]
    p_le = sum(pmf[: k + 1])
    p_ge = sum(pmf[k:])
    return min(1.0, 2.0 * min(p_le, p_ge))


def permutation_test_p(
    diffs: Sequence[float],
    *,
    n_permutations: int,
    seed: int,
    exhaustive_max_n: int = EXHAUSTIVE_PERM_MAX_N,
) -> tuple[float, str]:
    """Paired permutation test on the mean difference (sign-flip randomization).

    Exact enumeration for n <= ``exhaustive_max_n``; otherwise Monte Carlo with
    the Phipson–Smyth +1 correction.
    """
    n = len(diffs)
    if n == 0:
        return 1.0, "na"
    obs = sum(diffs)
    # Tolerance so sums that are mathematically equal to |obs| (e.g. the all-plus
    # mask) are not lost to float summation-order rounding.
    tol = 1e-9 * max(1.0, abs(obs))
    if n <= exhaustive_max_n:
        total = 1 << n
        count = 0
        for mask in range(total):
            s = 0.0
            for i in range(n):
                s += diffs[i] if (mask >> i) & 1 else -diffs[i]
            if abs(s) >= abs(obs) - tol:
                count += 1
        return count / total, "exact"
    rng = random.Random(seed)
    count = 0
    for _ in range(n_permutations):
        s = 0.0
        for d in diffs:
            s += d if rng.random() < 0.5 else -d
        if abs(s) >= abs(obs) - tol:
            count += 1
    return (count + 1) / (n_permutations + 1), "monte_carlo"


def bootstrap_mean_diff_ci(
    diffs: Sequence[float],
    *,
    n_boot: int,
    seed: int,
    alpha: float,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean paired difference (resample tasks)."""
    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[max(0, int((alpha / 2.0) * n_boot) - 1)]
    hi = means[min(n_boot - 1, int((1.0 - alpha / 2.0) * n_boot))]
    return lo, hi


def wilcoxon_signed_rank_p(diffs: Sequence[float]) -> tuple[float, str]:
    """Two-sided Wilcoxon signed-rank p-value (zeros dropped)."""
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    if n == 0:
        return 1.0, "na"
    if _sp is not None:
        try:
            res = _sp.wilcoxon(
                nz,
                zero_method="wilcox",
                alternative="two-sided",
                method="exact" if n <= 25 else "approx",
            )
            return float(res.pvalue), "scipy"
        except Exception:
            pass
    # Exact randomization distribution of W+ via DP over doubled ranks (ties -> .5).
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    sorted_abs = [abs(nz[i]) for i in order]
    doubled = [0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        avg_doubled = int(round((i + j + 2)))  # 2 * average rank, integral
        for m in range(i, j + 1):
            doubled[order[m]] = avg_doubled
        i = j + 1
    dist = {0: 1}
    for r in doubled:
        nxt = dict(dist)
        for s, c in dist.items():
            nxt[s + r] = nxt.get(s + r, 0) + c
        dist = nxt
    total = sum(doubled)
    w_obs = sum(r for r, d in zip(doubled, nz) if d > 0)
    mu = total / 2.0
    count = sum(c for s, c in dist.items() if abs(s - mu) >= abs(w_obs - mu))
    return count / (2.0**n), "exact-dp"


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant pair counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    if _sp is not None:
        try:
            return float(_sp.binomtest(min(b, c), n, 0.5).pvalue)
        except Exception:
            pass
    k = min(b, c)
    pmf = [math.comb(n, i) / (2.0**n) for i in range(n + 1)]
    threshold = pmf[k]
    return min(1.0, sum(p for p in pmf if p <= threshold + 1e-12))


def _chi2_sf(x: float, df: int) -> float:
    """Upper tail of the chi-square distribution (scipy, else incomplete gamma)."""
    if x <= 0:
        return 1.0
    if _sp is not None:
        return float(_sp.chi2.sf(x, df))
    a = df / 2.0
    z = x / 2.0
    if z < a + 1.0:  # series for P(a, z); Q = 1 - P
        term = 1.0 / a
        total = term
        ap = a
        for _ in range(1000):
            ap += 1.0
            term *= z / ap
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        p = total * math.exp(-z + a * math.log(z) - math.lgamma(a))
        return min(1.0, max(0.0, 1.0 - p))
    # Lentz continued fraction for Q(a, z)
    tiny = 1e-300
    b = z + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    q = math.exp(-z + a * math.log(z) - math.lgamma(a)) * h
    return min(1.0, max(0.0, q))


def _rank_block(values: Sequence[float]) -> list[float]:
    """Average ranks (1-based) within one block, ties handled."""
    k = len(values)
    order = sorted(range(k), key=lambda j: values[j])
    ranks = [0.0] * k
    j = 0
    while j < k:
        l_num = j
        while l_num + 1 < k and values[order[l_num + 1]] == values[order[j]]:
            l_num += 1
        avg = (j + l_num) / 2.0 + 1.0
        for m in range(j, l_num + 1):
            ranks[order[m]] = avg
        j = l_num + 1
    return ranks


def friedman_test(columns: Sequence[Sequence[float]]) -> tuple[float | None, str]:
    """Friedman repeated-measures test across k >= 3 configs (complete blocks)."""
    k = len(columns)
    n = len(columns[0]) if columns else 0
    if k < 3 or n < 2:
        return None, "na"
    if _sp is not None:
        try:
            res = _sp.friedmanchisquare(*columns)
            return float(res.pvalue), "scipy"
        except Exception:
            pass
    rank_sums = [0.0] * k
    for i in range(n):
        ranks = _rank_block([col[i] for col in columns])
        for j in range(k):
            rank_sums[j] += ranks[j]
    q = 12.0 / (n * k * (k + 1)) * sum(r * r for r in rank_sums) - 3.0 * n * (k + 1)
    return _chi2_sf(q, k - 1), "chi2-approx"


def cochran_q_test(columns: Sequence[Sequence[float]]) -> tuple[float | None, str]:
    """Cochran's Q for k >= 3 binary (0/1 or rate) columns over the same tasks."""
    k = len(columns)
    n = len(columns[0]) if columns else 0
    if k < 3 or n < 2:
        return None, "na"
    col_sums = [sum(col) for col in columns]
    row_sums = [sum(columns[j][i] for j in range(k)) for i in range(n)]
    total = sum(col_sums)
    if total == 0 or total == n * k:
        return 1.0, "degenerate"
    denominator = k * total - sum(l_num * l_num for l_num in row_sums)
    if denominator == 0:
        return 1.0, "degenerate"
    q = (k - 1) * (k * sum(g * g for g in col_sums) - total * total) / denominator
    return _chi2_sf(q, k - 1), "chi2-approx"


def prob_superiority(diffs: Sequence[float]) -> float | None:
    """P(diff > 0) + 0.5 * P(diff == 0) — common-language effect size."""
    n = len(diffs)
    if n == 0:
        return None
    wins = sum(1 for d in diffs if d > 0.0)
    ties = sum(1 for d in diffs if d == 0.0)
    return (wins + 0.5 * ties) / n


# --------------------------------------------------------------------------- #
# Multiple-testing adjustment
# --------------------------------------------------------------------------- #
def bh_adjust(pvals: Sequence[float | None]) -> list[float | None]:
    """Benjamini–Hochberg FDR-adjusted q-values (None stays None)."""
    m = len(pvals)
    order = sorted(
        range(m),
        key=lambda i: (pvals[i] is None, pvals[i] if pvals[i] is not None else 0.0),
    )
    adjusted: list[float | None] = [None] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        p = pvals[i]
        if p is None:
            continue
        q = min(1.0, prev, p * m / rank)
        adjusted[i] = q
        prev = q
    return adjusted


# --------------------------------------------------------------------------- #
# Comparison driver
# --------------------------------------------------------------------------- #
def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compare_metric(
    table: dict[str, dict[str, dict[str, Any]]],
    *,
    metric: str,
    kind: str,
    configs: Sequence[str],
    aggregate: str,
    min_pairs: int,
    n_permutations: int,
    n_boot: int,
    seed: int,
    alpha: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configs = sorted(configs)

    # Omnibus test across all k configs (complete blocks only).
    if len(configs) >= 3:
        blocks: list[list[float]] = []
        for task in sorted(table):
            vals: list[float] = []
            for config in configs:
                v = _config_value(table[task].get(config), kind, metric, aggregate)
                if v is None:
                    vals = []
                    break
                vals.append(v)
            if vals:
                blocks.append(vals)
        if len(blocks) >= min_pairs:
            columns = [[block[j] for block in blocks] for j in range(len(configs))]
            if kind == "metrics":
                p, method = friedman_test(columns)
                test_name = "friedman"
            else:
                p, method = cochran_q_test(columns)
                test_name = "cochran_q"
            if p is not None:
                rows.append(
                    {
                        "metric": metric,
                        "kind": kind,
                        "scope": "omnibus",
                        "config_a": "ALL_CONFIGS",
                        "config_b": "",
                        "n_pairs": len(blocks),
                        "mean_a": None,
                        "mean_b": None,
                        "mean_diff": None,
                        "effect_prob_superiority": None,
                        "ci_low": None,
                        "ci_high": None,
                        "p_permutation": None,
                        "p_wilcoxon": None,
                        "p_sign": None,
                        "p_mcnemar": None,
                        "discordant_b": None,
                        "discordant_c": None,
                        "omnibus_p": p,
                        "primary_test": test_name,
                        "primary_p": p,
                        "q_value": None,
                        "significant_at_alpha": None,
                        "method": method,
                    }
                )

    # Pairwise comparisons.
    for i, config_a in enumerate(configs):
        for config_b in configs[i + 1 :]:
            tasks, va, vb = paired_series(
                table,
                kind=kind,
                key=metric,
                config_a=config_a,
                config_b=config_b,
                how=aggregate,
            )
            if len(tasks) < min_pairs:
                continue
            diffs = [x - y for x, y in zip(va, vb)]
            row: dict[str, Any] = {
                "metric": metric,
                "kind": kind,
                "scope": "pairwise",
                "config_a": config_a,
                "config_b": config_b,
                "n_pairs": len(tasks),
                "mean_a": _mean(va),
                "mean_b": _mean(vb),
                "mean_diff": _mean(diffs),
                "effect_prob_superiority": prob_superiority(diffs),
                "ci_low": None,
                "ci_high": None,
                "p_permutation": None,
                "p_wilcoxon": None,
                "p_sign": None,
                "p_mcnemar": None,
                "discordant_b": None,
                "discordant_c": None,
                "omnibus_p": None,
                "primary_test": "",
                "primary_p": None,
                "q_value": None,
                "significant_at_alpha": None,
                "method": "",
            }
            if kind == "metrics":
                p_perm, perm_method = permutation_test_p(
                    diffs, n_permutations=n_permutations, seed=seed
                )
                p_w, w_method = wilcoxon_signed_rank_p(diffs)
                lo, hi = bootstrap_mean_diff_ci(
                    diffs, n_boot=n_boot, seed=seed, alpha=alpha
                )
                row.update(
                    p_permutation=p_perm,
                    p_wilcoxon=p_w,
                    p_sign=sign_test_p(diffs),
                    ci_low=lo,
                    ci_high=hi,
                    primary_test="paired_permutation",
                    primary_p=p_perm,
                    method=f"perm={perm_method};wilcoxon={w_method}",
                )
            else:
                row["p_sign"] = sign_test_p(diffs)
                if all(v in (0.0, 1.0) for v in va + vb):
                    b = sum(1 for x, y in zip(va, vb) if x == 1.0 and y == 0.0)
                    c = sum(1 for x, y in zip(va, vb) if x == 0.0 and y == 1.0)
                    p_m = mcnemar_exact_p(b, c)
                    row.update(
                        p_mcnemar=p_m,
                        discordant_b=b,
                        discordant_c=c,
                        primary_test="mcnemar_exact",
                        primary_p=p_m,
                        method="exact_binomial",
                    )
                else:
                    p_perm, perm_method = permutation_test_p(
                        diffs, n_permutations=n_permutations, seed=seed
                    )
                    lo, hi = bootstrap_mean_diff_ci(
                        diffs, n_boot=n_boot, seed=seed, alpha=alpha
                    )
                    row.update(
                        p_permutation=p_perm,
                        ci_low=lo,
                        ci_high=hi,
                        primary_test="paired_permutation_on_pass_rate",
                        primary_p=p_perm,
                        method=f"perm={perm_method}",
                    )
            rows.append(row)
    return rows


def run_comparison(
    rows: list[dict[str, str]],
    *,
    task_col: str,
    config_cols: Sequence[str],
    metrics: Sequence[str],
    binary_cols: Sequence[str],
    configs_filter: Sequence[str] | None,
    aggregate: str,
    min_pairs: int,
    n_permutations: int,
    n_boot: int,
    seed: int,
    alpha: float,
    fdr_scope: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table = build_task_config_table(
        rows,
        task_col=task_col,
        config_cols=config_cols,
        metrics=metrics,
        binary_cols=binary_cols,
    )
    all_configs = sorted({config for cells in table.values() for config in cells})
    if configs_filter:
        wanted = {c for c in configs_filter}
        missing = wanted - set(all_configs)
        all_configs = [c for c in all_configs if c in wanted]
        if missing:
            raise SystemExit(
                f"--configs not found in CSV: {sorted(missing)}\nAvailable: {all_configs}"
            )

    out_rows: list[dict[str, Any]] = []
    for metric in metrics:
        out_rows.extend(
            compare_metric(
                table,
                metric=metric,
                kind="metrics",
                configs=all_configs,
                aggregate=aggregate,
                min_pairs=min_pairs,
                n_permutations=n_permutations,
                n_boot=n_boot,
                seed=seed,
                alpha=alpha,
            )
        )
    for col in binary_cols:
        out_rows.extend(
            compare_metric(
                table,
                metric=col,
                kind="binary",
                configs=all_configs,
                aggregate=aggregate,
                min_pairs=min_pairs,
                n_permutations=n_permutations,
                n_boot=n_boot,
                seed=seed,
                alpha=alpha,
            )
        )

    # FDR adjustment on primary p-values.
    if fdr_scope == "all":
        groups: dict[str, list[int]] = {"__all__": list(range(len(out_rows)))}
    else:
        groups = {}
        for idx, row in enumerate(out_rows):
            groups.setdefault(row["metric"], []).append(idx)
    for indices in groups.values():
        pvals = [out_rows[i]["primary_p"] for i in indices]
        qs = bh_adjust(pvals)
        for i, q in zip(indices, qs):
            out_rows[i]["q_value"] = q
            out_rows[i]["significant_at_alpha"] = (
                bool(q is not None and q <= alpha)
                if out_rows[i]["primary_p"] is not None
                else None
            )

    meta = {
        "task_col": task_col,
        "config_cols": list(config_cols),
        "metrics": list(metrics),
        "binary_cols": list(binary_cols),
        "configs": all_configs,
        "n_tasks": len(table),
        "n_rows": len(rows),
        "aggregate": aggregate,
        "min_pairs": min_pairs,
        "alpha": alpha,
        "fdr_scope": fdr_scope,
        "n_permutations": n_permutations,
        "n_boot": n_boot,
        "seed": seed,
        "scipy_available": _sp is not None,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return out_rows, meta


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}g}"
    return str(value)


def print_table(out_rows: list[dict[str, Any]], alpha: float) -> None:
    header = (
        f"{'metric':28s} {'scope':8s} {'config_a':24s} {'config_b':24s} "
        f"{'n':>3s} {'mean_diff':>10s} {'p_perm':>8s} {'p_wilc':>8s} {'p_sign':>8s} "
        f"{'p_mcnem':>8s} {'q':>8s} {'sig':>4s}"
    )
    print(header)
    print("-" * len(header))
    for row in out_rows:
        sig = ""
        if row["significant_at_alpha"] is True:
            sig = f"p<{alpha}"
        elif row["significant_at_alpha"] is False:
            sig = "ns"
        print(
            f"{row['metric'][:28]:28s} {row['scope'][:8]:8s} "
            f"{row['config_a'][:24]:24s} {row['config_b'][:24]:24s} "
            f"{row['n_pairs']:>3d} {_fmt(row['mean_diff'], 4):>10s} "
            f"{_fmt(row['p_permutation']):>8s} {_fmt(row['p_wilcoxon']):>8s} "
            f"{_fmt(row['p_sign']):>8s} {_fmt(row['p_mcnemar']):>8s} "
            f"{_fmt(row['q_value']):>8s} {sig:>4s}"
        )


def write_outputs(
    out_rows: list[dict[str, Any]],
    meta: dict[str, Any],
    out_prefix: Path,
) -> tuple[Path, Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    json_path = out_prefix.with_suffix(".json")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k, "") for k in OUTPUT_COLUMNS})
    json_path.write_text(
        json.dumps({"meta": meta, "results": out_rows}, indent=2), encoding="utf-8"
    )
    return csv_path, json_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _split_csv_arg(raw: str | None, default: Sequence[str]) -> list[str]:
    if not raw:
        return list(default)
    return [part.strip() for part in raw.split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to eval_run_summary.csv (default: $AGENT_EVAL_RUN_SUMMARY_CSV_PATH or "
        "agent-redact/eval/outputs/eval_run_summary.csv).",
    )
    p.add_argument(
        "--task-col",
        default=DEFAULT_TASK_COL,
        help="Column identifying the task/gold case.",
    )
    p.add_argument(
        "--config-cols",
        default=",".join(DEFAULT_CONFIG_COLS),
        help="Comma-separated columns defining a compared configuration (default: model_name,ocr_method).",
    )
    p.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated continuous metric columns (default: the six gold geometry + OCR gold scores).",
    )
    p.add_argument(
        "--binary",
        default=",".join(DEFAULT_BINARY),
        help="Comma-separated binary pass columns (default: overall_pass,policy_strict_pass).",
    )
    p.add_argument(
        "--configs",
        default=None,
        help="Comma-separated config labels to restrict the comparison to (default: all).",
    )
    p.add_argument(
        "--aggregate",
        choices=("mean", "median"),
        default="mean",
        help="Aggregate repeated runs of the same (task, config) before pairing.",
    )
    p.add_argument(
        "--min-pairs",
        type=int,
        default=3,
        help="Minimum complete task pairs to run a test.",
    )
    p.add_argument(
        "--alpha", type=float, default=0.05, help="Significance level for q-values."
    )
    p.add_argument(
        "--fdr-scope",
        choices=("metric", "all"),
        default="metric",
        help="Benjamini-Hochberg FDR scope: within each metric, or across all tests.",
    )
    p.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Monte Carlo permutations when n > 16 pairs.",
    )
    p.add_argument(
        "--n-boot",
        type=int,
        default=10000,
        help="Bootstrap resamples for the CI of the mean diff.",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="RNG seed for Monte Carlo / bootstrap."
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output prefix; writes <out>.csv and <out>.json "
        "(default: agent-redact/eval/outputs/run_stats_comparison).",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Print the table only; do not write CSV/JSON.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    csv_path = (args.csv or default_csv_path()).resolve()
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 2
    config_cols = _split_csv_arg(args.config_cols, DEFAULT_CONFIG_COLS)
    metrics = _split_csv_arg(args.metrics, DEFAULT_METRICS)
    binary_cols = _split_csv_arg(args.binary, DEFAULT_BINARY)
    configs_filter = _split_csv_arg(args.configs, ()) or None

    rows = load_rows(csv_path)
    out_rows, meta = run_comparison(
        rows,
        task_col=args.task_col,
        config_cols=config_cols,
        metrics=metrics,
        binary_cols=binary_cols,
        configs_filter=configs_filter,
        aggregate=args.aggregate,
        min_pairs=args.min_pairs,
        n_permutations=args.n_permutations,
        n_boot=args.n_boot,
        seed=args.seed,
        alpha=args.alpha,
        fdr_scope=args.fdr_scope,
    )
    meta["csv"] = str(csv_path)
    print(
        f"Loaded {meta['n_rows']} rows, {meta['n_tasks']} tasks, configs: {meta['configs']}"
    )
    print(
        f"scipy: {'available' if meta['scipy_available'] else 'not available (stdlib exact fallbacks)'}"
    )
    print()
    if out_rows:
        print_table(out_rows, args.alpha)
    else:
        print("No comparisons met --min-pairs.")
    if not args.no_write:
        out_prefix = (
            args.out or (_EVAL_DIR / "outputs" / "run_stats_comparison")
        ).resolve()
        csv_out, json_out = write_outputs(out_rows, meta, out_prefix)
        print()
        print(f"Wrote {csv_out}")
        print(f"Wrote {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

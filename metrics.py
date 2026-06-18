"""
metrics.py
----------
One function per metric. Each function takes a 1-D array-like of daily returns
(floats, normalised to a common notional) and returns a scalar.

Reduction metrics take (strategy_returns, benchmark_returns) and return a
scalar whose sign convention is: positive = strategy is better.

Sign conventions throughout:
  - CVaR and VaR are returned as POSITIVE numbers representing expected loss.
  - Reduction = benchmark_metric - strategy_metric  (positive = less risk).
  - Drawdown is returned as a POSITIVE number (magnitude of peak-to-trough).
  - Sortino, Sharpe, Calmar: higher is better.
  - Participation betas: interpreted by the caller; sign follows the regression.

All annualisation uses ANN = 252 trading days.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

ANN = 252          # trading days per year
MIN_OBS = 20       # minimum observations; functions return np.nan below this


def _check(r):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    return r


# ── Annualised mean and volatility ──────────────────────────────────────────

def annualised_mean(r) -> float:
    """Annualised arithmetic mean return."""
    r = _check(r)
    if len(r) < MIN_OBS:
        return np.nan
    return r.mean() * ANN


def annualised_vol(r) -> float:
    """Annualised sample standard deviation (ddof=1)."""
    r = _check(r)
    if len(r) < MIN_OBS:
        return np.nan
    return r.std(ddof=1) * np.sqrt(ANN)


# ── Sharpe and Sortino ───────────────────────────────────────────────────────

def sharpe(r, rf_daily: float = 0.0) -> float:
    """Annualised Sharpe ratio (excess return / total vol)."""
    r = _check(r)
    if len(r) < MIN_OBS:
        return np.nan
    excess = r - rf_daily
    if excess.std(ddof=1) == 0:
        return np.nan
    return (excess.mean() / excess.std(ddof=1)) * np.sqrt(ANN)


def sortino(r, mar_daily: float = 0.0) -> float:
    """
    Annualised Sortino ratio (Sortino & Price 1994).
    Downside deviation divides by the FULL sample size N, not the number of
    downside observations, so days above MAR contribute zero to the sum and
    the denominator is consistent across strategies compared on the same slice.

        DD = sqrt( (1/N) * sum( min(r_t - MAR, 0)^2 ) )
        Sortino = (mean(r) - MAR) / DD * sqrt(ANN)

    MAR defaults to zero (absolute return objective).
    """
    r = _check(r)
    if len(r) < MIN_OBS:
        return np.nan
    dd = np.sqrt(np.sum(np.minimum(r - mar_daily, 0.0) ** 2) / len(r))
    if dd == 0:
        return np.nan
    return (r.mean() - mar_daily) / dd * np.sqrt(ANN)


# ── VaR and CVaR ────────────────────────────────────────────────────────────

def var(r, level: float = 0.95) -> float:
    """
    Historical VaR at the given confidence level.
    Returns a POSITIVE number: the loss exceeded with probability (1-level).
    E.g. var(r, 0.95) is the loss at the 5th percentile.
    """
    r = _check(r)
    if len(r) < MIN_OBS:
        return np.nan
    return float(-np.percentile(r, (1 - level) * 100))


def cvar(r, level: float = 0.95) -> float:
    """
    Historical (expected-shortfall) CVaR at the given confidence level.
    Returns a POSITIVE number: expected loss in the worst (1-level) fraction.
    """
    r = _check(r)
    if len(r) < MIN_OBS:
        return np.nan
    threshold = np.percentile(r, (1 - level) * 100)
    tail = r[r <= threshold]
    if len(tail) == 0:
        return np.nan
    return float(-tail.mean())


def var_cvar_gap(r, level: float = 0.95) -> float:
    """
    Difference CVaR - VaR at the same level (positive = heavier tail beyond VaR).
    """
    return cvar(r, level) - var(r, level)


# ── Reduction metrics ────────────────────────────────────────────────────────

def cvar_reduction(r_strategy, r_benchmark, level: float = 0.95) -> float:
    """
    CVaR reduction vs benchmark.
    Positive = strategy has LESS tail loss than benchmark.
    """
    return cvar(r_benchmark, level) - cvar(r_strategy, level)


def var_reduction(r_strategy, r_benchmark, level: float = 0.95) -> float:
    """
    VaR reduction vs benchmark.
    Positive = strategy has LESS VaR than benchmark.
    """
    return var(r_benchmark, level) - var(r_strategy, level)


def variance_reduction(r_strategy, r_benchmark) -> float:
    """
    Ederington variance-reduction ratio.
    1 - Var(strategy) / Var(benchmark).
    Positive and near 1 = strong hedge.
    """
    r_s, r_b = _check(r_strategy), _check(r_benchmark)
    if len(r_s) < MIN_OBS or len(r_b) < MIN_OBS:
        return np.nan
    var_b = r_b.var(ddof=1)
    if var_b == 0:
        return np.nan
    return 1.0 - r_s.var(ddof=1) / var_b


# ── Drawdown metrics ─────────────────────────────────────────────────────────

def max_drawdown(r) -> float:
    """
    Maximum peak-to-trough drawdown on the cumulative return path.
    Returns a POSITIVE number.
    """
    r = _check(r)
    if len(r) < MIN_OBS:
        return np.nan
    cum = np.cumsum(r)                      # arithmetic cumulative P&L
    running_max = np.maximum.accumulate(cum)
    drawdowns = running_max - cum           # always >= 0
    return float(drawdowns.max())


def max_drawdown_reduction(r_strategy, r_benchmark) -> float:
    """
    Drawdown reduction vs benchmark.
    Positive = strategy has a smaller maximum drawdown.
    """
    return max_drawdown(r_benchmark) - max_drawdown(r_strategy)


def calmar(r) -> float:
    """
    Calmar ratio = annualised mean return / maximum drawdown.
    Higher is better.
    """
    r = _check(r)
    mdd = max_drawdown(r)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return annualised_mean(r) / mdd


# ── Distributional metrics ───────────────────────────────────────────────────

def skewness(r) -> float:
    """Sample skewness (Fisher, bias-corrected)."""
    r = _check(r)
    if len(r) < MIN_OBS:
        return np.nan
    return float(stats.skew(r, bias=False))


def excess_kurtosis(r) -> float:
    """Excess kurtosis (Fisher definition; normal = 0)."""
    r = _check(r)
    if len(r) < MIN_OBS:
        return np.nan
    return float(stats.kurtosis(r, bias=False, fisher=True))


# ── Upside / downside participation ─────────────────────────────────────────

def participation(r_strategy, r_benchmark) -> dict:
    """
    Henriksson-Merton (1981) conditional participation regression with
    Newey-West HAC standard errors (automatic lag selection).

    Model:
        r_strategy = alpha
                   + beta_up   * r_benchmark * I(r_benchmark > 0)
                   + beta_down * r_benchmark * I(r_benchmark <= 0)
                   + epsilon

    Returns a dict with keys:
        alpha, beta_up, beta_down,
        se_alpha, se_beta_up, se_beta_down,
        t_alpha, t_beta_up, t_beta_down,
        p_alpha, p_beta_up, p_beta_down,
        n_obs, n_up, n_down
    """
    r_s = _check(r_strategy)
    r_b = _check(r_benchmark)
    # Align on length (caller is responsible for aligning dates; here we
    # take the shorter of the two after NaN removal for robustness).
    n = min(len(r_s), len(r_b))
    if n < MIN_OBS:
        return {k: np.nan for k in (
            "alpha","beta_up","beta_down",
            "se_alpha","se_beta_up","se_beta_down",
            "t_alpha","t_beta_up","t_beta_down",
            "p_alpha","p_beta_up","p_beta_down",
            "n_obs","n_up","n_down")}

    r_s, r_b = r_s[:n], r_b[:n]
    up   = (r_b > 0).astype(float)
    down = (r_b <= 0).astype(float)

    X = np.column_stack([np.ones(n), r_b * up, r_b * down])
    y = r_s

    # Automatic Newey-West lag: floor(4*(n/100)^(2/9)) is the standard rule
    nw_lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    model   = sm.OLS(y, X).fit(cov_type="HAC",
                                cov_kwds={"maxlags": nw_lags, "use_correction": True})
    params  = model.params
    bse     = model.bse
    tvals   = model.tvalues
    pvals   = model.pvalues

    return dict(
        alpha=params[0],       beta_up=params[1],       beta_down=params[2],
        se_alpha=bse[0],       se_beta_up=bse[1],       se_beta_down=bse[2],
        t_alpha=tvals[0],      t_beta_up=tvals[1],      t_beta_down=tvals[2],
        p_alpha=pvals[0],      p_beta_up=pvals[1],      p_beta_down=pvals[2],
        n_obs=n,
        n_up=int(up.sum()),
        n_down=int(down.sum()),
    )


def upside_participation(r_strategy, r_benchmark) -> float:
    """
    Scalar upside participation beta (beta_up from the HM regression).
    Primary metric; higher = more upside retained.
    """
    return participation(r_strategy, r_benchmark)["beta_up"]


# ── Full metric summary ──────────────────────────────────────────────────────

def compute_all(r_strategy, r_benchmark, label: str = "") -> pd.Series:
    """
    Compute the full metric set for one strategy vs one benchmark.
    Returns a named Series for easy stacking into a results table.
    """
    d = {}

    # --- primary ---
    d["ann_mean"]         = annualised_mean(r_strategy)
    d["cvar95_reduction"] = cvar_reduction(r_strategy, r_benchmark, 0.95)
    d["mdd_reduction"]    = max_drawdown_reduction(r_strategy, r_benchmark)
    d["upside_part"]      = upside_participation(r_strategy, r_benchmark)
    d["sortino"]          = sortino(r_strategy)

    # --- supporting risk ---
    d["var95"]            = var(r_strategy, 0.95)
    d["var99"]            = var(r_strategy, 0.99)
    d["cvar95"]           = cvar(r_strategy, 0.95)
    d["cvar99"]           = cvar(r_strategy, 0.99)
    d["var95_reduction"]  = var_reduction(r_strategy, r_benchmark, 0.95)
    d["var99_reduction"]  = var_reduction(r_strategy, r_benchmark, 0.99)
    d["cvar99_reduction"] = cvar_reduction(r_strategy, r_benchmark, 0.99)
    d["variance_red"]     = variance_reduction(r_strategy, r_benchmark)
    d["mdd"]              = max_drawdown(r_strategy)
    d["calmar"]           = calmar(r_strategy)

    # --- distributional ---
    d["skewness"]         = skewness(r_strategy)
    d["exc_kurtosis"]     = excess_kurtosis(r_strategy)
    d["var_cvar_gap95"]   = var_cvar_gap(r_strategy, 0.95)

    # --- risk-adjusted ---
    d["sharpe"]           = sharpe(r_strategy)
    d["ann_vol"]          = annualised_vol(r_strategy)

    # --- full HM participation block ---
    hm = participation(r_strategy, r_benchmark)
    for k, v in hm.items():
        d[f"hm_{k}"] = v

    return pd.Series(d, name=label)


# ── Validation ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n   = 500

    # Benchmark: random walk with slight negative drift (unhedged long CNH)
    r_bench = rng.normal(-0.0002, 0.003, n)

    # Strategy A: forward hedge  -- offsets most of benchmark, small positive carry
    r_fwd   = -r_bench * 0.95 + rng.normal(0.00008, 0.0003, n)

    # Strategy B: risk reversal  -- asymmetric; keeps some upside, limits downside
    r_rr    = np.where(r_bench < 0,
                       -r_bench * 0.85 + rng.normal(0.00005, 0.0004, n),
                        r_bench * 0.30 + rng.normal(0.00005, 0.0004, n))

    print("=== Validation on toy series (n=500) ===\n")

    for label, rs in [("Forward", r_fwd), ("RR", r_rr)]:
        s = compute_all(rs, r_bench, label)
        print(f"--- {label} ---")
        primary = ["ann_mean","cvar95_reduction","mdd_reduction",
                   "upside_part","sortino"]
        for m in primary:
            print(f"  {m:20s} {s[m]:+.4f}")
        print()

    print("=== Scalar sanity checks ===")
    print(f"  CVaR-95 benchmark         {cvar(r_bench, 0.95):+.6f}  (positive = loss)")
    print(f"  CVaR-95 forward           {cvar(r_fwd,   0.95):+.6f}")
    print(f"  CVaR-95 reduction (fwd)   {cvar_reduction(r_fwd, r_bench):+.6f}  (positive = less risk)")
    print(f"  Max drawdown benchmark    {max_drawdown(r_bench):+.6f}  (positive magnitude)")
    print(f"  Max drawdown forward      {max_drawdown(r_fwd):+.6f}")
    print(f"  Variance reduction (fwd)  {variance_reduction(r_fwd, r_bench):+.6f}  (near 1 = good hedge)")
    hm = participation(r_rr, r_bench)
    print(f"  HM beta_up  (rr)          {hm['beta_up']:+.4f}")
    print(f"  HM beta_down(rr)          {hm['beta_down']:+.4f}")
    print(f"  HM n_up / n_down          {hm['n_up']} / {hm['n_down']}")

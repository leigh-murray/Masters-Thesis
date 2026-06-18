"""
descriptives.py
---------------
Deterministic descriptive layer for the hedged USDCNH portfolios. NO inference,
NO bootstrap -- point numbers and pictures only. Reads the master return frame
produced by analysis.run_all() (index = trading days; columns ret_forward,
ret_rr_spot, ret_rr_delta, ret_unhedged + spot/vol/carry regime labels).

Provides:
  summary_table(master)            -> deterministic table, every strategy x regime
  plot_cumulative_pnl(master)      -> cumulative P&L paths, regime-shaded
  plot_rolling_return(master)      -> rolling 12-month return paths, regime-shaded
  plot_regime_pnl_bars(master)     -> P&L by regime, per strategy (spot/vol/carry)

Returns are daily P&L on a unit physical notional, so cumulative P&L = cumsum of
returns and a 12-month return = trailing 252-trading-day sum.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import metrics as M

ANN = 252

STRATS = {
    "ret_forward":  "Forward",
    "ret_rr_spot":  "RR (spot)",
    "ret_rr_delta": "RR (delta)",
    "ret_unhedged": "Unhedged",
}
STRAT_COLOUR = {
    "ret_forward":  "#1f77b4",
    "ret_rr_spot":  "#d62728",
    "ret_rr_delta": "#9467bd",
    "ret_unhedged": "#000000",
}
SPOT_SHADE = {"APPRECIATION": "#2ca02c", "DEPRECIATION": "#d62728", "RANGE": "#7f7f7f"}

REGIME_SLICES = [
    ("full",               None,           None),
    ("spot:APPRECIATION",  "spot_regime",  "APPRECIATION"),
    ("spot:DEPRECIATION",  "spot_regime",  "DEPRECIATION"),
    ("vol:LOW",            "vol_regime",   "LOW"),
    ("vol:MED",            "vol_regime",   "MED"),
    ("vol:HIGH",           "vol_regime",   "HIGH"),
    ("carry:POSITIVE",     "carry_regime", "POSITIVE"),
    ("carry:NEGATIVE",     "carry_regime", "NEGATIVE"),
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. DETERMINISTIC SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
def _upside_capture_ratio(rs, rb):
    """Mean strategy return on benchmark-up days / mean benchmark return on those days."""
    rs, rb = np.asarray(rs, float), np.asarray(rb, float)
    up = rb > 0
    if up.sum() < M.MIN_OBS or rb[up].mean() == 0:
        return np.nan
    return rs[up].mean() / rb[up].mean()


def _downside_capture_ratio(rs, rb):
    rs, rb = np.asarray(rs, float), np.asarray(rb, float)
    dn = rb < 0
    if dn.sum() < M.MIN_OBS or rb[dn].mean() == 0:
        return np.nan
    return rs[dn].mean() / rb[dn].mean()


def summary_table(master):
    """
    Deterministic point-metric table for every strategy in every regime slice.
    Columns:
        n            observations in the slice
        cum_pnl      total P&L over the slice (unit notional)
        ann_mean     annualised mean return
        ann_vol      annualised volatility
        cvar95       expected loss in worst 5% of days (positive = loss)
        max_drawdown peak-to-trough on the cumulative P&L (positive magnitude)
        upside_capture   Henriksson-Merton up-beta vs unhedged (same as the
                         'upside participation' metric used in the inference)
        up_capture_ratio / down_capture_ratio   intuitive capture ratios
    """
    bench = master["ret_unhedged"].values
    rows = []
    for key, col, val in REGIME_SLICES:
        sel = np.ones(len(master), bool) if val is None else (master[col] == val).values
        rb = bench[sel]
        for scol, sname in STRATS.items():
            rs = master[scol].values[sel]
            rows.append(dict(
                strategy=sname,
                regime=key,
                n=int(sel.sum()),
                cum_pnl=float(np.nansum(rs)),
                ann_mean=M.annualised_mean(rs),
                ann_vol=M.annualised_vol(rs),
                cvar95=M.cvar(rs, 0.95),
                max_drawdown=M.max_drawdown(rs),
                upside_capture=M.upside_participation(rs, rb),
                up_capture_ratio=_upside_capture_ratio(rs, rb),
                down_capture_ratio=_downside_capture_ratio(rs, rb),
            ))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# REGIME SHADING HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _regime_spans(labels):
    """Contiguous runs of a label series -> list of (start_date, end_date, label)."""
    vals, dates = labels.values, labels.index
    spans, start = [], 0
    for i in range(1, len(vals)):
        if vals[i] != vals[start]:
            spans.append((dates[start], dates[i], vals[start]))
            start = i
    spans.append((dates[start], dates[-1], vals[start]))
    return spans


def _shade_regimes(ax, master, regime_col="spot_regime"):
    for s, e, lab in _regime_spans(master[regime_col]):
        if lab in SPOT_SHADE:
            ax.axvspan(s, e, color=SPOT_SHADE[lab], alpha=0.10, lw=0)


def _shade_legend():
    return [mpatches.Patch(color=c, alpha=0.25, label=lab.title())
            for lab, c in SPOT_SHADE.items()]


# ─────────────────────────────────────────────────────────────────────────────
# 2. CUMULATIVE P&L
# ─────────────────────────────────────────────────────────────────────────────
def plot_cumulative_pnl(master, regime_col="spot_regime", figsize=(12, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    _shade_regimes(ax, master, regime_col)
    for scol, sname in STRATS.items():
        cum = master[scol].cumsum()
        ax.plot(master.index, cum, lw=1.4, color=STRAT_COLOUR[scol], label=sname,
                ls="--" if scol == "ret_unhedged" else "-")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("Cumulative P&L  (unit notional)")
    ax.set_title("Cumulative P&L by strategy  (background = spot regime)")
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles=h + _shade_legend(), loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. ROLLING 12-MONTH RETURN
# ─────────────────────────────────────────────────────────────────────────────
def plot_rolling_return(master, window=ANN, regime_col="spot_regime", figsize=(12, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    _shade_regimes(ax, master, regime_col)
    for scol, sname in STRATS.items():
        roll = master[scol].rolling(window).sum()
        ax.plot(master.index, roll, lw=1.4, color=STRAT_COLOUR[scol], label=sname,
                ls="--" if scol == "ret_unhedged" else "-")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel(f"Trailing {window}-day P&L  (unit notional)")
    ax.set_title(f"Rolling 12-month return by strategy  (background = spot regime)")
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles=h + _shade_legend(), loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. P&L BY REGIME, PER STRATEGY  (spot / vol / carry panels)
# ─────────────────────────────────────────────────────────────────────────────
def plot_regime_pnl_bars(master, figsize=(13, 4.2)):
    panels = [
        ("spot_regime",  ["APPRECIATION", "DEPRECIATION"], "Spot regime"),
        ("vol_regime",   ["LOW", "MED", "HIGH"],                     "Volatility regime"),
        ("carry_regime", ["POSITIVE", "NEGATIVE"],                   "Carry regime"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    width = 0.2
    for ax, (col, order, title) in zip(axes, panels):
        x = np.arange(len(order))
        for k, (scol, sname) in enumerate(STRATS.items()):
            vals = [master.loc[master[col] == r, scol].sum() for r in order]
            ax.bar(x + (k - 1.5) * width, vals, width,
                   color=STRAT_COLOUR[scol], label=sname)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(order, fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("Total P&L over regime  (unit notional)")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("P&L contribution by regime, per strategy", y=1.02)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import analysis
    master = analysis.run_all(B=20, verbose=False)["master"]
    tbl = summary_table(master)
    tbl.to_csv("descriptive_summary.csv", index=False)
    print(tbl.round(4).to_string(index=False))
    plot_cumulative_pnl(master).savefig("cum_pnl.png", dpi=120, bbox_inches="tight")
    plot_rolling_return(master).savefig("rolling_return.png", dpi=120, bbox_inches="tight")
    plot_regime_pnl_bars(master).savefig("regime_pnl_bars.png", dpi=120, bbox_inches="tight")
    print("\nSaved: descriptive_summary.csv, cum_pnl.png, rolling_return.png, regime_pnl_bars.png")

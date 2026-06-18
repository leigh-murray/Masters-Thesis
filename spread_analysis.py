"""
spread_analysis.py
------------------
Descriptive, inference-free comparison of two delta-neutral hedges by analysing
their P&L SPREAD directly:

        d_t = ret_<leg> - ret_<base>      (default: RR(delta) - Forward)

Because both legs are delta-neutral, d_t is a near-pure option/skew P&L stream --
"what you earned (or paid) for using the risk reversal instead of the forward".
Analysing this single series makes the small edge visible and decision-relevant
without the "could it be zero?" framing of the bootstrap.

Reads the master return frame from analysis.run_all() (DatetimeIndex; columns
ret_forward, ret_rr_spot, ret_rr_delta, ret_unhedged + spot/vol/carry regimes).

Outputs (all descriptive):
  spread_table(master)           -> per-regime spread stats (IR, hit rate, drawdown)
  monthly_spread(master)         -> monthly aggregation + monthly hit rate
  plot_cumulative_spread(master) -> the headline cumulative-spread chart
  plot_regime_spread_bars(master)-> where the edge comes from (spot/vol/carry)
  plot_monthly_spread(master)    -> monthly bars + cumulative overlay
  run_spread_report(master)      -> prints tables, returns everything, builds figs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ANN = 252
BPS = 1e4

LABELS = {
    "ret_forward":  "Forward",
    "ret_rr_spot":  "RR (spot)",
    "ret_rr_delta": "RR (delta)",
    "ret_unhedged": "Unhedged",
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
# SPREAD SERIES + STATS
# ─────────────────────────────────────────────────────────────────────────────
def spread_series(master, leg="ret_rr_delta", base="ret_forward"):
    """Daily spread d_t = leg - base, as a Series indexed by date."""
    return (master[leg] - master[base]).rename(f"{LABELS[leg]} - {LABELS[base]}")


def _spread_stats(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return dict(n=n, cum_spread_bps=np.nan, ann_mean_bps=np.nan,
                    ann_vol_bps=np.nan, info_ratio=np.nan, hit_rate=np.nan,
                    spread_max_dd_bps=np.nan)
    cum      = d.sum()
    ann_mean = d.mean() * ANN
    ann_vol  = d.std(ddof=1) * np.sqrt(ANN)
    ir       = ann_mean / ann_vol if ann_vol > 0 else np.nan
    hit      = float((d > 0).mean())
    C        = np.cumsum(d)
    max_dd   = float((np.maximum.accumulate(C) - C).max())
    return dict(
        n=n,
        cum_spread_bps=cum * BPS,
        ann_mean_bps=ann_mean * BPS,
        ann_vol_bps=ann_vol * BPS,
        info_ratio=ir,
        hit_rate=hit,
        spread_max_dd_bps=max_dd * BPS,
    )


def spread_table(master, leg="ret_rr_delta", base="ret_forward"):
    """
    Per-regime descriptive spread statistics.
      cum_spread_bps    total spread accumulated in the regime (bps of notional)
      ann_mean_bps      annualised mean edge (bps/yr); positive = leg beats base
      ann_vol_bps       annualised volatility of the spread (bps/yr)
      info_ratio        ann_mean / ann_vol  -- the descriptive 'reliability' of the edge
      hit_rate          fraction of days the leg beats the base
      spread_max_dd_bps worst peak-to-trough of the cumulative spread (regret of the leg)
    """
    d = spread_series(master, leg, base)
    rows = []
    for key, col, val in REGIME_SLICES:
        sel = np.ones(len(master), bool) if val is None else (master[col] == val).values
        stats = _spread_stats(d.values[sel])
        stats = {"regime": key, **stats}
        rows.append(stats)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# MONTHLY AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────
def monthly_spread(master, leg="ret_rr_delta", base="ret_forward"):
    """
    Aggregate the daily spread to calendar months (sum of daily P&L).
    Returns (monthly_series_bps, summary_dict).
    """
    d = spread_series(master, leg, base)
    m = d.groupby(d.index.to_period("M")).sum() * BPS      # monthly spread in bps
    m.index = m.index.to_timestamp()
    summary = dict(
        n_months=int(len(m)),
        monthly_hit_rate=float((m > 0).mean()),
        avg_monthly_bps=float(m.mean()),
        median_monthly_bps=float(m.median()),
        best_month_bps=float(m.max()),
        worst_month_bps=float(m.min()),
        monthly_info_ratio=float(m.mean() / m.std(ddof=1) * np.sqrt(12))
                            if m.std(ddof=1) > 0 else np.nan,
    )
    return m, summary


# ─────────────────────────────────────────────────────────────────────────────
# REGIME SHADING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _regime_spans(labels):
    vals, dates = labels.values, labels.index
    spans, start = [], 0
    for i in range(1, len(vals)):
        if vals[i] != vals[start]:
            spans.append((dates[start], dates[i], vals[start])); start = i
    spans.append((dates[start], dates[-1], vals[start]))
    return spans


def _shade(ax, master, regime_col="spot_regime"):
    for s, e, lab in _regime_spans(master[regime_col]):
        if lab in SPOT_SHADE:
            ax.axvspan(s, e, color=SPOT_SHADE[lab], alpha=0.10, lw=0)


def _shade_legend():
    return [mpatches.Patch(color=c, alpha=0.25, label=lab.title())
            for lab, c in SPOT_SHADE.items()]


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_cumulative_spread(master, leg="ret_rr_delta", base="ret_forward", figsize=(12, 5)):
    d = spread_series(master, leg, base)
    cum = d.cumsum() * BPS
    fig, ax = plt.subplots(figsize=figsize)
    _shade(ax, master)
    ax.plot(master.index, cum, lw=1.6, color="#1a1a1a")
    ax.axhline(0, color="grey", lw=0.7)
    ax.fill_between(master.index, cum, 0, where=(cum >= 0), color="#2ca02c", alpha=0.15)
    ax.fill_between(master.index, cum, 0, where=(cum < 0),  color="#d62728", alpha=0.15)
    final = cum.iloc[-1]
    ax.annotate(f"{final:+.0f} bps", xy=(master.index[-1], final),
                xytext=(-60, 10), textcoords="offset points", fontsize=10, weight="bold")
    ax.set_ylabel("Cumulative spread (bps of notional)")
    ax.set_title(f"Cumulative spread:  {LABELS[leg]}  -  {LABELS[base]}"
                 f"   (positive = {LABELS[leg]} ahead)")
    ax.legend(handles=_shade_legend(), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_regime_spread_bars(master, leg="ret_rr_delta", base="ret_forward", figsize=(13, 4.2)):
    d = spread_series(master, leg, base)
    panels = [
        ("spot_regime",  ["APPRECIATION", "DEPRECIATION"], "Spot regime"),
        ("vol_regime",   ["LOW", "MED", "HIGH"],                     "Volatility regime"),
        ("carry_regime", ["POSITIVE", "NEGATIVE"],                   "Carry regime"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    for ax, (col, order, title) in zip(axes, panels):
        vals = [d.values[(master[col] == r).values].sum() * BPS for r in order]
        colours = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
        ax.bar(order, vals, color=colours)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Cumulative spread (bps)")
    fig.suptitle(f"Where the edge comes from:  {LABELS[leg]} - {LABELS[base]}  by regime",
                 y=1.02)
    fig.tight_layout()
    return fig


def plot_monthly_spread(master, leg="ret_rr_delta", base="ret_forward", figsize=(12, 5)):
    m, summ = monthly_spread(master, leg, base)
    fig, ax = plt.subplots(figsize=figsize)
    colours = ["#2ca02c" if v >= 0 else "#d62728" for v in m.values]
    ax.bar(m.index, m.values, width=20, color=colours, alpha=0.7)
    ax.axhline(0, color="grey", lw=0.7)
    ax.set_ylabel("Monthly spread (bps)")
    ax2 = ax.twinx()
    ax2.plot(m.index, m.cumsum(), color="#1a1a1a", lw=1.6, label="cumulative")
    ax2.set_ylabel("Cumulative (bps)")
    ax.set_title(f"Monthly spread:  {LABELS[leg]} - {LABELS[base]}   "
                 f"(hit rate {summ['monthly_hit_rate']:.0%}, "
                 f"avg {summ['avg_monthly_bps']:+.1f} bps/mo)")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DRIVER
# ─────────────────────────────────────────────────────────────────────────────
def run_spread_report(master, leg="ret_rr_delta", base="ret_forward", make_figs=True):
    tbl = spread_table(master, leg, base)
    m, msumm = monthly_spread(master, leg, base)

    print("=" * 72)
    print(f"SPREAD REPORT:  {LABELS[leg]}  vs  {LABELS[base]}   (positive = {LABELS[leg]} ahead)")
    print("=" * 72)
    full = tbl[tbl["regime"] == "full"].iloc[0]
    print(f"Full sample: {int(full['n'])} days")
    print(f"  cumulative edge   : {full['cum_spread_bps']:+.1f} bps of notional over the sample")
    print(f"  annualised edge   : {full['ann_mean_bps']:+.1f} bps/yr")
    print(f"  information ratio : {full['info_ratio']:+.2f}   (consistency of the edge)")
    print(f"  daily hit rate    : {full['hit_rate']:.1%}")
    print(f"  spread max drawdown: {full['spread_max_dd_bps']:.1f} bps  "
          f"(worst stretch of {LABELS[leg]} trailing {LABELS[base]})")
    print(f"\nMonthly: {msumm['n_months']} months | hit rate {msumm['monthly_hit_rate']:.0%} "
          f"| avg {msumm['avg_monthly_bps']:+.1f} bps/mo | IR {msumm['monthly_info_ratio']:+.2f}")
    print(f"  best/worst month  : {msumm['best_month_bps']:+.1f} / {msumm['worst_month_bps']:+.1f} bps")

    print("\nPer-regime spread (bps):")
    show = tbl.copy()
    for c in ["cum_spread_bps", "ann_mean_bps", "ann_vol_bps", "spread_max_dd_bps"]:
        show[c] = show[c].round(1)
    show["info_ratio"] = show["info_ratio"].round(2)
    show["hit_rate"] = (show["hit_rate"] * 100).round(0).astype("Int64").astype(str) + "%"
    print(show[["regime","n","cum_spread_bps","ann_mean_bps","info_ratio","hit_rate","spread_max_dd_bps"]]
          .to_string(index=False))

    figs = {}
    if make_figs:
        figs["cumulative"] = plot_cumulative_spread(master, leg, base)
        figs["regime"]     = plot_regime_spread_bars(master, leg, base)
        figs["monthly"]    = plot_monthly_spread(master, leg, base)

    return dict(table=tbl, monthly=m, monthly_summary=msumm, figs=figs)


if __name__ == "__main__":
    import analysis
    master = analysis.run_all(B=20, verbose=False)["master"]
    res = run_spread_report(master, leg="ret_rr_delta", base="ret_forward")
    res["figs"]["cumulative"].savefig("spread_cumulative.png", dpi=120, bbox_inches="tight")
    res["figs"]["regime"].savefig("spread_regime.png", dpi=120, bbox_inches="tight")
    res["figs"]["monthly"].savefig("spread_monthly.png", dpi=120, bbox_inches="tight")
    print("\nSaved: spread_cumulative.png, spread_regime.png, spread_monthly.png")

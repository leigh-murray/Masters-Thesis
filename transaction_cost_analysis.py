"""
transaction_cost_analysis.py
-----------------------------
Transaction-cost HURDLE sweep for the hedged USDCNH portfolios.

Question
--------
All headline results assume zero transaction costs. A risk reversal trades two
option legs where the forward trades one, so zero costs biases every comparison
toward the RRs. This module asks: how large does the per-leg option cost have to
be before each RR edge over the forward disappears?

Two break-evens are the focus (per the thesis plan):
  1. RR (delta) - forward, full sample, ann_mean   -> expected FRAGILE
  2. RR (spot)  - forward, APPRECIATION, ann_mean   -> expected ROBUST

Roll structure (read directly from the regime CSVs)
---------------------------------------------------
Each hedged portfolio runs a 3-CONTRACT OVERLAPPING LADDER. The columns
Position1/2/3_Active_Contract each hold one ~3M contract; the three are staggered
~1 month apart. They roll ONE AT A TIME (never simultaneously), giving 60 roll
events over the sample at ~monthly cadence. A roll event is therefore any date on
which ANY of the three Position columns changes versus the previous trading day.
The forward and both RRs share the SAME roll schedule (identical contract IDs),
so the roll dates coincide across strategies.

Because the three ladder slots SHARE the book (total Hedging_Notional ~ 0.94 at a
hedge ratio of ~1.0), each slot carries ~1/3 of the notional. One monthly roll
therefore transacts ~1/3 of the book. This is set by ROLL_NOTIONAL_FRACTION
(default 1/N_POSITIONS = 1/3); set it to 1.0 if you prefer a "full notional per
roll" convention (this scales every break-even by 1/fraction, i.e. x3).

Cost model
----------
Cost charged on each roll event for strategy s:

    cost = LEGS[s] * cost_bps * 1e-4 * (Hedging_Notional[date] * ROLL_NOTIONAL_FRACTION)

  - LEGS: forward = 1 (one forward rolled); each RR = 2 (call + put rolled).
  - cost_bps: the swept variable, in bps per leg per roll (interpret as the all-in
    round-trip cost of rolling one leg).
  - OPTION legs (both RRs) use cost_bps_option (swept); the FORWARD leg uses
    cost_bps_forward (default 0, isolating the incremental option-cost hurdle and
    conservative against the RR). Set them equal for a symmetric per-leg cost.

net_return = gross_return - cost_bps * cost_unit, where cost_unit is a fixed
per-day series. ann_mean is exactly linear in cost; CVaR/drawdown are piecewise,
so the sweep evaluates metrics over a fine cost grid and locates the zero-crossing
by linear interpolation (uniform across linear and nonlinear metrics).

Reuses metrics.py; reuses analysis.load_and_build for the exact same return
construction; optionally reuses pairwise_bootstrap.py for CIs net of cost.

Outputs
-------
  cost_sweep(...)          -> tidy (comparison, regime, metric, cost_bps, value)
  breakeven_table(...)     -> interpolated break-even cost per row
  bootstrap_at_cost(...)   -> CIs / verdicts net of a chosen cost level
  run_cost_analysis(...)   -> driver: prints headline break-evens, saves CSVs
"""

import numpy as np
import pandas as pd

import metrics as M
import analysis  # for load_and_build (shares the exact return-construction logic)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  --  column names verified against the uploaded hedged_*_regimes.csv
# ─────────────────────────────────────────────────────────────────────────────
COL_PNL       = "Portfolio_Cumulative_PnL"            # cumulative P&L (used by analysis.py)
COL_CONTRACTS = ["Position1_Active_Contract",         # roll = ANY of these changes
                 "Position2_Active_Contract",
                 "Position3_Active_Contract"]
COL_NOTIONAL  = "Hedging_Notional"                    # total book notional (~0.94)
N_POSITIONS   = 3                                      # ladder slots sharing the book

# fraction of the book transacted at a single roll event. One of the three
# overlapping slots rolls per event, so the data-consistent default is 1/3.
ROLL_NOTIONAL_FRACTION = 1.0 / N_POSITIONS

# legs traded when one contract of the strategy rolls
LEGS = {"forward": 1, "rr_spot": 2, "rr_delta": 2, "unhedged": 0}

ANN = 252
BPS = 1e4

OPTION_STRATS  = ("rr_spot", "rr_delta")
FORWARD_STRATS = ("forward",)

RET = {
    "ret_forward":  "forward",
    "ret_rr_spot":  "rr_spot",
    "ret_rr_delta": "rr_delta",
    "ret_unhedged": "unhedged",
}

REGIME_SLICES = [
    ("full",              None,          None),
    ("spot:APPRECIATION", "spot_regime", "APPRECIATION"),
    ("spot:DEPRECIATION", "spot_regime", "DEPRECIATION"),
    ("spot:RANGE",        "spot_regime", "RANGE"),
]

COMPARISONS = [
    ("rr_spot - fwd",  "rr_spot"),
    ("rr_delta - fwd", "rr_delta"),
]


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY DIFFERENCE METRICS  (identical definitions to analysis.py)
# ─────────────────────────────────────────────────────────────────────────────
def _beta_up_fast(rs, rb):
    rs = np.asarray(rs, float); rb = np.asarray(rb, float)
    ok = np.isfinite(rs) & np.isfinite(rb)
    rs, rb = rs[ok], rb[ok]
    if len(rs) < M.MIN_OBS:
        return np.nan
    up = (rb > 0).astype(float); down = (rb <= 0).astype(float)
    X = np.column_stack([np.ones_like(rb), rb * up, rb * down])
    beta, *_ = np.linalg.lstsq(X, rs, rcond=None)
    return beta[1]


PRIMARY_DIFF = {
    "ann_mean":         lambda rs, rb: M.annualised_mean(rs),
    "cvar95_reduction": lambda rs, rb: M.cvar_reduction(rs, rb, 0.95),
    "mdd_reduction":    lambda rs, rb: M.max_drawdown_reduction(rs, rb),
    "upside_part":      lambda rs, rb: _beta_up_fast(rs, rb),
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. BUILD COST-UNIT SERIES  (per strategy, per trading day)
# ─────────────────────────────────────────────────────────────────────────────
def _roll_flags(frame, index, name):
    """Boolean Series on `index`: True where ANY Position contract changes.

    No calendar fallback: the roll schedule is read from the contract columns.
    Raises if the expected columns are absent so the error is loud, not silent.
    """
    missing = [c for c in COL_CONTRACTS if c not in frame.columns]
    if missing:
        raise KeyError(
            f"{name}: missing contract columns {missing}. "
            f"Expected {COL_CONTRACTS}. Found {list(frame.columns)}.")
    flags = pd.Series(False, index=index)
    for c in COL_CONTRACTS:
        col = frame[c].reindex(index)
        ch = col.ne(col.shift(1)) & col.notna()
        flags |= ch
    flags.iloc[0] = False                     # don't charge the dropped-t0 boundary
    return flags


def _notional(frame, index, name):
    """Per-day total hedging notional on `index`; raises if the column is absent."""
    if COL_NOTIONAL not in frame.columns:
        raise KeyError(
            f"{name}: notional column '{COL_NOTIONAL}' not found. "
            f"Found {list(frame.columns)}.")
    return frame[COL_NOTIONAL].reindex(index).ffill().bfill()


def build_cost_units(master, frames_norm, roll_fraction=ROLL_NOTIONAL_FRACTION,
                     verbose=True):
    """
    Cost-unit series per strategy: the P&L drag PER 1 bp of per-leg cost, per day.

        cost_unit_<name>[t] = LEGS[name] * 1e-4 * notional[t] * roll_fraction
                              on roll days, else 0

    Net return at cost_bps:  ret_<name> - cost_bps * cost_unit_<name>.
    Returns (units_df aligned to master.index, report dict).
    """
    idx = master.index
    units = pd.DataFrame(index=idx)
    report = {}
    for ret_col, name in RET.items():
        if name == "unhedged":
            units[name] = 0.0
            report[name] = dict(rolls=0, legs=0, notional_mean=np.nan,
                                per_roll_notional=np.nan)
            continue
        frame = frames_norm[name]
        flags = _roll_flags(frame, idx, name)
        notl  = _notional(frame, idx, name)
        per_roll_notional = notl.values * roll_fraction
        unit  = np.where(flags.values, LEGS[name] * 1e-4 * per_roll_notional, 0.0)
        units[name] = unit
        report[name] = dict(rolls=int(flags.sum()),
                            legs=LEGS[name],
                            notional_mean=float(notl.mean()),
                            per_roll_notional=float(np.mean(per_roll_notional)))
    if verbose:
        print("Cost-unit construction "
              f"(roll_fraction={roll_fraction:.4f} = 1/{N_POSITIONS} of the book per roll):")
        for name, r in report.items():
            if name == "unhedged":
                continue
            warn = "" if 0.05 <= r["notional_mean"] <= 20 else "  <-- notional not ~1; check units"
            print(f"  {name:9s} rolls={r['rolls']:3d}  legs/roll={r['legs']}  "
                  f"book notional~{r['notional_mean']:.3f}  "
                  f"traded/roll~{r['per_roll_notional']:.3f}{warn}")
        yrs = (idx[-1] - idx[0]).days / 365.25
        rpy = {n: report[n]['rolls'] / yrs for n in ('forward', 'rr_spot', 'rr_delta')}
        print(f"  sample years~{yrs:.2f}  roll events/yr: "
              f"fwd={rpy['forward']:.1f}, rr_spot={rpy['rr_spot']:.1f}, rr_delta={rpy['rr_delta']:.1f}")
        print()
    return units, report


# ─────────────────────────────────────────────────────────────────────────────
# 2. NET RETURNS AT A GIVEN COST
# ─────────────────────────────────────────────────────────────────────────────
def net_returns(master, units, cost_bps_option, cost_bps_forward=0.0):
    """Return a copy of master with ret_* columns replaced by cost-adjusted nets."""
    out = master.copy()
    for ret_col, name in RET.items():
        if name in OPTION_STRATS:
            out[ret_col] = master[ret_col] - cost_bps_option * units[name]
        elif name in FORWARD_STRATS:
            out[ret_col] = master[ret_col] - cost_bps_forward * units[name]
        # unhedged unchanged
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. DETERMINISTIC SWEEP  +  BREAK-EVEN
# ─────────────────────────────────────────────────────────────────────────────
def _diff_metrics_at(master_net):
    rows = []
    rb_full = master_net["ret_unhedged"].values
    rf_full = master_net["ret_forward"].values
    for cl, leg in COMPARISONS:
        rs_full = master_net[f"ret_{leg}"].values
        for rk, col, val in REGIME_SLICES:
            sel = (np.ones(len(master_net), bool) if val is None
                   else (master_net[col] == val).values)
            rb, rf, rs = rb_full[sel], rf_full[sel], rs_full[sel]
            for mn, mf in PRIMARY_DIFF.items():
                rows.append(dict(comparison=cl, regime=rk, metric=mn,
                                 value=mf(rs, rb) - mf(rf, rb)))
    return rows


def cost_sweep(master, units, grid_bps, cost_bps_forward=0.0):
    """Difference metrics across a grid of per-option-leg costs (long tidy frame)."""
    out = []
    for c in grid_bps:
        mnet = net_returns(master, units, cost_bps_option=c,
                           cost_bps_forward=cost_bps_forward)
        for row in _diff_metrics_at(mnet):
            row = dict(row); row["cost_bps"] = float(c)
            out.append(row)
    df = pd.DataFrame(out)
    return df[["comparison", "regime", "metric", "cost_bps", "value"]]


def _zero_crossing(cost, value):
    """First cost at which `value` crosses 0 (linear interp). Returns (be, status)."""
    cost = np.asarray(cost, float); value = np.asarray(value, float)
    order = np.argsort(cost); cost, value = cost[order], value[order]
    v0 = value[0]
    if not np.isfinite(v0):
        return np.nan, "undefined"
    if v0 <= 0:
        return 0.0, "negative at zero cost"
    for i in range(1, len(cost)):
        if value[i] <= 0:
            c0, c1, y0, y1 = cost[i-1], cost[i], value[i-1], value[i]
            be = c0 if y1 == y0 else c0 - y0 * (c1 - c0) / (y1 - y0)
            return float(be), "break-even within grid"
    return np.nan, "survives to grid ceiling"


def breakeven_table(sweep_df, rolls_per_year=None,
                    roll_fraction=ROLL_NOTIONAL_FRACTION, notional_means=None):
    """Break-even option cost per (comparison, regime, metric), in two units.

    breakeven_bps_notional_per_yr is the annual transaction cost expressed as bps
    of book notional at the break-even per-leg cost. It must include the same
    roll_fraction and per-strategy notional that are baked into cost_unit, so it
    equals:

        be * LEGS * roll_fraction * notional_mean * rolls_per_year

    By construction this equals the gross ann_mean edge at the break-even row,
    which doubles as an internal consistency check.
    """
    rows = []
    for (cl, rk, mn), g in sweep_df.groupby(["comparison", "regime", "metric"]):
        g = g.sort_values("cost_bps")
        gross = float(g.loc[g["cost_bps"] == g["cost_bps"].min(), "value"].iloc[0])
        be, status = _zero_crossing(g["cost_bps"].values, g["value"].values)
        leg = "rr_spot" if cl.startswith("rr_spot") else "rr_delta"
        rpy  = (rolls_per_year or {}).get(leg, np.nan)
        notl = (notional_means or {}).get(leg, np.nan)
        be_bps_yr = (be * LEGS[leg] * roll_fraction * notl * rpy
                     if (be is not None and np.isfinite(be) and np.isfinite(notl))
                     else np.nan)
        rows.append(dict(comparison=cl, regime=rk, metric=mn,
                         gross_value=gross,
                         breakeven_bps_per_leg_roll=be,
                         breakeven_bps_notional_per_yr=be_bps_yr,
                         status=status))
    order = {"ann_mean": 0, "cvar95_reduction": 1, "mdd_reduction": 2, "upside_part": 3}
    out = pd.DataFrame(rows)
    out["_o"] = out["metric"].map(order)
    return out.sort_values(["comparison", "regime", "_o"]).drop(columns="_o").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. BOOTSTRAP NET OF A CHOSEN COST  (significance, not just point break-even)
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_at_cost(master, units, cost_bps_option, cost_bps_forward=0.0,
                      B=10_000, verbose=True):
    """Re-run the paired block bootstrap (pairwise_bootstrap.py) on net returns."""
    import pairwise_bootstrap as pb
    mnet = net_returns(master, units, cost_bps_option, cost_bps_forward)
    parts = []
    for leg_col in ("ret_rr_spot", "ret_rr_delta"):
        summ, _, block = pb.bootstrap_pairwise(
            mnet, leg=leg_col, base="ret_forward",
            benchmark="ret_unhedged", B=B, verbose=False)
        summ.insert(0, "cost_bps_option", cost_bps_option)
        parts.append(summ)
    out = pd.concat(parts, ignore_index=True)
    if verbose:
        print(f"Bootstrap net of {cost_bps_option:.2f} bps/option-leg/roll "
              f"(forward {cost_bps_forward:.2f}) | block={block} | B={B}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# DRIVER
# ─────────────────────────────────────────────────────────────────────────────
def run_cost_analysis(frames, spot_dates,
                      grid_bps=None, cost_bps_forward=0.0,
                      roll_fraction=ROLL_NOTIONAL_FRACTION,
                      boot_costs=(0.5, 1.0, 2.0), B=10_000,
                      save_prefix="cost_", verbose=True):
    """
    frames     : dict {forward, rr_spot, rr_delta, unhedged} of raw
                 hedged_*_regimes DataFrames (Pricing_Date, regime cols, COL_PNL,
                 Position{1,2,3}_Active_Contract, Hedging_Notional).
    spot_dates : the pure trading-day calendar, e.g. spot_df["Pricing_Date"].
    grid_bps   : option-cost grid in bps/leg/roll (default 0..25 step 0.25).
    roll_fraction : book fraction transacted per roll event (default 1/3).
    boot_costs : option-cost levels at which to bootstrap verdicts.

    Saves <prefix>sweep.csv, <prefix>breakeven.csv, <prefix>bootstrap.csv.
    """
    if grid_bps is None:
        grid_bps = np.round(np.arange(0.0, 25.0001, 0.25), 4)

    master, frames_norm, trading_days, _ = analysis.load_and_build(
        frames=frames, spot_dates=spot_dates)

    units, report = build_cost_units(master, frames_norm,
                                     roll_fraction=roll_fraction, verbose=verbose)
    yrs = (master.index[-1] - master.index[0]).days / 365.25
    rolls_per_year = {n: report[n]["rolls"] / yrs for n in ("forward", "rr_spot", "rr_delta")}

    sweep = cost_sweep(master, units, grid_bps, cost_bps_forward=cost_bps_forward)
    notional_means = {n: report[n]["notional_mean"] for n in ("forward", "rr_spot", "rr_delta")}
    be    = breakeven_table(sweep, rolls_per_year=rolls_per_year,
                            roll_fraction=roll_fraction, notional_means=notional_means)

    boot_parts = []
    for c in boot_costs:
        boot_parts.append(bootstrap_at_cost(master, units, c,
                                            cost_bps_forward=cost_bps_forward,
                                            B=B, verbose=verbose))
    boot = pd.concat(boot_parts, ignore_index=True) if boot_parts else pd.DataFrame()

    sweep.to_csv(f"{save_prefix}sweep.csv", index=False)
    be.to_csv(f"{save_prefix}breakeven.csv", index=False)
    if len(boot):
        boot.to_csv(f"{save_prefix}bootstrap.csv", index=False)

    if verbose:
        print("\n" + "=" * 78)
        print("HEADLINE BREAK-EVENS  (option cost, bps per leg per roll)")
        print("=" * 78)
        head = be[be["metric"] == "ann_mean"][[
            "comparison", "regime", "gross_value",
            "breakeven_bps_per_leg_roll", "breakeven_bps_notional_per_yr", "status"]].copy()
        head["gross_value"] = (head["gross_value"] * BPS).round(1)
        head = head.rename(columns={"gross_value": "gross_ann_mean_bps"})
        head["breakeven_bps_per_leg_roll"] = head["breakeven_bps_per_leg_roll"].round(2)
        head["breakeven_bps_notional_per_yr"] = head["breakeven_bps_notional_per_yr"].round(1)
        print(head.to_string(index=False))
        print("\nFull break-even table -> {0}breakeven.csv".format(save_prefix))
        print("Saved: {0}sweep.csv, {0}breakeven.csv, {0}bootstrap.csv".format(save_prefix))

    return dict(master=master, units=units, report=report,
                rolls_per_year=rolls_per_year,
                sweep=sweep, breakeven=be, bootstrap=boot)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL: metric-vs-cost plot for the two headline comparisons
# ─────────────────────────────────────────────────────────────────────────────
def plot_breakeven(sweep_df, breakeven_df, figsize=(12, 4.6)):
    import matplotlib.pyplot as plt
    panels = [("rr_delta - fwd", "full",              "Delta RR - Forward  (full)"),
              ("rr_spot - fwd",  "spot:APPRECIATION", "Spot RR - Forward  (appreciation)")]
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, (cl, rk, title) in zip(axes, panels):
        g = sweep_df[(sweep_df["comparison"] == cl) &
                     (sweep_df["regime"] == rk) &
                     (sweep_df["metric"] == "ann_mean")].sort_values("cost_bps")
        ax.plot(g["cost_bps"], g["value"] * BPS, lw=1.6, color="#1a1a1a")
        ax.axhline(0, color="grey", lw=0.7)
        be = breakeven_df[(breakeven_df["comparison"] == cl) &
                          (breakeven_df["regime"] == rk) &
                          (breakeven_df["metric"] == "ann_mean")]
        if len(be) and np.isfinite(be["breakeven_bps_per_leg_roll"].iloc[0]):
            x = be["breakeven_bps_per_leg_roll"].iloc[0]
            ax.axvline(x, color="#d62728", ls="--", lw=1.2)
            ax.annotate(f"break-even\n{x:.2f} bps/leg/roll", xy=(x, 0),
                        xytext=(6, 10), textcoords="offset points",
                        fontsize=8, color="#d62728")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("option cost (bps per leg per roll)")
        ax.set_ylabel("net ann_mean edge (bps of notional)")
    fig.suptitle("Transaction-cost hurdle: RR edge over forward vs option cost", y=1.02)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Adjust these paths to your project's folder layout.
    paths = {
        "forward":  "Regimes/hedged_forward_spot_regimes.csv",
        "rr_spot":  "Regimes/hedged_rr45_spot_regimes.csv",
        "rr_delta": "Regimes/hedged_rr45_delta_regimes.csv",
        "unhedged": "Regimes/unhedged_regimes.csv",
    }
    frames = {k: pd.read_csv(p) for k, p in paths.items()}
    spot = pd.read_csv("Market Data/spot_price.csv")
    spot["Pricing_Date"] = pd.to_datetime(spot["Pricing_Date"], dayfirst=True)
    res = run_cost_analysis(frames, spot["Pricing_Date"])

"""
analysis.py
-----------
Full statistical analysis of the hedged USDCNH portfolios.

Pipeline:
  1. Load the four regime-tagged P&L frames + the spot trading calendar.
  2. Build daily MTM returns (diff of cumulative P&L), restricted to TRADING
     days only -- the forward-filled weekend/holiday rows are dropped so they
     do not deflate volatility or corrupt the sqrt(252) annualisation.
  3. Data sanity checks.
  4. Point estimates: full metric set, every strategy, full sample + each
     parallel regime slice (spot x3, vol x3, carry x2).
  5. RR-minus-forward paired differences with stationary block-bootstrap CIs
     (one resampled path per replicate, applied to all strategies, differences
     taken on the same path -> cross-strategy correlation preserved; regime
     labels travel with the resampled rows so non-contiguous slices are valid).
     Sortino is bootstrapped as a per-strategy LEVEL, not a difference, because
     the difference is ill-conditioned when a fully-hedged leg has ~zero downside.
  6. Bootstrap sanity checks + adjudication against the locked decision rule.

Returns a dict of results for downstream inference.  Run as a script for a
full report, or import run_all() and call it.

Requires: numpy, pandas, scipy, statsmodels (for the point-estimate HM block);
optionally `arch` for Politis-White block-length selection (falls back to the
3M contract tenor of 63 trading days if unavailable).
"""

import numpy as np
import pandas as pd

import metrics as M     # the validated metric library (metrics.py in same dir)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PATHS = {
    "forward":  "hedged_forward_spot_regimes.csv",
    "rr_spot":  "hedged_rr45_spot_regimes.csv",
    "rr_delta": "hedged_rr45_delta_regimes.csv",
    "unhedged": "unhedged_regimes.csv",
}
SPOT_PATH       = "spot_price.csv"      # the pure trading-day calendar
PNL_COL         = "Portfolio_Cumulative_PnL"
CONTRACT_TENOR  = 69                    # Maximum contract tenor in trading days -> block-length floor
N_BOOT          = 10_000               # bootstrap replicates (set lower to test)
CI              = 95                    # confidence level (%)
SEED            = 20240617

# Primary metrics (all "higher = better"); each takes (r_strategy, r_benchmark)
def _beta_up_fast(rs, rb):
    """Closed-form Henriksson-Merton upside beta (fast; used inside bootstrap)."""
    rs = np.asarray(rs, float); rb = np.asarray(rb, float)
    m = np.isfinite(rs) & np.isfinite(rb)
    rs, rb = rs[m], rb[m]
    if len(rs) < M.MIN_OBS:
        return np.nan
    up = (rb > 0).astype(float); down = (rb <= 0).astype(float)
    X = np.column_stack([np.ones_like(rb), rb * up, rb * down])
    beta, *_ = np.linalg.lstsq(X, rs, rcond=None)
    return beta[1]

# Difference metrics (RR - forward), all "higher = better".  Sortino is NOT a
# difference metric: a fully-hedged strategy can have almost no downside in some
# regimes, so its downside deviation -> 0, the ratio explodes, and the DIFFERENCE
# of two such ratios is ill-conditioned (enormous CIs).  Sortino is therefore
# reported as a per-strategy LEVEL, each with its own bootstrap CI.
PRIMARY_DIFF = {
    "ann_mean":         lambda rs, rb: M.annualised_mean(rs),
    "cvar95_reduction": lambda rs, rb: M.cvar_reduction(rs, rb, 0.95),
    "mdd_reduction":    lambda rs, rb: M.max_drawdown_reduction(rs, rb),
    "upside_part":      lambda rs, rb: _beta_up_fast(rs, rb),
}

# Strategies for which Sortino is bootstrapped as a LEVEL (not a difference).
STRAT_COLS = ["ret_forward", "ret_rr_spot", "ret_rr_delta", "ret_unhedged"]

# Parallel regime slices (key, column, value). value=None -> full sample.
REGIME_SLICES = [
    ("full",               None,           None),
    ("spot:APPRECIATION",  "spot_regime",  "APPRECIATION"),
    ("spot:DEPRECIATION",  "spot_regime",  "DEPRECIATION"),
    # RANGE removed: spot is now two-state (appreciation / depreciation).
    ("vol:LOW",            "vol_regime",   "LOW"),
    ("vol:MED",            "vol_regime",   "MED"),
    ("vol:HIGH",           "vol_regime",   "HIGH"),
    ("carry:POSITIVE",     "carry_regime", "POSITIVE"),
    ("carry:NEGATIVE",     "carry_regime", "NEGATIVE"),
]

COMPARISONS = [   # (label, strategy column) -- always differenced against forward
    ("rr_spot - fwd",  "ret_rr_spot"),
    ("rr_delta - fwd", "ret_rr_delta"),
]

# Two-state spot scheme (RANGE removed). Trading-day counts on the spot calendar.
# Approximate; the sanity check tolerates +/- 3. Confirm against your spot_price.csv.
EXPECTED_SPOT = {"APPRECIATION": 571, "DEPRECIATION": 660}  # trading days


# ─────────────────────────────────────────────────────────────────────────────
# 1-2. LOAD + BUILD RETURNS
# ─────────────────────────────────────────────────────────────────────────────
def load_and_build(paths=PATHS, spot_path=SPOT_PATH, pnl_col=PNL_COL,
                   frames=None, spot_dates=None):
    """
    Build the master return frame.

    Reads from disk by default.  To run from a notebook on frames you already
    hold in memory, pass:
        frames     = {"forward": fwd_df, "rr_spot": rr_spot_df,
                      "rr_delta": rr_delta_df, "unhedged": unhedged_df}
        spot_dates = spot_df["Pricing_Date"]
    Each frame must contain a Pricing_Date column (or datetime index), the
    regime columns, and `pnl_col`.
    """
    # trading-day calendar
    if spot_dates is not None:
        trading_days = pd.DatetimeIndex(sorted(pd.to_datetime(spot_dates).unique()))
    else:
        spot = pd.read_csv(spot_path)
        spot["Pricing_Date"] = pd.to_datetime(spot["Pricing_Date"], dayfirst=True)
        trading_days = pd.DatetimeIndex(sorted(spot["Pricing_Date"].unique()))

    # frames: read from disk, or take the in-memory dict
    if frames is None:
        frames = {}
        for name, p in paths.items():
            frames[name] = pd.read_csv(p)

    # normalise every frame to a sorted datetime index on Pricing_Date
    norm = {}
    for name, df in frames.items():
        d = df.copy()
        if "Pricing_Date" in d.columns:
            d["Pricing_Date"] = pd.to_datetime(d["Pricing_Date"])
            d = d.sort_values("Pricing_Date").set_index("Pricing_Date")
        else:
            d.index = pd.to_datetime(d.index)
            d = d.sort_index()
        norm[name] = d
    frames = norm

    # daily return = diff of cumulative P&L on the FULL calendar (so the first
    # trading day after a gap correctly spans the gap), THEN restrict to
    # trading days only.
    master = pd.DataFrame(index=trading_days)
    for name, df in frames.items():
        daily = df[pnl_col].diff()
        master[f"ret_{name}"] = daily.reindex(trading_days)

    # regime labels (from the forward frame; verified identical across frames)
    reg_cols = ["spot_regime", "vol_regime", "carry_regime"]
    for c in reg_cols:
        master[c] = frames["forward"][c].reindex(trading_days)

    master = master.dropna(subset=[f"ret_{n}" for n in frames])  # drop t0 (NaN diff)
    return master, frames, trading_days, reg_cols


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA SANITY CHECKS
# ─────────────────────────────────────────────────────────────────────────────
def sanity_data(master, frames, trading_days, reg_cols, pnl_col=PNL_COL):
    print("=" * 70); print("[SANITY] DATA"); print("=" * 70)
    cal = len(next(iter(frames.values())))
    print(f"calendar rows per frame      : {cal}")
    print(f"trading days (spot calendar) : {len(trading_days)}")
    print(f"non-trading rows dropped     : {cal - len(trading_days)}")
    print(f"return rows after dropping t0: {len(master)}")

    # regimes identical across frames?
    ok = all(
        (
            frames[n][reg_cols].reindex(trading_days).astype(str).values
            == frames["forward"][reg_cols].reindex(trading_days).astype(str).values
        ).all()
        for n in frames
    )
    print(f"regime labels identical across all 4 frames: {ok}")

    # zero-return fraction (should be ~0 if filled days truly dropped)
    for n in ("forward", "rr_spot", "rr_delta", "unhedged"):
        z = (master[f"ret_{n}"] == 0).mean()
        flag = "  <-- HIGH, check filling" if z > 0.05 else ""
        print(f"zero-return fraction {n:9s}: {z:6.2%}{flag}")

    # P&L reconciliation: sum of daily returns ~ final cumulative P&L
    print("\nreconciliation (sum daily ret vs final cumulative P&L):")
    for n in ("forward", "rr_spot", "rr_delta", "unhedged"):
        recon = master[f"ret_{n}"].sum()
        final = frames[n][pnl_col].reindex(trading_days).iloc[-1]
        print(f"  {n:9s} sum={recon:+.5f}  final={final:+.5f}  diff={recon-final:+.2e}")

    # spot regime counts vs expected
    print("\nspot-regime counts (trading days):")
    vc = master["spot_regime"].value_counts()
    for k, exp in EXPECTED_SPOT.items():
        got = int(vc.get(k, 0))
        print(f"  {k:13s} got={got:4d}  expected~{exp}  {'ok' if abs(got-exp)<=3 else 'CHECK'}")
    print(f"  UNCLASSIFIED  got={int(vc.get('UNCLASSIFIED',0))}")
    print("vol-regime counts :", dict(master["vol_regime"].value_counts()))
    print("carry-regime counts:", dict(master["carry_regime"].value_counts()))

    # benchmark plausibility
    bvol = M.annualised_vol(master["ret_unhedged"].values)
    fred = M.variance_reduction(master["ret_forward"].values, master["ret_unhedged"].values)
    print(f"\nunhedged annualised vol      : {bvol:.2%}  (expect ~3-6%)")
    print(f"forward variance reduction   : {fred:.3f}  (expect high, >0.8)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 4. POINT ESTIMATES (full metric set, every strategy, every slice)
# ─────────────────────────────────────────────────────────────────────────────
def point_estimates(master):
    rows = []
    bench = master["ret_unhedged"].values
    for key, col, val in REGIME_SLICES:
        sel = np.ones(len(master), bool) if val is None else (master[col] == val).values
        rb = bench[sel]
        for strat in ("forward", "rr_spot", "rr_delta", "unhedged"):
            rs = master[f"ret_{strat}"].values[sel]
            s = M.compute_all(rs, rb, label=f"{strat}|{key}")
            s["strategy"] = strat; s["regime"] = key; s["n"] = int(sel.sum())
            rows.append(s)
    tbl = pd.DataFrame(rows).set_index(["regime", "strategy"])
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# 5. PAIRED STATIONARY BLOCK BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────
def optimal_block_length(x, floor=CONTRACT_TENOR):
    try:
        from arch.bootstrap import optimal_block_length as _obl
        L = float(_obl(np.asarray(x, float))["stationary"].iloc[0])
        return max(int(round(L)), floor), f"Politis-White={L:.1f}, floored at {floor}"
    except Exception:
        return floor, f"arch unavailable -> using contract tenor {floor}"


def _sb_indices(n, exp_block, rng):
    """Stationary (Politis-Romano) bootstrap indices: geometric blocks, wrapped."""
    p = 1.0 / exp_block
    out, count = [], 0
    while count < n:
        start = rng.integers(0, n)
        L = rng.geometric(p)
        out.append((start + np.arange(L)) % n)
        count += L
    return np.concatenate(out)[:n]


def bootstrap_differences(master, B=N_BOOT, ci=CI, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(master)
    block, block_msg = optimal_block_length(master["ret_unhedged"].values)

    # pre-extract arrays for speed
    R = {col: master[col].values for col in STRAT_COLS}
    codes, targets = {}, {}
    for _, col, _ in REGIME_SLICES:
        if col and col not in codes:
            c, uniq = pd.factorize(master[col].values)
            codes[col] = c; targets[col] = {u: i for i, u in enumerate(uniq)}

    # storage
    dist = {(cl, rk, mn): np.empty(B)                         # RR-forward diffs
            for cl, _ in COMPARISONS for rk, _, _ in REGIME_SLICES for mn in PRIMARY_DIFF}
    sort_dist = {(sc, rk): np.empty(B)                        # per-strategy Sortino levels
                 for sc in STRAT_COLS for rk, _, _ in REGIME_SLICES}

    for b in range(B):
        idx = _sb_indices(n, block, rng)
        Ridx   = {sc: R[sc][idx] for sc in STRAT_COLS}        # one resampled path, all strategies
        rb_all = Ridx["ret_unhedged"]; rfwd = Ridx["ret_forward"]
        rcode  = {col: codes[col][idx] for col in codes}
        for rk, col, val in REGIME_SLICES:
            mask = np.ones(n, bool) if val is None else (rcode[col] == targets[col][val])
            rb = rb_all[mask]; rf = rfwd[mask]
            # difference metrics (RR - forward)
            for cl, scol in COMPARISONS:
                rs = Ridx[scol][mask]
                for mn, mfun in PRIMARY_DIFF.items():
                    dist[(cl, rk, mn)][b] = mfun(rs, rb) - mfun(rf, rb)
            # Sortino LEVELS (one per strategy)
            for sc in STRAT_COLS:
                sort_dist[(sc, rk)][b] = M.sortino(Ridx[sc][mask])

    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    bench = R["ret_unhedged"]

    # ---- difference summary (4 metrics, with adjudication) ----
    diff_rows = []
    for cl, scol in COMPARISONS:
        for rk, col, val in REGIME_SLICES:
            sel = np.ones(n, bool) if val is None else (master[col] == val).values
            rb = bench[sel]; rf = R["ret_forward"][sel]; rs = master[scol].values[sel]
            for mn, mfun in PRIMARY_DIFF.items():
                point = mfun(rs, rb) - mfun(rf, rb)
                arr = dist[(cl, rk, mn)]
                lo, hi = np.nanpercentile(arr, [lo_q, hi_q])
                if lo > 0:      verdict = "RR better"
                elif hi < 0:    verdict = "forward better"
                else:           verdict = "inconclusive"
                diff_rows.append(dict(comparison=cl, regime=rk, metric=mn,
                                      point=point, ci_lo=lo, ci_hi=hi,
                                      n=int(sel.sum()), verdict=verdict,
                                      n_nan=int(np.isnan(arr).sum())))
    summ_diff = pd.DataFrame(diff_rows)

    # ---- Sortino-level summary (per strategy; compared by CI overlap) ----
    sort_rows = []
    for sc in STRAT_COLS:
        for rk, col, val in REGIME_SLICES:
            sel = np.ones(n, bool) if val is None else (master[col] == val).values
            point = M.sortino(R[sc][sel])
            arr = sort_dist[(sc, rk)]
            lo, hi = np.nanpercentile(arr, [lo_q, hi_q])
            sort_rows.append(dict(strategy=sc.replace("ret_", ""), regime=rk,
                                  sortino=point, ci_lo=lo, ci_hi=hi,
                                  n=int(sel.sum()), n_nan=int(np.isnan(arr).sum())))
    summ_sortino = pd.DataFrame(sort_rows)

    return summ_diff, summ_sortino, dist, sort_dist, block, block_msg


def sanity_bootstrap(summ, dist, block, block_msg, B, summ_sortino=None):
    print("=" * 70); print("[SANITY] BOOTSTRAP"); print("=" * 70)
    print(f"replicates B           : {B}")
    print(f"block length           : {block}   ({block_msg})")
    # point inside CI?
    inside = ((summ["point"] >= summ["ci_lo"]) & (summ["point"] <= summ["ci_hi"])).mean()
    print(f"point estimate inside CI: {inside:.1%}  (expect ~100%)")
    nan_any = summ["n_nan"].sum()
    print(f"diff NaN bootstrap draws : {nan_any}  (high -> a slice is too small)")
    tiny = summ[summ["n"] < 60]
    if len(tiny):
        print(f"slices with n<60 (thin) : {sorted(tiny['regime'].unique())}")
    if summ_sortino is not None:
        sn = int(summ_sortino["n_nan"].sum())
        width = (summ_sortino["ci_hi"] - summ_sortino["ci_lo"]).abs()
        wide = summ_sortino[width > 50]
        print(f"sortino-level NaN draws  : {sn}  (forward in low-downside regimes is unstable)")
        if len(wide):
            print("sortino levels with very wide CI (interpret with caution):")
            for _, r in wide.iterrows():
                print(f"    {r['strategy']:9s} {r['regime']:18s} "
                      f"[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# DRIVER
# ─────────────────────────────────────────────────────────────────────────────
def run_all(B=N_BOOT, ci=CI, seed=SEED, verbose=True, frames=None, spot_dates=None):
    master, frm, trading_days, reg_cols = load_and_build(frames=frames, spot_dates=spot_dates)
    if verbose:
        sanity_data(master, frm, trading_days, reg_cols)
    points = point_estimates(master)
    summ_diff, summ_sortino, dist, sort_dist, block, block_msg = \
        bootstrap_differences(master, B=B, ci=ci, seed=seed)
    if verbose:
        sanity_bootstrap(summ_diff, dist, block, block_msg, B, summ_sortino=summ_sortino)
    return dict(master=master, points=points,
                differences=summ_diff, sortino_levels=summ_sortino,
                boot_dist=dist, sortino_dist=sort_dist, block=block)


if __name__ == "__main__":
    res = run_all()
    summ = res["differences"]; sortino = res["sortino_levels"]

    for regime in ["full", "spot:APPRECIATION", "spot:DEPRECIATION"]:
        print("=" * 70)
        print(f"HEADLINE: RR - forward, difference metrics, {regime}")
        print("=" * 70)
        print(summ[summ["regime"] == regime]
            [["comparison","metric","point","ci_lo","ci_hi","verdict"]]
            .to_string(index=False))

        print(f"\nSortino LEVELS per strategy (own bootstrap CI), {regime}")
        print(sortino[sortino["regime"] == regime]
            [["strategy","sortino","ci_lo","ci_hi","n"]]
            .to_string(index=False))
        print()

    for regime in ["full", "spot:APPRECIATION", "spot:DEPRECIATION"]:
        safe_regime = regime.replace(":", "_")

        print("=" * 70)
        print(f"HEADLINE: RR - forward, difference metrics, {regime}")
        print("=" * 70)

        df_diff = summ[summ["regime"] == regime]
        df_sort = sortino[sortino["regime"] == regime]

        print(df_diff[["comparison","metric","point","ci_lo","ci_hi","verdict"]]
            .to_string(index=False))

        print(f"\nSortino LEVELS per strategy (own bootstrap CI), {regime}")
        print(df_sort[["strategy","sortino","ci_lo","ci_hi","n"]]
            .to_string(index=False))
        print()

        df_diff.to_csv(f"differences_{safe_regime}.csv", index=False)
        df_sort.to_csv(f"sortino_{safe_regime}.csv", index=False)
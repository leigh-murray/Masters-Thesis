"""
pairwise_bootstrap.py
---------------------
General paired stationary block-bootstrap for the difference between ANY two
strategies (leg - base), using the same method and metrics as analysis.py.

The four difference metrics are all "higher = leg better":
    ann_mean, cvar95_reduction, mdd_reduction, upside_part
(the reduction/participation metrics are taken vs the unhedged benchmark, which
cancels in the leg-base difference, leaving a clean pairwise comparison).

One resampled path per replicate is applied to BOTH legs and the benchmark, so
cross-strategy correlation is preserved; regime labels travel with the resampled
rows so the per-regime slices are valid for non-contiguous regimes.

Sortino is intentionally NOT bootstrapped here as a difference (ill-conditioned);
per-strategy Sortino LEVELS with their own CIs already come from analysis.py.

Usage:
    summ, dist, block = bootstrap_pairwise(master, leg="ret_rr_delta",
                                                   base="ret_rr_spot", B=10_000)
"""

import numpy as np
import pandas as pd
import metrics as M

CONTRACT_TENOR = 63
N_BOOT = 10_000
CI = 95
SEED = 20240619

LABELS = {
    "ret_forward":  "forward",
    "ret_rr_spot":  "rr_spot",
    "ret_rr_delta": "rr_delta",
    "ret_unhedged": "unhedged",
}

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


def _sb_indices(n, exp_block, rng):
    p = 1.0 / exp_block; out, count = [], 0
    while count < n:
        start = rng.integers(0, n)
        L = rng.geometric(p)
        out.append((start + np.arange(L)) % n)
        count += L
    return np.concatenate(out)[:n]


def optimal_block_length(x, floor=CONTRACT_TENOR):
    try:
        from arch.bootstrap import optimal_block_length as _obl
        L = float(_obl(np.asarray(x, float))["stationary"].iloc[0])
        return max(int(round(L)), floor), f"Politis-White={L:.1f}, floored at {floor}"
    except Exception:
        return floor, f"arch unavailable -> contract tenor {floor}"


def bootstrap_pairwise(master, leg="ret_rr_delta", base="ret_rr_spot",
                       benchmark="ret_unhedged", B=N_BOOT, ci=CI, seed=SEED,
                       verbose=True):
    rng = np.random.default_rng(seed)
    n = len(master)
    block, block_msg = optimal_block_length(master[benchmark].values)
    leg_lab, base_lab = LABELS[leg], LABELS[base]

    R = {c: master[c].values for c in (leg, base, benchmark)}
    codes, targets = {}, {}
    for _, col, _ in REGIME_SLICES:
        if col and col not in codes:
            c, u = pd.factorize(master[col].values)
            codes[col] = c; targets[col] = {x: i for i, x in enumerate(u)}

    dist = {(rk, mn): np.empty(B)
            for rk, _, _ in REGIME_SLICES for mn in PRIMARY_DIFF}

    for b in range(B):
        idx = _sb_indices(n, block, rng)
        rb_all = R[benchmark][idx]; rleg = R[leg][idx]; rbase = R[base][idx]
        rcode = {col: codes[col][idx] for col in codes}
        for rk, col, val in REGIME_SLICES:
            mask = np.ones(n, bool) if val is None else (rcode[col] == targets[col][val])
            rb = rb_all[mask]
            for mn, mf in PRIMARY_DIFF.items():
                dist[(rk, mn)][b] = mf(rleg[mask], rb) - mf(rbase[mask], rb)

    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    rows = []
    for rk, col, val in REGIME_SLICES:
        sel = np.ones(n, bool) if val is None else (master[col] == val).values
        rb = R[benchmark][sel]
        for mn, mf in PRIMARY_DIFF.items():
            point = mf(R[leg][sel], rb) - mf(R[base][sel], rb)
            arr = dist[(rk, mn)]
            lo, hi = np.nanpercentile(arr, [lo_q, hi_q])
            if lo > 0:    verdict = f"{leg_lab} better"
            elif hi < 0:  verdict = f"{base_lab} better"
            else:         verdict = "inconclusive"
            rows.append(dict(comparison=f"{leg_lab} - {base_lab}", regime=rk,
                             metric=mn, point=point, ci_lo=lo, ci_hi=hi,
                             n=int(sel.sum()), verdict=verdict,
                             n_nan=int(np.isnan(arr).sum())))
    summ = pd.DataFrame(rows)
    if verbose:
        print(f"{leg_lab} - {base_lab} | B={B} | block={block} ({block_msg})")
    return summ, dist, block


if __name__ == "__main__":
    import analysis
    master = analysis.run_all(B=20, verbose=False)["master"]
    summ, dist, block = bootstrap_pairwise(master, "ret_rr_delta", "ret_rr_spot", B=N_BOOT)
    summ.to_csv("delta_vs_spot_bootstrap.csv", index=False)
    for rk in ["full", "spot:APPRECIATION", "spot:DEPRECIATION"]:
        print(f"\n--- {rk} ---")
        print(summ[summ["regime"] == rk]
              [["metric", "point", "ci_lo", "ci_hi", "verdict"]].to_string(index=False))
    print("\nSaved: delta_vs_spot_bootstrap.csv")

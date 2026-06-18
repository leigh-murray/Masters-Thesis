"""
Bry-Boschan / Pagan-Sossounov turning-point dating for daily USDCNH spot.

Implements the procedure in Appendix B of Pagan & Sossounov (2003,
"A simple framework for analysing bull and bear markets", J. Appl. Econ. 18:23-46),
applied to daily FX spot in the style of Chen (2012, MPRA 35772).

Faithful to PS:
  - works on log spot, no smoothing (PS's explicit first deviation from Bry-Boschan);
  - initial windowed extrema (+/- w);
  - alternation enforced after EVERY operation;
  - censoring order: endpoint-buffer -> endpoint-extreme -> min CYCLE -> min PHASE
    (PS censor cycles before phases), with the short-phase amplitude override.

Two-state labelling (native to PS):
  - every dated phase is labelled APPRECIATION or DEPRECIATION by the sign of its
    log move. The magnitude/conviction dimension is captured separately by the
    EWMA volatility conditioning state and is not encoded here.
  - Days before the first dated turning point and after the last are UNCLASSIFIED.

SIGN CONVENTION: USDCNH up = CNH depreciation.
  trough -> peak  (spot rising)  = DEPRECIATION  (CNH weaker)
  peak   -> trough (spot falling) = APPRECIATION  (CNH stronger)
"""

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Baseline parameters (daily trading-day units).
# PS monthly-equity originals noted in comments for reference.
# Values are FX-recalibrated (not naive day-count rescaling) to match
# the typical duration of FX directional cycles and the 3M hedge horizon.
# See NAIVE_RESCALE below for the naive rescaling as a robustness comparison.
# -----------------------------------------------------------------------------
BASELINE = dict(
    window=14,          # w: half-window for local extrema      (PS: 8 months ~176d)
    min_phase=25,       # minimum phase, trading days            (PS: 4 months  ~88d)
    min_cycle=100,      # minimum cycle (P->P / T->T), days      (PS: 16 months ~352d)
    end_buffer=14,      # endpoint exclusion, trading days       (PS: 6 months  ~132d)
    amp_override=0.04,  # keep a sub-min_phase phase if |dln s| >= this  (PS: ~0.20)
)

# Naive day-count rescaling of PS monthly parameters to daily frequency
# (22 trading days per month). Provided as a robustness comparison only;
# the very long durations collapse a 5-year FX sample to ~2-3 phases.
NAIVE_RESCALE = dict(
    window=176,         # 8 months * 22
    min_phase=88,       # 4 months * 22
    min_cycle=352,      # 16 months * 22
    end_buffer=132,     # 6 months * 22
    amp_override=0.04,  # kept FX-specific; PS equity value (~0.20) is not meaningful for FX
)


# -----------------------------------------------------------------------------
# Core algorithm. Turning points are carried as a list of (position, kind),
# kind in {'P','T'}, always sorted by position.
# -----------------------------------------------------------------------------
def _initial_turning_points(s, w):
    """Step 1(a): strict local maxima/minima over a window of +/- w."""
    n = len(s)
    tps = []
    for t in range(w, n - w):
        left, right = s[t - w:t], s[t + 1:t + w + 1]
        if s[t] > left.max() and s[t] > right.max():
            tps.append((t, 'P'))
        elif s[t] < left.min() and s[t] < right.min():
            tps.append((t, 'T'))
    return tps


def _enforce_alternation(tps, s):
    """Step 1(b): keep the highest of adjacent peaks, the lowest of adjacent troughs."""
    tps = sorted(tps, key=lambda x: x[0])
    i = 0
    while i < len(tps) - 1:
        (pa, ka), (pb, kb) = tps[i], tps[i + 1]
        if ka == kb:
            if ka == 'P':
                drop = i if s[pa] <= s[pb] else i + 1     # keep higher peak
            else:
                drop = i if s[pa] >= s[pb] else i + 1     # keep lower trough
            tps.pop(drop)
            i = max(i - 1, 0)                              # step back to recheck
        else:
            i += 1
    return tps


def _censor_endpoint_buffer(tps, n, buf):
    """Step 2(a): drop turns within `buf` of either end."""
    return [(p, k) for (p, k) in tps if buf <= p <= n - 1 - buf]


def _censor_endpoint_extreme(tps, s, n):
    """Step 2(b): at each end, drop the first/last turn if a more-extreme value of
    the same type lies between it and that endpoint."""
    tps = sorted(tps, key=lambda x: x[0])
    if not tps:
        return tps
    p0, k0 = tps[0]
    if (k0 == 'P' and p0 > 0 and s[:p0].max() > s[p0]) or \
       (k0 == 'T' and p0 > 0 and s[:p0].min() < s[p0]):
        tps = tps[1:]
    if tps:
        pL, kL = tps[-1]
        tail = s[pL + 1:]
        if (kL == 'P' and tail.size and tail.max() > s[pL]) or \
           (kL == 'T' and tail.size and tail.min() < s[pL]):
            tps = tps[:-1]
    return tps


def _censor_min_cycle(tps, s, min_cycle):
    """Step 2(c): eliminate cycles (P->P or T->T) shorter than min_cycle, shortest
    first, by deleting the less-extreme of the two same-type turns; re-alternate."""
    while True:
        tps = _enforce_alternation(tps, s)
        viol = []
        for k in range(len(tps) - 2):
            (p0, k0), (p2, k2) = tps[k], tps[k + 2]
            if k0 == k2 and (p2 - p0) < min_cycle:
                viol.append((p2 - p0, k))
        if not viol:
            return tps
        _, k = min(viol)
        p0, k0 = tps[k]
        p2, _ = tps[k + 2]
        if k0 == 'P':
            drop = k if s[p0] <= s[p2] else k + 2          # drop lower peak
        else:
            drop = k if s[p0] >= s[p2] else k + 2          # drop higher trough
        tps.pop(drop)


def _censor_min_phase(tps, s, min_phase, amp_override):
    """Step 2(d): eliminate phases shorter than min_phase UNLESS |move| >= amp_override.
    Shortest first; eliminate a phase by removing BOTH its delimiting turns."""
    while True:
        tps = _enforce_alternation(tps, s)
        viol = []
        for k in range(len(tps) - 1):
            (p0, _), (p1, _) = tps[k], tps[k + 1]
            dur, amp = p1 - p0, abs(s[p1] - s[p0])
            if dur < min_phase and amp < amp_override:
                viol.append((dur, k))
        if not viol:
            return tps
        _, k = min(viol)
        tps.pop(k + 1)
        tps.pop(k)


def date_turning_points(s, params=BASELINE):
    """Run the full PS Appendix B procedure on a log-price array `s`."""
    p = params
    tps = _initial_turning_points(s, p['window'])               # 1(a)
    tps = _enforce_alternation(tps, s)                           # 1(b)
    tps = _censor_endpoint_buffer(tps, len(s), p['end_buffer'])  # 2(a)
    tps = _enforce_alternation(tps, s)
    tps = _censor_endpoint_extreme(tps, s, len(s))               # 2(b)
    tps = _enforce_alternation(tps, s)
    prev = None                                                  # 2(c) then 2(d) to fixed point
    while tps != prev:
        prev = list(tps)
        tps = _censor_min_cycle(tps, s, p['min_cycle'])
        tps = _censor_min_phase(tps, s, p['min_phase'], p['amp_override'])
    return _enforce_alternation(tps, s)


# -----------------------------------------------------------------------------
# Labelling: two-state {APPRECIATION, DEPRECIATION} plus UNCLASSIFIED endpoints.
# -----------------------------------------------------------------------------
def label_phases(tps, s, dates):
    """Label each phase by the sign of its log move. Two states only (native to PS).
    The magnitude dimension is left to the separate EWMA vol conditioning state."""
    rows = []
    for k in range(len(tps) - 1):
        i0, i1 = tps[k][0], tps[k + 1][0]
        move = s[i1] - s[i0]
        regime = 'DEPRECIATION' if move > 0 else 'APPRECIATION'
        rows.append(dict(
            start_date=dates[i0], end_date=dates[i1],
            start_idx=i0, end_idx=i1,
            start_type=tps[k][1], end_type=tps[k + 1][1],
            duration_days=i1 - i0,
            log_amplitude=move,
            pct_move=np.exp(move) - 1.0,
            regime=regime,
        ))
    return pd.DataFrame(rows)


def daily_regime_series(phases, dates):
    """One label per trading day; days before the first / after the last
    dated turning point are UNCLASSIFIED (endpoint buffer and open final phase)."""
    lab = pd.Series('UNCLASSIFIED', index=pd.Index(dates, name='Pricing_Date'))
    for _, r in phases.iterrows():
        lab.iloc[r['start_idx']:r['end_idx']] = r['regime']
    return lab


# -----------------------------------------------------------------------------
def run(csv_path, date_col='Pricing_Date', price_col='Current_Spot', params=BASELINE):
    df = pd.read_csv(csv_path)
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    df = df.sort_values(date_col).reset_index(drop=True)
    dates = df[date_col].to_numpy()
    s = np.log(df[price_col].to_numpy(dtype=float))

    tps = date_turning_points(s, params)
    phases = label_phases(tps, s, dates)
    daily = daily_regime_series(phases, dates)
    return df, tps, phases, daily


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/spot_price.csv'
    df, tps, phases, daily = run(path)

    print(f"Observations: {len(df)}  ({df['Pricing_Date'].iloc[0].date()} -> "
          f"{df['Pricing_Date'].iloc[-1].date()})")
    print(f"Turning points retained: {len(tps)}\n")

    show = phases.copy()
    show['start_date'] = pd.to_datetime(show['start_date']).dt.date
    show['end_date'] = pd.to_datetime(show['end_date']).dt.date
    show['log_amplitude'] = show['log_amplitude'].round(4)
    show['pct_move'] = (show['pct_move'] * 100).round(2)
    print(show[['start_date', 'end_date', 'start_type', 'end_type',
                'duration_days', 'pct_move', 'regime']].to_string(index=False))

    print("\nDaily regime counts:")
    print(daily.value_counts().to_string())

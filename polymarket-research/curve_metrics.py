"""Equity-curve "beauty" metrics for a cumulative P&L series (in dollars).

A Polymarket P&L curve has no capital base, so every metric is expressed
relative to the window's total gain or is scale-free.
"""
import numpy as np

DAY = 86400


def window_series(t, y, t0, t1):
    """Slice a daily cumulative series to [t0, t1], rebased so it starts at 0.

    If the wallet existed before t0 the base is its value at t0; if it was
    created inside the window, a zero point is prepended one day before its
    first observation (P&L before joining is zero by definition).
    """
    if len(t) == 0:
        return None, None
    before = np.where(t <= t0)[0]
    inside = (t > t0) & (t <= t1)
    if before.size:
        base_t, base_y = t0, float(y[before[-1]])
    else:
        if not inside.any():
            return None, None
        base_t, base_y = int(t[inside][0]) - DAY, 0.0
    tt = np.concatenate([[base_t], t[inside]]).astype(np.float64)
    yy = np.concatenate([[base_y], y[inside].astype(np.float64)]) - base_y
    return tt, yy


def metrics(tt, yy):
    """Return dict of smoothness metrics for a rebased cumulative series."""
    n = len(yy)
    out = {"n_days": n - 1}
    if n < 15:
        return out
    gain = yy[-1]
    out["gain"] = gain
    x = (tt - tt[0]) / DAY
    # OLS fit y = a + b x
    xm, ym = x.mean(), yy.mean()
    sxx = ((x - xm) ** 2).sum()
    b = ((x - xm) * (yy - ym)).sum() / sxx
    resid = yy - (ym + b * (x - xm))
    ss_res = (resid ** 2).sum()
    ss_tot = ((yy - ym) ** 2).sum()
    out["slope_per_day"] = b
    out["r2"] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    se_b = np.sqrt(ss_res / max(n - 2, 1) / sxx) if sxx > 0 else np.inf
    out["k_ratio"] = (b / se_b) / np.sqrt(n) if se_b > 0 else 0.0
    # drawdowns
    peak = np.maximum.accumulate(yy)
    dd = peak - yy
    out["max_dd"] = dd.max()
    g = max(gain, 1e-9)
    out["dd_ratio"] = dd.max() / g if gain > 0 else np.inf
    out["underwater_frac"] = float((dd > 0.02 * abs(gain)).mean()) if gain > 0 else 1.0
    # daily / weekly increments
    d = np.diff(yy)
    out["sharpe_ann"] = d.mean() / d.std() * np.sqrt(365) if d.std() > 0 else 0.0
    out["up_day_frac"] = float((d > 0).sum() / max((np.abs(d) > 1e-6).sum(), 1))
    w = np.diff(yy[::7]) if n > 14 else d
    act = np.abs(w) > 1.0
    out["pos_week_frac"] = float((w[act] > 0).mean()) if act.any() else 0.0
    out["n_active_weeks"] = int(act.sum())
    out["worst_week_share"] = float(-w.min() / g) if gain > 0 else np.inf
    out["max_day_share"] = float(d.max() / g) if gain > 0 else np.inf
    out["top5_day_share"] = float(np.sort(d)[-5:].sum() / g) if gain > 0 else np.inf
    out["worst_day_share"] = float(-d.min() / g) if gain > 0 else np.inf
    # second half vs first half (does it keep climbing?)
    h = n // 2
    out["first_half_gain"] = yy[h] - yy[0]
    out["second_half_gain"] = yy[-1] - yy[h]
    return out


def beauty_score(m):
    """Single number for ranking smooth, steadily rising curves (higher = nicer)."""
    if not m.get("gain") or m["gain"] <= 0:
        return -np.inf
    return (
        2.0 * m["r2"]
        + 1.0 * min(m["k_ratio"], 1.0)
        + 1.0 * m["pos_week_frac"]
        - 1.5 * min(m["dd_ratio"], 1.0)
        - 1.0 * min(m["max_day_share"], 1.0)
        + 0.25 * np.log10(max(m["gain"], 1.0)) / 6
    )


def steady_metrics(tt, yy, block_days=30):
    """Metrics that capture 'steadily and surely builds up' (vs staircases).

    - eff_start: first day the curve moved by >2% of the final gain; the climb
      is measured from there, so flat-then-surge curves get a short span.
    - month_pos_frac / max_month_share: 30-day blocks positive, and the biggest
      block's share of the gain (a staircase has one block ~ 50-100%).
    - chord_mad / chord_max: mean / max distance between the curve and the
      straight line from start to end, as a fraction of the gain.
    """
    out = {}
    gain = yy[-1] - yy[0]
    if gain <= 0 or len(yy) < 15:
        return out
    moved = np.where(np.abs(yy - yy[0]) > 0.02 * gain)[0]
    i0 = max(int(moved[0]) - 1, 0) if moved.size else 0
    t2, y2 = tt[i0:], yy[i0:] - yy[i0]
    g2 = y2[-1]
    out["eff_span_days"] = int((t2[-1] - t2[0]) // DAY)
    if len(y2) < 15 or g2 <= 0:
        return out
    x = (t2 - t2[0]) / DAY
    chord = g2 * x / max(x[-1], 1e-9)
    dev = np.abs(y2 - chord) / g2
    out["chord_mad"] = float(dev.mean())
    out["chord_max"] = float(dev.max())
    xm, ym = x.mean(), y2.mean()
    sxx = ((x - xm) ** 2).sum()
    b = ((x - xm) * (y2 - ym)).sum() / sxx
    ss_res = ((y2 - (ym + b * (x - xm))) ** 2).sum()
    ss_tot = ((y2 - ym) ** 2).sum()
    out["eff_r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    # share of the gain earned in each third of the climbing period
    cut = np.searchsorted(x, [x[-1] / 3, 2 * x[-1] / 3])
    thirds = np.diff(np.concatenate([[0.0], y2[cut], [g2]]))
    out["min_third_share"] = float(thirds.min() / g2)
    # still working lately?
    k = np.searchsorted(x, x[-1] - 90)
    out["last90_gain"] = float(g2 - y2[k])
    idx = np.searchsorted(x, np.arange(0, x[-1] + 1e-9, block_days))
    idx = np.unique(np.append(idx, len(y2) - 1))
    blocks = np.diff(y2[idx])
    if len(blocks):
        out["n_months"] = int(len(blocks))
        out["month_pos_frac"] = float((blocks > 0).mean())
        out["max_month_share"] = float(blocks.max() / g2)
        out["worst_month_share"] = float(-blocks.min() / g2)
    return out


def steady_score(m):
    """Rank score for steady climbers (higher = smoother & more consistent)."""
    need = ("eff_r2", "chord_mad", "month_pos_frac", "max_month_share", "dd_ratio", "top5_day_share")
    if any(m.get(k) is None or not np.isfinite(m.get(k)) for k in need):
        return -np.inf
    return (2.0 * m["eff_r2"] - 3.0 * m["chord_mad"] + 1.5 * m["month_pos_frac"]
            - 1.0 * min(m["max_month_share"], 1.0) - 1.5 * min(m["dd_ratio"], 1.0)
            - 0.5 * min(m["top5_day_share"], 1.0))

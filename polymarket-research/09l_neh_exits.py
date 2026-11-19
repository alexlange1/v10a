"""Step 9l: does a stop-loss help the deadline fade? And do holding-reward markets differ?"""
import pickle
import importlib
import numpy as np
import pandas as pd

neh = importlib.import_module("09f_neh_backtest")
SPLIT = neh.SPLIT
u = neh.universe().set_index("conditionId")
P = pickle.load(open(neh.OUT, "rb"))
e = pd.read_parquet("data/bt/neh_core_entries.parquet")


def with_stop(stop):
    out = []
    for r in e.itertuples(index=False):
        m = u.loc[r.conditionId]
        t, p0 = P[r.conditionId]
        p0 = p0.astype(float)
        fav0 = abs(r.mid - p0[np.searchsorted(t, r.entry_ts)]) < 1e-6 if np.searchsorted(t, r.entry_ts) < len(t) else True
        pf = p0 if fav0 else 1 - p0
        after = (t > r.entry_ts) & (t < r.close_ts - 3600)
        hit = np.where(after & (pf < stop))[0]
        if stop > 0 and hit.size:
            px = pf[hit[0]]
            sell = max(px - m.tick / 2 - 0.002, 0.0)
            sell -= m.fee_rate * sell * (1 - sell)
            out.append((sell - r.cost) / r.cost)
        else:
            out.append(r.roi)
    return np.array(out)


pd.set_option("display.width", 200)
for stop in (0, 0.90, 0.85, 0.80, 0.70, 0.50):
    roi = with_stop(stop)
    isr, oos = roi[e.entry_ts.values < SPLIT], roi[e.entry_ts.values >= SPLIT]
    print(f"stop at {stop:.2f}: ROI/trade {100 * roi.mean():6.3f}%  IS {100 * isr.mean():6.3f}%  OOS {100 * oos.mean():6.3f}%  worst {100 * roi.min():.0f}%  "
          f"p5 {100 * np.percentile(roi, 5):.1f}%")
hr = e.conditionId.map(u.holdingRewardsEnabled).fillna(False).astype(bool)
for flag in (True, False):
    g = e[hr == flag]
    print(f"holdingRewardsEnabled={flag}: n {len(g)}, win {100 * g.won.mean():.2f}%, ROI {100 * g.roi.mean():.3f}%, median hold {g.hold_d.median():.0f}d")
neg = e.conditionId.map(u.negRisk).fillna(False).astype(bool)
for flag in (True, False):
    g = e[neg == flag]
    print(f"negRisk (multi-outcome event)={flag}: n {len(g)}, win {100 * g.won.mean():.2f}%, ROI {100 * g.roi.mean():.3f}%")
vb = pd.qcut(e.volume, 4)
print(e.groupby(vb, observed=True).apply(lambda g: pd.Series({"n": len(g), "win%": 100 * g.won.mean(), "roi%": 100 * g.roi.mean()})).round(2))

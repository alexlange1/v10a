"""Step 9j: how noisy is the deadline-fade edge for a capital-limited trader?
Bootstrap one-year outcomes: draw trades (with replacement) at the rate a $10k
account can actually take them, compound, and look at the spread of results."""
import importlib
import numpy as np
import pandas as pd

bt = importlib.import_module("09c_backtest")
neh = importlib.import_module("09f_neh_backtest")
e = pd.read_parquet("data/bt/neh_core_entries.parquet")
u = neh.universe().set_index("conditionId")
tick = e.conditionId.map(u.tick)
e["roi_maker"] = (e.payoff - (e.mid - tick / 2)) / (e.mid - tick / 2)
T, curve = bt.portfolio(e, bank=10_000, frac=0.02, max_event_frac=0.05)
taken = e.sort_values("entry_ts").reset_index(drop=True)
print("all trades: mean ROI %.3f%% | portfolio took %d trades" % (100 * e.roi.mean(), len(T)))
# which trades did the portfolio take? match by entry_ts/stake order -> recompute realized mean
real = (T.payout - T.stake).sum() / T.stake.sum()
print("realized ROI on taken trades: %.3f%%" % (100 * real))
# monthly trade availability
e["month"] = pd.to_datetime(e.entry_ts, unit="s").dt.to_period("M")
print("trades per month (median):", e.groupby("month").size().median())
rng = np.random.default_rng(7)
def sim(rois, n_per_year=730, stake=0.02, years=1, reps=20000):
    # each trade risks `stake` of current equity; ~50 concurrent 25-day positions -> ~730 trades/yr
    out = np.empty(reps)
    for i in range(reps):
        r = rng.choice(rois, size=int(n_per_year * years))
        out[i] = np.prod(1 + stake * r)
    return out
for label, rois in (("taker", e.roi.values), ("maker", e.roi_maker.values)):
    for stake in (0.01, 0.02):
        o = sim(rois, stake=stake)
        print(f"{label} stake {stake:.0%}: median 1y return {100 * (np.median(o) - 1):5.1f}%  | 10th pct {100 * (np.percentile(o, 10) - 1):5.1f}%  "
              f"90th {100 * (np.percentile(o, 90) - 1):5.1f}%  | P(loss) {100 * (o < 1).mean():4.1f}%")

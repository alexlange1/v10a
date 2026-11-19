"""Step 9g: the robust cell (favourite/NO at 0.95-0.98, 7-30 days before the
deadline): category split, execution sensitivity, portfolio equity curve."""
import importlib
import numpy as np
import pandas as pd

neh = importlib.import_module("09f_neh_backtest")
bt = importlib.import_module("09c_backtest")
from curve_metrics import metrics, steady_metrics
SPLIT = neh.SPLIT

u = neh.universe()
import pickle
P = pickle.load(open(neh.OUT, "rb"))
e = neh.entries(u, P, 0.95, 0.98, 7, 30)
e.to_parquet("data/bt/neh_core_entries.parquet")
def s(g):
    a, b = g[g.entry_ts < SPLIT], g[g.entry_ts >= SPLIT]
    return pd.Series({"n": len(g), "win%": 100 * g.won.mean(), "roi%": 100 * g.roi.mean(), "IS_n": len(a), "IS_roi%": 100 * a.roi.mean() if len(a) else np.nan,
                      "OOS_n": len(b), "OOS_roi%": 100 * b.roi.mean() if len(b) else np.nan, "worst_loss$per$1": g.roi.min()})
pd.set_option("display.width", 220)
print("core cell:", len(e), "trades")
cat = e.groupby("category").apply(s).sort_values("n", ascending=False)
print(cat.round(2)); cat.to_csv("data/bt/neh_core_by_category.csv")

# execution sensitivity: taker (as tested), mid (no slippage, still fee), maker (bid = mid - half tick, no fee)
u_ix = u.set_index("conditionId")
tick = e.conditionId.map(u_ix.tick); fee_rate = e.conditionId.map(u_ix.fee_rate)
for name, fill in (("taker: mid + half tick + 0.2c, fee", e.fill), ("at mid, fee", e.mid),
                   ("maker: mid - half tick, no fee", e.mid - tick / 2)):
    fee = 0 if name.startswith("maker") else fee_rate * fill * (1 - fill)
    cost = fill + fee
    roi = (e.payoff - cost) / cost
    print(f"{name:40s} avg ROI {100 * roi.mean():6.3f}%  IS {100 * roi[e.entry_ts < SPLIT].mean():6.3f}%  OOS {100 * roi[e.entry_ts >= SPLIT].mean():6.3f}%")

# exclude categories that were negative in the first half only (choose on IS, report OOS)
is_cat = e[e.entry_ts < SPLIT].groupby("category").roi.mean()
keep = set(is_cat[is_cat > 0].index)
ek = e[e.category.isin(keep)]
print("IS-positive categories:", sorted(keep), " -> OOS ROI %.3f%% on %d trades" % (100 * ek[ek.entry_ts >= SPLIT].roi.mean(), (ek.entry_ts >= SPLIT).sum()))

# portfolio: $10k, 2% per market, 5% per event, compounding; equity marked at cost until resolution
for frac, evf in ((0.02, 0.05), (0.03, 0.06), (0.05, 0.10)):
    T, curve = bt.portfolio(e.rename(columns={}), bank=10_000, frac=frac, max_event_frac=evf)
    t = ((curve.index - pd.Timestamp(0)) // pd.Timedelta(seconds=1)).values.astype(float)
    m = metrics(t, curve.values - 10_000); m.update(steady_metrics(t, curve.values - 10_000))
    peak = np.maximum.accumulate(curve.values)
    yrs = (curve.index[-1] - curve.index[0]).days / 365
    print(f"stake {frac:.0%}/event cap {evf:.0%}: trades {len(T)}, final ${curve.iloc[-1]:,.0f}, CAGR {100 * ((curve.iloc[-1] / 1e4) ** (1 / yrs) - 1):.1f}%, "
          f"max DD {100 * ((peak - curve.values) / peak).max():.1f}%, R2 {m.get('r2', np.nan):.2f}, months up {m.get('month_pos_frac', np.nan):.0%}, span {yrs:.2f}y")
    curve.to_csv(f"data/bt/neh_curve_{int(frac * 100)}pct.csv")

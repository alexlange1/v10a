"""Step 9h: does following steady climbers work? (no look-ahead)

Wallets are selected as steady using ONLY the first half of the year. We then
copy every position they opened in the second half: enter 1 hour after their
first fill at the then-current hourly price (taker: + half tick + 0.2c, plus the
new fee), hold to resolution. Compared with the market-wide baseline."""
import glob
import importlib
import numpy as np
import pandas as pd

bt = importlib.import_module("09c_backtest")
SPLIT = 1779148800
H = 3600

S = set(pd.read_csv("data/is_steady.csv").user_id)
cl = pd.concat([pd.read_parquet(p) for p in glob.glob("data/detail/closed_*.parquet") + glob.glob("data/detail_is/closed_*.parquet")], ignore_index=True)
cl = cl[cl.proxy_wallet.isin(S) & (cl.first_entry_at >= SPLIT)].drop_duplicates(["proxy_wallet", "condition_id", "outcome_index"])
u, P = bt.load()
u = u.reset_index(drop=True)
ui = u.set_index("conditionId")
cl = cl[cl.condition_id.isin(ui.index)]
print("smart OOS positions in resolved markets with price paths:", len(cl), " wallets:", cl.proxy_wallet.nunique(), flush=True)

rows = []
for r in cl.itertuples(index=False):
    m = ui.loc[r.condition_id]
    t, p0 = P[r.condition_id]
    tgt = r.first_entry_at + H
    j = np.searchsorted(t, tgt)
    if j >= len(t) or t[j] >= m.ct - 1800 or t[j] - tgt > 6 * H:   # no price within 6h after the copy time, or market closing
        continue
    oi = int(r.outcome_index)
    px = float(p0[j]) if oi == 0 else 1 - float(p0[j])
    if px < 0.02 or px > 0.995:
        continue
    fill = min(px + m.tick / 2 + bt.EXTRA_SLIP, 0.999)
    cost = fill + m.fee_rate * fill * (1 - fill)
    payoff = m.final[oi]
    rows.append({"wallet": r.proxy_wallet, "condition_id": r.condition_id, "category": m.category, "their_avg": r.avg_price,
                 "copy_px": px, "cost": cost, "payoff": payoff, "roi": (payoff - cost) / cost, "won": payoff >= 0.99,
                 "entry_ts": int(t[j]), "close_ts": int(m.ct), "event_slug": m.event_slug})
C = pd.DataFrame(rows)
C.to_parquet("data/bt/copy_trades.parquet")
print("copyable trades:", len(C), " wallets:", C.wallet.nunique())
band = pd.cut(C.copy_px, [0, .3, .5, .7, .85, .9, .95, .98, .995], labels=["<0.30", "0.30-0.50", "0.50-0.70", "0.70-0.85", "0.85-0.90", "0.90-0.95", "0.95-0.98", "0.98-0.995"])
def s(g):
    w = g.groupby("wallet").roi.mean()  # per-wallet average, so a few hyperactive wallets can't dominate
    return pd.Series({"trades": len(g), "wallets": g.wallet.nunique(), "win%": 100 * g.won.mean(), "avg_px": g.copy_px.mean(),
                      "roi%": 100 * g.roi.mean(), "se%": 100 * g.roi.std() / np.sqrt(len(g)),
                      "median_wallet_roi%": 100 * w.median(), "wallets_positive%": 100 * (w > 0).mean()})
pd.set_option("display.width", 220)
print("\nALL:", s(C).round(3).to_dict())
print("\nby copy price band:\n", C.groupby(band, observed=True).apply(s).round(2))
print("\nby category:\n", C.groupby("category").apply(s).round(2).sort_values("trades", ascending=False))
# consensus: >=2 distinct steady wallets on the same outcome
k = C.groupby(["condition_id"]).wallet.transform("nunique")
print("\nconsensus >=2 wallets:", s(C[k >= 2]).round(3).to_dict())
print("consensus >=3 wallets:", s(C[k >= 3]).round(3).to_dict())

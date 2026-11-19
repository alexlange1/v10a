"""Step 8: where exactly do the steady 'bond' buyers make (and lose) money?

Uses their resolved positions (won/lost known from Gamma final prices) and
breaks results down by entry price, hours from entry to market close, and
category. Cost-weighted: what matters is dollars, not position counts.
"""
import numpy as np
import pandas as pd

F = pd.read_parquet("data/features_arch.parquet")
pos = pd.read_parquet("data/positions_enriched.parquet")
bond_users = set(F[F.archetype == "High-probability 'bond' buyer"].user_id)
p = pos[pos.proxy_wallet.isin(bond_users) & pos.won.notna() & (pos.cost > 0)].copy()
p["px"] = pd.cut(p.avg_price, [0, .5, .8, .9, .93, .95, .97, .98, .99, 1.0001],
                 labels=["<0.50", "0.50-0.80", "0.80-0.90", "0.90-0.93", "0.93-0.95", "0.95-0.97", "0.97-0.98", "0.98-0.99", ">=0.99"])
p["ttc"] = pd.cut(p.entry_to_close_h, [-1e9, 1, 6, 24, 72, 168, 720, 1e9],
                  labels=["<1h", "1-6h", "6-24h", "1-3d", "3-7d", "7-30d", ">30d"])


def table(g):
    cost = g.cost.sum()
    return pd.Series({
        "positions": len(g),
        "cost_$M": cost / 1e6,
        "implied_win%": 100 * (g.avg_price * g.cost).sum() / cost,
        "actual_win%(cost-wtd)": 100 * (g.won * g.cost).sum() / cost,
        "roi_%": 100 * g.realized_pnl.sum() / cost,
        "pnl_$k": g.realized_pnl.sum() / 1e3,
        "worst_loss_$k": -g.realized_pnl.min() / 1e3,
    })


pd.set_option("display.width", 250)
print("bond buyers:", len(bond_users), " resolved positions:", len(p), " cost $%.0fM" % (p.cost.sum() / 1e6))
print("\n== by entry price ==\n", p.groupby("px", observed=True).apply(table).round(2))
hp = p[p.avg_price >= 0.90]
print("\n== entry >= 0.90: by hours from entry to market close ==\n", hp.groupby("ttc", observed=True).apply(table).round(2))
print("\n== entry >= 0.90: by category ==\n", hp.groupby("category").apply(table).sort_values("cost_$M", ascending=False).round(2))
print("\n== entry >= 0.90: fee-free vs fee markets ==\n", hp.groupby(hp.fee_rate.fillna(0) > 0).apply(table).round(2))
print("\n== entry >= 0.90: outcome side ==\n", hp.groupby(hp.outcome.str.lower().isin(["no"])).apply(table).round(2))
out = {"by_price": p.groupby("px", observed=True).apply(table), "by_ttc": hp.groupby("ttc", observed=True).apply(table),
       "by_cat": hp.groupby("category").apply(table)}
for k, v in out.items():
    v.to_csv(f"data/bond_{k}.csv")

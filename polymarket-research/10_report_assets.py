"""Step 10: small JSON/CSV assets for the report (no raw data needed to view it)."""
import glob
import json
import pickle
import numpy as np
import pandas as pd
from selection import RULES, add_derived, mask
from curve_metrics import window_series, DAY

cm = add_derived(pd.read_parquet("data/curve_metrics.parquet"))
F = pd.read_parquet("data/features_arch.parquet")
sel = cm[mask(cm)].copy()
assets = {}

# funnel
fun, m = [("Wallets scanned (leaderboards)", len(cm))], pd.Series(True, index=cm.index)
labels = {"y1_gain": "Made >= $3k in the last 12 months", "y1_eff_span_days": "Climbed for >= 150 days",
          "days_since_trade": "Still trading (last 30 days)", "y1_eff_r2": "Close to a straight line (R² >= 0.8)",
          "y1_r2": "...over the whole 1Y chart too", "y1_month_pos_frac": ">= 70% of months positive",
          "y1_max_month_share": "No month > 40% of profit", "y1_dd_ratio": "Worst drawdown <= 25% of profit",
          "y1_top5_day_share": "5 best days <= 50% of profit", "y1_min_third_share": "Every third of the year added >= 10%",
          "y1_last90_gain": "Still up over the last 90 days"}
for k, v in RULES.items():
    m &= mask(cm, {k: v})
    fun.append((labels[k], int(m.sum())))
assets["funnel"] = fun

# gallery: 36 real curves (top by steadiness, spread across archetypes)
S = {}
need = set(sel.user_id)
for p in sorted(glob.glob("data/pnl/*.pkl")):
    d = pickle.load(open(p, "rb"))
    for u in need & d.keys():
        S[u] = d[u]
now = 1795046400
arch = F.set_index("user_id").archetype
gal = []
order = sel.sort_values("y1_steady", ascending=False)
picked = {}
for r in order.itertuples():
    a = arch.get(r.user_id, "?")
    if picked.get(a, 0) >= 8:
        continue
    picked[a] = picked.get(a, 0) + 1
    tt, yy = window_series(S[r.user_id]["t"], S[r.user_id]["economic_pnl"], now - 365 * DAY, now)
    idx = np.linspace(0, len(yy) - 1, min(72, len(yy))).astype(int)
    gal.append({"name": (r.user_name if isinstance(r.user_name, str) else r.user_id[:10])[:22], "gain": round(float(r.y1_gain)),
                "archetype": a, "pts": [round(float(v)) for v in yy[idx]], "start_day": int((tt[0] - (now - 365 * DAY)) // DAY)})
    if len(gal) >= 36:
        break
assets["gallery"] = gal

# archetype summary
assets["archetypes"] = pd.read_csv("data/archetype_summary.csv").to_dict(orient="records")
for k in ("by_price", "by_ttc", "by_cat"):
    assets["bond_" + k] = pd.read_csv(f"data/bond_{k}.csv").to_dict(orient="records")
assets["persistence"] = pd.read_csv("data/persistence_summary.csv").to_dict(orient="records")

# steady bond buyers to watch
b = F[F.archetype == "High-probability 'bond' buyer"].sort_values("y1_gain", ascending=False)
watch = b[["user_id", "user_name", "y1_gain", "y1_volume_usdc", "edge_bps", "win_rate", "avg_entry_price_w", "primary_cat",
           "trades_per_day", "median_pos_cost", "y1_dd_ratio"]].copy()
watch.to_csv("report/steady_bond_buyers.csv", index=False)
F[["user_id", "user_name", "archetype", "primary_cat", "y1_gain", "y1_volume_usdc", "edge_bps", "win_rate", "avg_entry_price_w",
   "trades_per_day", "median_trade_usdc", "y1_r2", "y1_dd_ratio", "y1_month_pos_frac", "lp_income_share"]].sort_values(
    "y1_gain", ascending=False).to_csv("report/steady_climbers_543.csv", index=False)
assets["watch"] = watch.head(15).to_dict(orient="records")
assets["totals"] = {"selected": len(sel), "pool": len(cm), "sum_gain": float(sel.y1_gain.sum()),
                    "sum_vol": float(sel.y1_d_volume_usdc.sum()), "median_gain": float(sel.y1_gain.median())}
json.dump(assets, open("report/assets.json", "w"), default=float)
print({k: (len(v) if isinstance(v, list) else v) for k, v in assets.items()})

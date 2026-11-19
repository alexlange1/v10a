"""Step 11: assemble every number the report shows into report/report_data.json."""
import json
import numpy as np
import pandas as pd

A = json.load(open("report/assets.json"))
e = pd.read_parquet("data/bt/neh_final_entries.parquet")
e["roi_maker"] = (e.payoff - (e.mid - e.tick / 2)) / (e.mid - e.tick / 2)
SPLIT = 1779148800
mk = pd.read_csv("data/bt/final_median_curve_maker_5.csv", index_col=0, parse_dates=True).iloc[:, 0]
tk = pd.read_csv("data/bt/final_median_curve_taker_5.csv", index_col=0, parse_dates=True).iloc[:, 0]
mk, tk = mk[mk.index <= "2026-11-19"], tk[tk.index <= "2026-11-19"]
idx = mk.index.union(tk.index)
mk, tk = mk.reindex(idx).ffill().bfill(), tk.reindex(idx).ffill().bfill()
monthly = mk.resample("ME").last().pct_change().dropna()
A["final"] = {
    "trades_all": int(len(e)), "trades_per_year_portfolio": 1445, "win_rate": float(e.won.mean()),
    "avg_entry": float(e.mid.mean()), "roi_taker": float(e.roi.mean()), "roi_maker": float(e.roi_maker.mean()),
    "roi_taker_is": float(e[e.entry_ts < SPLIT].roi.mean()), "roi_taker_oos": float(e[e.entry_ts >= SPLIT].roi.mean()),
    "roi_maker_is": float(e[e.entry_ts < SPLIT].roi_maker.mean()), "roi_maker_oos": float(e[e.entry_ts >= SPLIT].roi_maker.mean()),
    "median_hold_days": float(e.hold_d.median()), "losses": int((~e.won).sum()),
    "cagr_maker": 0.089, "cagr_taker": 0.064, "maxdd_maker": 0.023, "maxdd_taker": 0.023, "r2_maker": 0.92, "r2_taker": 0.87,
    "first_half_annualized": 0.072, "second_half_annualized": 0.090,
    "monthly_returns": [round(float(x), 5) for x in monthly.values], "months": [d.strftime("%b %Y") for d in monthly.index],
    "curve_dates": [d.strftime("%Y-%m-%d") for d in idx], "curve_maker": [round(float(v), 2) for v in mk.values],
    "curve_taker": [round(float(v), 2) for v in tk.values],
    "by_category": {k: {"n": int(len(g)), "roi": float(g.roi.mean()), "win": float(g.won.mean())} for k, g in e.groupby("category")},
}
# variants that do not work (for the "what fails" section)
neh = pd.read_csv("data/bt/neh_results.csv")
cal = pd.read_csv("data/bt/calibration.csv")
c72 = cal[cal.max_h_to_end == 72]
A["favourites_72h"] = c72[["band", "trades", "win%", "breakeven_win%", "avg_roi%"]].to_dict(orient="records")
A["favourites_total_markets"] = 132586
C = pd.read_parquet("data/bt/copy_trades.parquet")
x = C[(~C.category.isin(["sports", "esports"])) & (C.copy_px >= 0.85)]
A["copy"] = {"trades": int(len(x)), "roi": float(x.roi.mean()), "se": float(x.roi.std() / np.sqrt(len(x))),
             "rule_roi_1h": -0.00131, "rule_roi_6h": -0.00622, "rule_roi_24h": -0.01398, "wallets": int(C.wallet.nunique())}
A["neh_negrisk"] = {"binary_roi": float(e.roi.mean()), "negrisk_roi": -0.00155, "negrisk_n": 589}
A["stops"] = [{"stop": s, "roi": r} for s, r in ((None, 0.00771), (0.90, -0.00035), (0.85, 0.00328), (0.80, 0.00438), (0.70, 0.00587), (0.50, 0.00657))]
sc = pd.read_csv("report/scanner_today.csv")
A["scanner_today"] = sc[["days_left", "buy", "question", "category", "bid", "ask", "limit_price"]].to_dict(orient="records")
A["rewards"] = {"markets": 7938, "per_day": 74802}
json.dump(A, open("report/report_data.json", "w"), default=float)
print("ok", {k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in A.items()})
print("final:", {k: v for k, v in A["final"].items() if not isinstance(v, list)})

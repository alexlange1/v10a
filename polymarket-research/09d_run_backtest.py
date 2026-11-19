"""Step 9d: run the market-level backtest: calibration grid, rule variants,
first-half/second-half stability, and a $10k portfolio simulation."""
import json
import importlib
import numpy as np
import pandas as pd

bt = importlib.import_module("09c_backtest")
from curve_metrics import metrics, steady_metrics

LOW_FEE = {"geopolitics", "weather", "economics", "culture", "mentions", "politics", "crypto"}
SPLIT = 1779148800  # 2026-05-19: first half vs second half of the year


def summarize(e, name):
    if not len(e):
        return {"variant": name, "trades": 0}
    h1, h2 = e[e.entry_ts < SPLIT], e[e.entry_ts >= SPLIT]
    return {"variant": name, "trades": len(e), "win%": 100 * e.won.mean(), "avg_fill": e.fill.mean(),
            "avg_roi%": 100 * e.roi.mean(), "median_hold_h": e.hold_h.median(),
            "roi%_first_half": 100 * h1.roi.mean() if len(h1) else np.nan,
            "roi%_second_half": 100 * h2.roi.mean() if len(h2) else np.nan,
            "worst_loss%": 100 * e.roi.min(), "losses": int((e.payoff < e.cost).sum())}


def curve_stats(curve, bank):
    t = ((curve.index - pd.Timestamp(0)) // pd.Timedelta(seconds=1)).values.astype(float)
    y = curve.values - bank
    m = metrics(t, y)
    m.update(steady_metrics(t, y))
    yrs = (curve.index[-1] - curve.index[0]).days / 365
    peak = np.maximum.accumulate(curve.values)
    return {"final": curve.iloc[-1], "total_return%": 100 * (curve.iloc[-1] / bank - 1),
            "cagr%": 100 * ((curve.iloc[-1] / bank) ** (1 / max(yrs, 1e-9)) - 1),
            "max_dd%": 100 * ((peak - curve.values) / peak).max(), "r2": m.get("r2"),
            "month_pos_frac": m.get("month_pos_frac"), "days": len(curve)}


def main():
    u, P = bt.load()
    print("markets with usable price paths:", len(u), flush=True)
    print(u.category.value_counts().to_dict(), flush=True)
    bands = [(0.80, 0.85), (0.85, 0.90), (0.90, 0.93), (0.93, 0.95), (0.95, 0.97), (0.97, 0.98), (0.98, 0.99), (0.99, 0.995)]
    C = bt.calibration(u, P, bands, [6, 24, 72, 168, 336])
    C.to_csv("data/bt/calibration.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(C.round(3).to_string(), flush=True)

    variants = {
        "V1 all cats 0.90-0.97 <=72h": (u, 0.90, 0.97, 72),
        "V2 low-fee cats 0.93-0.98 <=72h": (u[u.category.isin(LOW_FEE)], 0.93, 0.98, 72),
        "V3 all cats 0.95-0.99 <=24h": (u, 0.95, 0.99, 24),
        "V4 all cats 0.93-0.98 <=168h": (u, 0.93, 0.98, 168),
        "V5 all cats 0.85-0.93 <=72h": (u, 0.85, 0.93, 72),
        "V6 sports 0.90-0.97 <=6h": (u[u.category == "sports"], 0.90, 0.97, 6),
        "V7 low-fee cats 0.90-0.97 <=168h": (u[u.category.isin(LOW_FEE)], 0.90, 0.97, 168),
    }
    rows, E = [], {}
    for name, (uu, lo, hi, hz) in variants.items():
        e = bt.entries(uu, P, lo, hi, hz)
        E[name] = e
        rows.append(summarize(e, name))
        e.to_parquet(f"data/bt/entries_{name.split()[0]}.parquet")
    V = pd.DataFrame(rows)
    V.to_csv("data/bt/variants.csv", index=False)
    print(V.round(3).to_string(), flush=True)

    # category breakdown for the broad variant
    e = E["V4 all cats 0.93-0.98 <=168h"]
    cb = e.groupby("category").apply(lambda g: pd.Series({"trades": len(g), "win%": 100 * g.won.mean(), "avg_fill": g.fill.mean(),
                                                         "avg_roi%": 100 * g.roi.mean(), "fee_bps": 1e4 * (g.fee / g.cost).mean()}))
    cb.to_csv("data/bt/by_category.csv")
    print(cb.round(3).sort_values("trades", ascending=False).to_string(), flush=True)

    out = {}
    for name in variants:
        if len(E[name]) < 50:
            continue
        T, curve = bt.portfolio(E[name], bank=10_000, frac=0.02, max_event_frac=0.05)
        cs = curve_stats(curve, 10_000)
        cs.update(trades_taken=len(T), avg_stake=T.stake.mean())
        out[name] = cs
        curve.to_csv(f"data/bt/curve_{name.split()[0]}.csv")
    print(pd.DataFrame(out).T.round(3).to_string(), flush=True)
    pd.DataFrame(out).T.to_csv("data/bt/portfolio.csv")


if __name__ == "__main__":
    main()

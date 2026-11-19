"""Step 3: score every wallet's P&L curve and pick the 'beautiful chart' traders.

Windows (now = last daily point, 2026-11-19):
  y1   : last 365 days (the 1Y chart in the screenshot)
  is_  : in-sample half  [now-365d, now-183d]   (for persistence test)
  oos  : out-of-sample   [now-183d, now]
"""
import glob
import pickle
import numpy as np
import pandas as pd
from curve_metrics import window_series, metrics, beauty_score, steady_metrics, steady_score, DAY

SERIES = "economic_pnl"  # position P&L incl. unrealized + rebates/rewards/yield


def load_all():
    data = {}
    for p in sorted(glob.glob("data/pnl/*.pkl")):
        with open(p, "rb") as f:
            data.update(pickle.load(f))
    return data


def delta(rec, field, t0, t1):
    t = rec["t"]
    v = rec[field]
    i1 = np.where(t <= t1)[0]
    if not i1.size:
        return 0.0
    i0 = np.where(t <= t0)[0]
    return float(v[i1[-1]]) - (float(v[i0[-1]]) if i0.size else 0.0)


def main():
    data = load_all()
    now = max(int(r["t"][-1]) for r in data.values() if "t" in r and len(r["t"]))
    wins = {"y1": (now - 365 * DAY, now), "is": (now - 365 * DAY, now - 183 * DAY), "oos": (now - 183 * DAY, now)}
    rows = []
    for u, rec in data.items():
        if "error" in rec or len(rec.get("t", [])) == 0:
            rows.append({"user_id": u, "status": "error" if "error" in rec else "empty"})
            continue
        t = rec["t"]
        row = {"user_id": u, "status": "ok", "first_ts": int(t[0]), "last_ts": int(t[-1]),
               "hist_days": int((t[-1] - t[0]) // DAY) + 1,
               "alltime_econ_pnl": float(rec["economic_pnl"][-1]),
               "alltime_pos_pnl": float(rec["position_pnl"][-1]),
               "alltime_volume": float(rec["volume"][-1]),
               "alltime_volume_usdc": float(rec["volume_usdc"][-1]),
               "alltime_trades": float(rec["trade_count"][-1])}
        tc = rec["trade_count"]
        chg = np.where(np.diff(tc) > 0)[0]
        row["last_trade_ts"] = int(t[chg[-1] + 1]) if chg.size else int(t[0])
        for w, (t0, t1) in wins.items():
            tt, yy = window_series(t, rec[SERIES], t0, t1)
            if tt is None:
                continue
            m = metrics(tt, yy)
            m.update(steady_metrics(tt, yy))
            m["steady"] = steady_score(m)
            m["span_days"] = int((tt[-1] - tt[0]) // DAY)
            for f in ("volume", "volume_usdc", "trade_count", "wallet_income", "fees_paid", "maker_rebate",
                      "taker_rebate", "reward_income", "yield_income", "position_pnl", "realized_pnl"):
                m["d_" + f] = delta(rec, f, t0, t1)
            m["beauty"] = beauty_score(m)
            for k, v in m.items():
                row[f"{w}_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    cand = pd.concat([pd.read_csv("data/candidates.csv"), pd.read_csv("data/candidates_ext.csv")], ignore_index=True).drop_duplicates("user_id")
    df = df.merge(cand, on="user_id", how="left")
    df.to_parquet("data/curve_metrics.parquet")
    print(df["status"].value_counts())
    print("now =", pd.to_datetime(now, unit="s"))


if __name__ == "__main__":
    main()

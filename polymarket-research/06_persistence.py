"""Step 6: do beautiful charts persist? (out-of-sample test)

Select steady climbers using ONLY the first half of the year
(IS = [now-365d, now-183d]) and look at what they did in the second half
(OOS = [now-183d, now]). Compare with traders who were just as profitable in
IS but whose curves were not steady.

Bias note: the wallet pool comes from today's leaderboards, which partly
depend on OOS results. The test is therefore also run on the sub-universe
drawn from the all-time VOLUME leaderboard (not ranked on profit).
"""
import glob
import pickle
import numpy as np
import pandas as pd
from curve_metrics import window_series, metrics, steady_metrics, DAY

IS_RULES = dict(gain=1500, span=90, r2=0.80, mpos=0.70, mmax=0.50, dd=0.25, top5=0.50, third=0.10)


def steady(m, R):
    try:
        return (m["gain"] >= R["gain"] and m["eff_span_days"] >= R["span"] and m["eff_r2"] >= R["r2"]
                and m["month_pos_frac"] >= R["mpos"] and m["max_month_share"] <= R["mmax"]
                and m["dd_ratio"] <= R["dd"] and m["top5_day_share"] <= R["top5"] and m["min_third_share"] >= R["third"])
    except KeyError:
        return False


def main():
    cand = pd.concat([pd.read_csv("data/candidates.csv"), pd.read_csv("data/candidates_ext.csv")]).drop_duplicates("user_id").set_index("user_id")
    rows = []
    now = None
    for p in sorted(glob.glob("data/pnl/*.pkl")):
        with open(p, "rb") as f:
            d = pickle.load(f)
        for u, rec in d.items():
            if "t" not in rec or len(rec["t"]) < 30:
                continue
            if now is None:
                now = 1795046400  # 2026-11-19 00:00 UTC (last daily point)
            t, y = rec["t"], rec["economic_pnl"]
            is0, is1, oos1 = now - 365 * DAY, now - 183 * DAY, now
            tt, yy = window_series(t, y, is0, is1)
            if tt is None or len(tt) < 30:
                continue
            m = metrics(tt, yy); m.update(steady_metrics(tt, yy))
            tc = rec["trade_count"]
            def tc_at(ts):
                i = np.where(t <= ts)[0]
                return float(tc[i[-1]]) if i.size else 0.0
            active_end_is = tc_at(is1) - tc_at(is1 - 30 * DAY) > 0
            to, yo = window_series(t, y, is1, oos1)
            mo = {}
            if to is not None and len(to) >= 30:
                mo = metrics(to, yo); mo.update(steady_metrics(to, yo))
            rows.append({
                "user_id": u, "is_gain": m.get("gain", np.nan), "is_steady": steady(m, IS_RULES),
                "active_end_is": active_end_is,
                "oos_gain": mo.get("gain", 0.0) if mo else 0.0,
                "oos_steady": steady(mo, IS_RULES) if mo else False,
                "oos_trades": tc_at(oos1) - tc_at(is1),
                "oos_dd_ratio": mo.get("dd_ratio", np.nan), "oos_r2": mo.get("eff_r2", np.nan),
                "is_vol": float(rec["volume_usdc"][np.where(t <= is1)[0][-1]] - (rec["volume_usdc"][np.where(t <= is0)[0][-1]] if (t <= is0).any() else 0)) if (t <= is1).any() else 0.0,
            })
    df = pd.DataFrame(rows)
    df["lists"] = df.user_id.map(cand.lists)
    df.to_parquet("data/persistence.parquet")

    def summarize(d, label):
        d = d[(d.is_gain >= IS_RULES["gain"]) & d.active_end_is]
        out = []
        for name, g in (("steady in IS", d[d.is_steady]), ("profitable but not steady in IS", d[~d.is_steady])):
            act = g[g.oos_trades > 0]
            out.append({
                "universe": label, "group": name, "n": len(g),
                "still_trading_oos": (g.oos_trades > 0).mean(),
                "oos_profitable": (act.oos_gain > 0).mean(),
                "oos_steady": act.oos_steady.mean(),
                "median_oos_gain": act.oos_gain.median(),
                "median_oos_over_is": (act.oos_gain / act.is_gain).median(),
                "oos_lost_more_than_is_gain": (act.oos_gain < -act.is_gain).mean(),
            })
        return out
    res = summarize(df, "all candidates") + summarize(df[df.lists.fillna("").str.contains("VOLUME")], "volume-leaderboard only")
    R = pd.DataFrame(res)
    R.to_csv("data/persistence_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(R.round(3))


if __name__ == "__main__":
    main()

"""Step 9e: which observable conditions separate favourites that are safe to buy
from ones that are not? Rules are chosen on the first half of the year (IS) and
checked on the second half (OOS), so the final rule is not fit to the test data."""
import importlib
import numpy as np
import pandas as pd

bt = importlib.import_module("09c_backtest")
SPLIT = 1779148800  # 2026-05-19
LOW_FEE = {"geopolitics", "weather", "economics", "culture", "mentions", "politics", "crypto"}


def stats(e):
    if not len(e):
        return dict(n=0)
    return dict(n=len(e), win=100 * e.won.mean(), fill=e.fill.mean(), roi=100 * e.roi.mean(),
                se=100 * e.roi.std() / np.sqrt(len(e)), hold_h=e.hold_h.median())


def split(e):
    a, b = e[e.entry_ts < SPLIT], e[e.entry_ts >= SPLIT]
    sa, sb = stats(a), stats(b)
    return {**{f"IS_{k}": v for k, v in sa.items()}, **{f"OOS_{k}": v for k, v in sb.items()}}


COND = {
    "band only": None,
    "never above 0.985 before (not a fallen favourite)": lambda b: b.fav_max_before < 0.985,
    "rising/flat last 24h": lambda b: b.fav_chg_24h >= 0,
    "rising/flat 24h & never above 0.985": lambda b: (b.fav_chg_24h >= 0) & (b.fav_max_before < 0.985),
    "calm (avg hourly move <= 0.5c)": lambda b: b.vol_24h <= 0.005,
    "calm & rising/flat 24h": lambda b: (b.vol_24h <= 0.005) & (b.fav_chg_24h >= 0),
    "event still ahead (0-72h to scheduled end)": lambda b: b.h_end >= 0,
    "past scheduled end (awaiting resolution)": lambda b: b.h_end < 0,
    "ahead & calm & rising/flat": lambda b: (b.h_end >= 0) & (b.vol_24h <= 0.005) & (b.fav_chg_24h >= 0),
}


def main():
    u, P = bt.load()
    u = u.reset_index(drop=True)
    print("markets:", len(u), flush=True)
    rows = []
    for (lo, hi) in [(0.90, 0.95), (0.95, 0.98), (0.98, 0.995)]:
        for hz in (24, 72):
            for cname, c in COND.items():
                for scope, mk in (("all", None), ("low-fee", u.category.isin(LOW_FEE).values), ("sports+esports", u.category.isin({"sports", "esports"}).values)):
                    e = bt.entries(u, P, lo, hi, hz, cond=c, markets=mk)
                    rows.append({"band": f"{lo:.2f}-{hi:.3f}", "h": hz, "cond": cname, "scope": scope, **split(e)})
            print(f"done band {lo}-{hi} h<={hz}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv("data/bt/conditions.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 20); pd.set_option("display.max_rows", 400)
    cols = ["band", "h", "scope", "cond", "IS_n", "IS_win", "IS_roi", "IS_se", "OOS_n", "OOS_win", "OOS_roi", "OOS_se"]
    print(R[cols].round(2).to_string())


if __name__ == "__main__":
    main()

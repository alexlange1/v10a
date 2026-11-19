"""Step 9f: 'Nothing ever happens' backtest: buy the favourite (usually NO) on
deadline markets ('Will X happen by <date>?') weeks before the deadline and hold
to resolution. Daily price paths over each market's whole life (interval=max)."""
import os
import re
import sys
import json
import time
import pickle
import importlib
import concurrent.futures as cf
import numpy as np
import pandas as pd
import pm_api as A

bt = importlib.import_module("09c_backtest")
DEADLINE = re.compile(r"\bby\b|\bbefore\b|\bin 20\d\d\b|\bthis year\b|\bend of\b", re.I)
CATS = {"politics", "politics_election", "geopolitics", "economics", "tech", "culture", "finance", "crypto", "mentions"}
OUT = "data/bt/neh_prices.pkl"
SPLIT = 1779148800  # 2026-05-19


def universe():
    u = pd.read_parquet("data/bt/universe.parquet")
    u["category"] = [bt.category(a, b, c) for a, b, c in zip(u.tags, u.question, u.slug)]
    u = u[u.category.isin(CATS) & u.question.fillna("").str.contains(DEADLINE)]
    u = u[(u.volumeNum >= 25000) & u.clobTokenIds.notna()].copy()
    u["fee_rate"] = u.category.map(bt.FEE_RATE).fillna(0.05)
    u["ct"] = bt.to_ts(u.closedTime)
    u["et"] = bt.to_ts(u.endDate)
    u["final"] = u.outcomePrices.map(lambda s: [float(x) for x in json.loads(s)] if s else None)
    u["tick"] = pd.to_numeric(u.orderPriceMinTickSize, errors="coerce").fillna(0.01)
    u = u[u.final.map(lambda f: f is not None and len(f) == 2 and abs(sum(f) - 1) < 0.02)]
    return u.reset_index(drop=True)


def fetch(u):
    P = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}
    todo = [(c, json.loads(t)[0]) for c, t in zip(u.conditionId, u.clobTokenIds) if c not in P]
    def one(a):
        cid, tok = a
        for i in range(6):
            try:
                h = A.prices_history(tok, interval="max", fidelity=1440)
                return cid, (np.array([x["t"] for x in h], dtype=np.int64), np.array([x["p"] for x in h], dtype=np.float32))
            except A.ApiError as e:
                time.sleep(5 * (i + 1))
        return cid, None
    with cf.ThreadPoolExecutor(32) as ex:
        for i, (cid, arr) in enumerate(ex.map(one, todo)):
            P[cid] = arr
    pickle.dump(P, open(OUT, "wb"))
    return P


def entries(u, P, lo, hi, dmin, dmax, side_filter=None):
    """First daily bar with favourite price in [lo, hi) and dmin..dmax days before the scheduled end."""
    out = []
    for r in u.itertuples(index=False):
        a = P.get(r.conditionId)
        if a is None or len(a[0]) == 0:
            continue
        t, p0 = a
        ok = t < r.ct - 3600
        t, p0 = t[ok], p0[ok].astype(float)
        if not len(t):
            continue
        side = np.where(p0 >= 0.5, 0, 1)
        pf = np.where(side == 0, p0, 1 - p0)
        d_end = (r.et - t) / 86400
        m = (pf >= lo) & (pf < hi) & (d_end >= dmin) & (d_end <= dmax)
        if side_filter is not None:
            outs = json.loads(r.outcomes) if r.outcomes else ["Yes", "No"]
            names = np.array([outs[s].lower() if s < len(outs) else "" for s in side])
            m &= names == side_filter
        if not m.any():
            continue
        i = int(np.argmax(m))
        fill = min(pf[i] + r.tick / 2 + bt.EXTRA_SLIP, 0.999)
        fee = r.fee_rate * fill * (1 - fill)
        cost = fill + fee
        payoff = r.final[side[i]]
        out.append({"conditionId": r.conditionId, "question": r.question, "category": r.category, "event_slug": r.event_slug,
                    "entry_ts": int(t[i]), "close_ts": int(r.ct), "d_to_end": d_end[i], "hold_d": (r.ct - t[i]) / 86400,
                    "mid": pf[i], "fill": fill, "cost": cost, "payoff": payoff, "won": payoff >= 0.99,
                    "roi": (payoff - cost) / cost, "volume": r.volumeNum})
    return pd.DataFrame(out)


def main():
    u = universe()
    print("deadline markets:", len(u), u.category.value_counts().to_dict(), flush=True)
    P = fetch(u)
    print("price paths:", sum(1 for v in P.values() if v is not None and len(v[0])), flush=True)
    rows = []
    for lo, hi in [(0.80, 0.90), (0.90, 0.95), (0.95, 0.98), (0.98, 0.995)]:
        for dmin, dmax in [(7, 30), (30, 90), (7, 90)]:
            for sf in (None, "no"):
                e = entries(u, P, lo, hi, dmin, dmax, sf)
                if not len(e):
                    continue
                a, b = e[e.entry_ts < SPLIT], e[e.entry_ts >= SPLIT]
                rows.append({"band": f"{lo:.2f}-{hi:.3f}", "days_before": f"{dmin}-{dmax}", "side": sf or "favourite",
                             "n": len(e), "win%": 100 * e.won.mean(), "avg_cost": e.cost.mean(), "roi%": 100 * e.roi.mean(),
                             "se%": 100 * e.roi.std() / np.sqrt(len(e)), "median_hold_d": e.hold_d.median(),
                             "IS_n": len(a), "IS_roi%": 100 * a.roi.mean() if len(a) else np.nan,
                             "OOS_n": len(b), "OOS_roi%": 100 * b.roi.mean() if len(b) else np.nan,
                             "ann_roi%_simple": 100 * e.roi.mean() * 365 / max(e.hold_d.median(), 1)})
                if lo == 0.90 and dmin == 7 and dmax == 90 and sf == "no":
                    e.to_parquet("data/bt/neh_entries.parquet")
    R = pd.DataFrame(rows)
    R.to_csv("data/bt/neh_results.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(R.round(3).to_string())
    e = pd.read_parquet("data/bt/neh_entries.parquet")
    print("\nby category (NO @0.90-0.95, 7-90d):")
    print(e.groupby("category").apply(lambda g: pd.Series({"n": len(g), "win%": 100 * g.won.mean(), "roi%": 100 * g.roi.mean()})).round(2))


if __name__ == "__main__":
    main()

"""Step 9k: snapshot of liquidity rewards (unchanged by the Nov-20 fee overhaul).
Estimate what a small two-sided quote would earn per day per $1,000 in each
reward market, using the live order book to measure competing qualifying depth.
Scoring (docs): order within max_spread v (cents) of the adjusted mid at
distance s scores ((v-s)/v)^2 * size; a two-sided quote scores min(bid, ask)."""
import json
import time
import concurrent.futures as cf
import numpy as np
import pandas as pd
import pm_api as A

rows, cur = [], None
while True:
    p = {"next_cursor": cur} if cur else {}
    d = A.get(A.CLOB + "rewards/markets/current", p)
    rows += d.get("data", [])
    cur = d.get("next_cursor")
    if not cur or cur == "LTE=" or not d.get("data"):
        break
R = pd.DataFrame(rows)
R = R[R.total_daily_rate > 0].copy()
print("reward markets:", len(R), " total $/day:", round(R.total_daily_rate.sum()))
print("daily rate quantiles:", R.total_daily_rate.quantile([.5, .75, .9, .99]).round(1).to_dict())

# market metadata + order books for the 600 richest pools
top = R.sort_values("total_daily_rate", ascending=False).head(600)
mk = {m["conditionId"]: m for m in A.gamma_markets(top.condition_id.tolist())}


def score_side(levels, mid, v, bid):
    sc = 0.0
    for lv in levels:
        px, sz = float(lv["price"]), float(lv["size"])
        s = (mid - px if bid else px - mid) * 100
        if 0 <= s < v:
            sc += sz * ((v - s) / v) ** 2
    return sc


def one(r):
    m = mk.get(r.condition_id)
    if not m or m.get("closed"):
        return None
    tok = json.loads(m["clobTokenIds"])[0]
    try:
        b = A.get(A.CLOB + "book", {"token_id": tok})
    except A.ApiError:
        return None
    bids, asks = b.get("bids", []), b.get("asks", [])
    if not bids or not asks:
        return None
    bb, ba = max(float(x["price"]) for x in bids), min(float(x["price"]) for x in asks)
    mid = (bb + ba) / 2
    v = float(r.rewards_max_spread)
    comp = min(score_side(bids, mid, v, True), score_side(asks, mid, v, False))
    tick = float(m.get("orderPriceMinTickSize") or 0.01)
    # our quote: N shares each side, one tick inside the max spread from mid (realistic, not at mid)
    s = min(max(tick * 100, 0.5), v * 0.5)
    N = max(1000.0, float(r.rewards_min_size))
    mine = N * ((v - s) / v) ** 2
    if not (0.10 <= mid <= 0.90):
        pass  # outside 0.10-0.90 only two-sided quotes score (we model two-sided anyway)
    share = mine / (mine + comp)
    daily = r.total_daily_rate * share
    end = pd.to_datetime(m.get("endDate"), utc=True, errors="coerce")
    return {"condition_id": r.condition_id, "question": m.get("question"), "rate_per_day": r.total_daily_rate,
            "min_size": r.rewards_min_size, "max_spread_c": v, "mid": mid, "spread_c": (ba - bb) * 100,
            "competing_score": comp, "est_daily_per_1k": daily * 1000 / N, "days_left": (end - pd.Timestamp.now(tz="UTC")).days if pd.notna(end) else np.nan,
            "tags_hint": m.get("slug")}


with cf.ThreadPoolExecutor(16) as ex:
    out = [x for x in ex.map(one, top.itertuples(index=False)) if x]
B = pd.DataFrame(out)
B["apr_est%"] = B.est_daily_per_1k / 1000 * 365 * 100
B.to_csv("data/bt/rewards_snapshot.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print("markets with books:", len(B))
print("est. reward per $1k quoted per day: quantiles", B.est_daily_per_1k.quantile([.25, .5, .75, .9]).round(2).to_dict())
calm = B[(B.mid >= 0.03) & (B.mid <= 0.97) & (B.spread_c <= 2)]
print(B.sort_values("est_daily_per_1k", ascending=False)[["question", "rate_per_day", "max_spread_c", "mid", "spread_c", "est_daily_per_1k", "apr_est%", "days_left"]].head(20).round(3).to_string())

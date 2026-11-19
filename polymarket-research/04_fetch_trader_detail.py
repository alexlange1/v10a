"""Step 4: pull trading detail for the selected 'beautiful chart' wallets.

For each wallet: user-stats, most recent closed positions (sorted by time),
open positions, last N trades and recent activity (to count merges, splits,
redemptions, reward/rebate payouts). Then Gamma metadata (tags, fee type,
reward/holding-reward flags, end dates, outcomes) for every market touched.
Writes parquet files in data/detail/.
"""
import os
import sys
import time
import json
import concurrent.futures as cf
import pandas as pd
import pm_api as A

import os as _os
OUT = _os.environ.get("DETAIL_OUT", "data/detail")
N_CLOSED, N_OPEN, N_TRADES, N_ACT = 3000, 1000, 3000, 1000
POS_COLS = ["proxy_wallet", "token_id", "condition_id", "current_size", "avg_price", "entry_cost_usdc",
            "entry_fees_usdc", "total_cost_usdc", "current_price", "current_value", "total_size", "realized_pnl",
            "unrealized_pnl", "total_pnl", "status", "redeemable", "mergeable", "negative_risk", "title",
            "event_slug", "outcome", "outcome_index", "end_date", "last_event_at", "first_entry_at"]
TRADE_COLS = ["proxy_wallet", "side", "token_id", "condition_id", "size", "price", "timestamp", "title",
              "event_slug", "outcome", "outcome_index", "transaction_hash"]
ACT_COLS = ["proxy_wallet", "timestamp", "condition_id", "type", "size", "usdc_size", "price", "side",
            "outcome_index", "title", "event_slug", "transaction_hash", "token_id"]


def trim(rows, cols):
    return [{c: r.get(c) for c in cols} for r in rows]


def fetch_wallet(u):
    out = {"user": u}
    try:
        out["stats"] = A.user_stats(u)
        out["closed"] = trim(A.positions(u, status="closed", sort_by="TIMESTAMP", max_items=N_CLOSED), POS_COLS)
        out["open"] = trim(A.positions(u, status="open", max_items=N_OPEN), POS_COLS)
        out["trades"] = trim(A.trades(u, max_items=N_TRADES), TRADE_COLS)
        out["activity"] = trim(A.activity(u, max_items=N_ACT), ACT_COLS)
    except A.ApiError as e:
        out["error"] = str(e)[:300]
    return out


def main(sel_path="data/selected.csv"):
    os.makedirs(OUT, exist_ok=True)
    users = pd.read_csv(sel_path)["user_id"].tolist()
    done = set()
    if os.path.exists(f"{OUT}/done.txt"):
        done = set(open(f"{OUT}/done.txt").read().split())
    todo = [u for u in users if u not in done]
    print(f"{len(todo)} wallets to fetch", flush=True)
    t0 = time.time()
    B = 100
    for bi in range(0, len(todo), B):
        batch = todo[bi:bi + B]
        res = []
        with cf.ThreadPoolExecutor(16) as ex:
            res = list(ex.map(fetch_wallet, batch))
        tag = f"{int(time.time() * 1000)}"
        stats, closed, opn, trd, act = [], [], [], [], []
        for r in res:
            if "error" in r:
                print("ERR", r["user"], r["error"][:120], flush=True)
            s = r.get("stats") or {}
            atp = s.pop("all_time_pnl", None) or {}
            s = {k: v for k, v in s.items() if not isinstance(v, (dict, list))}
            s["user"] = r["user"]
            s["error"] = r.get("error")
            stats.append(s)
            closed += r.get("closed", [])
            opn += r.get("open", [])
            trd += r.get("trades", [])
            act += r.get("activity", [])
        for name, rows in (("stats", stats), ("closed", closed), ("open", opn), ("trades", trd), ("activity", act)):
            if rows:
                pd.DataFrame(rows).to_parquet(f"{OUT}/{name}_{tag}.parquet")
        with open(f"{OUT}/done.txt", "a") as f:
            f.write("\n".join(r["user"] for r in res if "error" not in r) + "\n")
        print(f"{bi + len(batch)}/{len(todo)} wallets, {time.time() - t0:.0f}s", flush=True)


def fetch_markets(closed=False):
    """Gamma metadata for every condition id seen in positions/trades.

    Gamma hides resolved markets unless closed=true is passed, so this is run
    twice: once for open markets and once (closed=True) for the rest."""
    import glob
    cids = set()
    for pat in ("closed_*", "open_*", "trades_*"):
        for p in glob.glob(f"{OUT}/{pat}.parquet"):
            cids.update(pd.read_parquet(p, columns=["condition_id"])["condition_id"].dropna().unique())
    have = set()
    mpath = f"{OUT}/markets.parquet"
    if os.path.exists(mpath):
        have = set(pd.read_parquet(mpath, columns=["conditionId"])["conditionId"])
    todo = sorted(c for c in cids if c not in have)
    print(f"{len(cids)} markets seen, {len(todo)} to fetch", flush=True)
    keep = ["conditionId", "question", "slug", "endDate", "closedTime", "startDate", "createdAt", "outcomes",
            "outcomePrices", "clobTokenIds", "negRisk", "feesEnabled", "feeType", "feeSchedule",
            "holdingRewardsEnabled", "rewardsMinSize", "rewardsMaxSpread", "sportsMarketType", "gameStartTime",
            "volumeNum", "liquidityNum", "umaResolutionStatus", "closed", "events", "tags", "orderPriceMinTickSize"]

    def one(chunk):
        params = [("condition_ids", c) for c in chunk] + [("limit", 100), ("include_tag", "true")]
        if closed:
            params.append(("closed", "true"))
        try:
            d = A.get(A.GAMMA + "markets", params)
        except A.ApiError as e:
            print("ERR", str(e)[:200], flush=True)
            return []
        rows = []
        for m in d:
            r = {k: m.get(k) for k in keep}
            ev = (m.get("events") or [{}])[0]
            r["event_slug"] = ev.get("slug")
            r["event_title"] = ev.get("title")
            r["series_slug"] = ev.get("seriesSlug")
            r["tags"] = "|".join(t.get("slug", "") for t in (m.get("tags") or []))
            r["feeSchedule"] = json.dumps(m.get("feeSchedule")) if m.get("feeSchedule") else None
            del r["events"]
            rows.append(r)
        return rows

    chunks = [todo[i:i + 40] for i in range(0, len(todo), 40)]
    rows = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(16) as ex:
        for i, rr in enumerate(ex.map(one, chunks)):
            rows += rr
            if i % 200 == 0:
                print(f"  {i}/{len(chunks)} chunks {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    if os.path.exists(mpath):
        df = pd.concat([pd.read_parquet(mpath), df], ignore_index=True)
    for c in ("rewardsMinSize", "rewardsMaxSpread", "volumeNum", "liquidityNum", "orderPriceMinTickSize"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.drop_duplicates("conditionId").to_parquet(mpath)
    print(f"markets saved: {len(df)}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "markets":
        fetch_markets(closed=False)
        fetch_markets(closed=True)
    else:
        main(*(sys.argv[1:2]))

"""Step 1: build a large candidate pool of wallets from Polymarket leaderboards.

Lists pulled (Data API v2 /leaderboard):
  overall  x {all, month, week} x PNL      -- profitable traders, long + recent
  overall  x {all, month} x VOLUME         -- high-volume (market makers / grinders)
  10 categories x {all, month} x PNL       -- specialists (weather, mentions, ...)
Output: data/candidates.csv (one row per wallet, with the lists it appeared on).
"""
import sys
import time
import concurrent.futures as cf
import pandas as pd
import pm_api as A

CATS = ["sports", "politics", "crypto", "culture", "weather", "economics", "tech", "finance", "mentions", "esports"]
JOBS = [
    ("overall", "all", "PNL", 30000),
    ("overall", "month", "PNL", 10000),
    ("overall", "week", "PNL", 5000),
    ("overall", "all", "VOLUME", 10000),
    ("overall", "month", "VOLUME", 5000),
] + [(c, "all", "PNL", 3000) for c in CATS] + [(c, "month", "PNL", 1000) for c in CATS]


def run(job):
    cat, tp, sb, n = job
    rows = A.leaderboard(time_period=tp, sort_by=sb, category=cat, max_items=n)
    for r in rows:
        r["list"] = f"{cat}/{tp}/{sb}"
    return job, rows


def main():
    t0 = time.time()
    allrows = []
    with cf.ThreadPoolExecutor(8) as ex:
        for job, rows in ex.map(run, JOBS):
            print(f"{job}: {len(rows)} rows", flush=True)
            allrows.extend(rows)
    df = pd.DataFrame(allrows)
    df["user_id"] = df["user_id"].str.lower()
    df.to_csv("data/leaderboard_rows.csv.gz", index=False)
    g = df.groupby("user_id")
    cand = pd.DataFrame({
        "user_name": g["user_name"].first(),
        "lists": g["list"].apply(lambda s: ";".join(sorted(set(s)))),
        "n_lists": g["list"].nunique(),
        "best_rank": g["rank"].min(),
    })
    allpnl = df[df["list"] == "overall/all/PNL"].set_index("user_id")
    cand["lb_all_pnl"] = allpnl["pnl"].reindex(cand.index)
    cand["lb_all_vol"] = allpnl["volume"].reindex(cand.index)
    cand.to_csv("data/candidates.csv")
    print(f"unique wallets: {len(cand)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

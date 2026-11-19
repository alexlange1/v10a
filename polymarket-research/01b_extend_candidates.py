"""Step 1b: extend the pool deeper into the leaderboards (wallets not already
in data/candidates.csv). Output: data/candidates_ext.csv"""
import pandas as pd
import pm_api as A

JOBS = [("overall", "all", "PNL", 100000), ("overall", "month", "PNL", 30000), ("overall", "all", "VOLUME", 40000)]
old = pd.read_csv("data/candidates.csv")
have = set(old.user_id.str.lower())
rows = []
for cat, tp, sb, n in JOBS:
    rr = A.leaderboard(time_period=tp, sort_by=sb, category=cat, max_items=n)
    for r in rr:
        r["list"] = f"{cat}/{tp}/{sb}"
    print(cat, tp, sb, len(rr), flush=True)
    rows += rr
df = pd.DataFrame(rows)
df["user_id"] = df.user_id.str.lower()
df = df[~df.user_id.isin(have)]
g = df.groupby("user_id")
ext = pd.DataFrame({"user_name": g.user_name.first(), "lists": g.list.apply(lambda s: ";".join(sorted(set(s)))),
                    "n_lists": g.list.nunique(), "best_rank": g["rank"].min()})
allpnl = df[df.list == "overall/all/PNL"].drop_duplicates("user_id").set_index("user_id")
ext["lb_all_pnl"] = allpnl.pnl.reindex(ext.index)
ext["lb_all_vol"] = allpnl.volume.reindex(ext.index)
ext.to_csv("data/candidates_ext.csv")
print("new wallets:", len(ext))

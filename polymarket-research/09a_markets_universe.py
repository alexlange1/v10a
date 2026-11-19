"""Step 9a: universe for the market-level backtest: every market that resolved
(closedTime) in the last 365 days with >= $10k volume, from Gamma keyset pages."""
import json
import time
import pandas as pd
import pm_api as A

NOW = pd.Timestamp("2026-11-19", tz="UTC")
keep = ["conditionId", "question", "slug", "endDate", "closedTime", "startDate", "outcomes", "outcomePrices",
        "clobTokenIds", "negRisk", "feesEnabled", "feeType", "feeSchedule", "volumeNum", "liquidityNum",
        "orderPriceMinTickSize", "sportsMarketType", "gameStartTime", "umaResolutionStatus", "holdingRewardsEnabled"]
import sys
SLICES = [("2025-10-01", "2025-12-01"), ("2025-12-01", "2026-02-01"), ("2026-02-01", "2026-04-01"), ("2026-04-01", "2026-06-01"),
          ("2026-06-01", "2026-08-01"), ("2026-08-01", "2026-10-01"), ("2026-10-01", "2029-01-01")]
SL = int(sys.argv[1]) if len(sys.argv) > 1 else -1
p = {"closed": "true", "limit": 100, "volume_num_min": 10000, "include_tag": "true",
     "end_date_min": SLICES[SL][0] + "T00:00:00Z" if SL >= 0 else "2025-10-01T00:00:00Z"}
if SL >= 0:
    p["end_date_max"] = SLICES[SL][1] + "T00:00:00Z"
import os, pickle
STATE = f"data/bt/universe_state_{SL}.pkl"
rows, cur, pages = [], None, 0
if os.path.exists(STATE):
    rows, cur, pages = pickle.load(open(STATE, "rb"))
t0 = time.time()
while True:
    q = dict(p)
    if cur:
        q["after_cursor"] = cur
    for attempt in range(20):
        try:
            d = A.get(A.GAMMA + "markets/keyset", q, retries=3)
            break
        except A.ApiError as e:
            if "429" not in str(e):
                raise
            time.sleep(20 + 10 * attempt)
    time.sleep(0.25)
    for m in d.get("markets", []):
        r = {k: m.get(k) for k in keep}
        r["feeSchedule"] = json.dumps(m.get("feeSchedule")) if m.get("feeSchedule") else None
        r["tags"] = "|".join(t.get("slug", "") for t in (m.get("tags") or []))
        ev = (m.get("events") or [{}])[0]
        r["event_slug"], r["series_slug"] = ev.get("slug"), ev.get("seriesSlug")
        rows.append(r)
    pages += 1
    cur = d.get("next_cursor")
    if pages % 100 == 0:
        print(pages, "pages", len(rows), "markets", f"{time.time() - t0:.0f}s", flush=True)
        pickle.dump((rows, cur, pages), open(STATE, "wb"))
    if not cur or not d.get("markets"):
        break
pickle.dump((rows, None, pages), open(STATE, "wb"))
df = pd.DataFrame(rows).drop_duplicates("conditionId")
ct = pd.to_datetime(df.closedTime, utc=True, errors="coerce", format="mixed")
df = df[(ct >= NOW - pd.Timedelta(days=365)) & (ct < NOW - pd.Timedelta(days=1))]
df.to_parquet(f"data/bt/universe_{SL}.parquet")
print("resolved in last year with >=$10k volume:", len(df))

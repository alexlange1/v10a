"""Deadline Fade scanner: lists today's markets that match the backtested rules.

Rules (see README.md / report):
  * single yes/no market (not part of a multi-outcome 'negRisk' event)
  * question is a deadline question ("... by <date>?", "... in 2026?", "before ...")
  * category: politics, geopolitics, economics, finance, tech, culture, crypto price
    (no sports, esports, weather, or crypto up/down)
  * >= $25k traded volume
  * the favourite (usually NO) trades between 95c and 98c
  * the deadline is 7-30 days away; book spread <= 2c
Read-only: it places no orders. Usage:  python3 scanner.py [bankroll_usd]
"""
import re
import sys
import json
import importlib
import pandas as pd
import pm_api as A

category = importlib.import_module("05_features").category
DEADLINE = re.compile(r"\bby\b|\bbefore\b|\bin 20\d\d\b|\bthis year\b|\bend of\b", re.I)
ALLOWED = {"politics", "geopolitics", "economics", "finance", "tech", "culture", "crypto"}
FEE_RATE = {"geopolitics": 0.0, "politics": 0.01, "economics": 0.01, "culture": 0.01, "crypto": 0.01, "finance": 0.05, "tech": 0.05}
BAND = (0.95, 0.98)
DAYS = (7, 30)


def fetch_active():
    now = pd.Timestamp.now(tz="UTC")
    p = {"closed": "false", "active": "true", "limit": 100, "volume_num_min": 25000, "include_tag": "true",
         "end_date_min": (now + pd.Timedelta(days=DAYS[0])).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "end_date_max": (now + pd.Timedelta(days=DAYS[1])).strftime("%Y-%m-%dT%H:%M:%SZ")}
    rows, cur = [], None
    while True:
        q = dict(p)
        if cur:
            q["after_cursor"] = cur
        d = A.get(A.GAMMA + "markets/keyset", q)
        rows += d.get("markets", [])
        cur = d.get("next_cursor")
        if not cur or not d.get("markets"):
            break
    return rows, now


def main(bankroll=10_000.0):
    rows, now = fetch_active()
    out = []
    for m in rows:
        if m.get("negRisk") or not m.get("acceptingOrders", True):
            continue
        q = m.get("question") or ""
        if not DEADLINE.search(q):
            continue
        tags = "|".join(t.get("slug", "") for t in (m.get("tags") or []))
        cat = category(tags, q, m.get("slug"))
        if cat == "politics" and "elections" in tags:
            cat = "politics_election"
        if cat not in ALLOWED:
            continue
        try:
            outcomes = json.loads(m["outcomes"]); prices = [float(x) for x in json.loads(m["outcomePrices"])]
        except Exception:
            continue
        if len(prices) != 2:
            continue
        fav = 0 if prices[0] >= prices[1] else 1
        bb, ba = m.get("bestBid"), m.get("bestAsk")   # quoted for the first outcome
        if bb is None or ba is None:
            continue
        bid, ask = (float(bb), float(ba)) if fav == 0 else (1 - float(ba), 1 - float(bb))
        mid = (bid + ask) / 2
        if not (BAND[0] <= mid < BAND[1]) or ask - bid > 0.02:   # skip wide (illiquid) books
            continue
        tick = float(m.get("orderPriceMinTickSize") or 0.01)
        end = pd.to_datetime(m.get("endDate"), utc=True)
        days = (end - now).total_seconds() / 86400
        # rest a limit order one tick below the ask, never above the midpoint (maker: no fee)
        limit = round(max(bid, min(ask - tick, (mid // tick) * tick)), 4)
        ev = (m.get("events") or [{}])[0]
        out.append({"days_left": round(days, 1), "buy": outcomes[fav], "question": q, "category": cat,
                    "bid": round(bid, 4), "ask": round(ask, 4), "limit_price": limit,
                    "stake_usd": round(0.005 * bankroll, 2), "shares": int(0.005 * bankroll / limit),
                    "volume_usd": round(float(m.get("volumeNum") or 0)), "event": ev.get("slug"),
                    "url": f"https://polymarket.com/event/{ev.get('slug')}" if ev.get("slug") else ""})
    df = pd.DataFrame(out)
    if df.empty:
        print("No markets match the rules right now.")
        return df
    df = df.sort_values(["days_left", "volume_usd"], ascending=[True, False])
    # at most 3 markets (1.5% of bankroll) per event
    df = df.groupby("event", sort=False).head(3)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    print(f"{len(df)} candidates  (bankroll ${bankroll:,.0f}: {0.5:.1f}% = ${0.005 * bankroll:,.0f} per market)\n")
    print(df.drop(columns=["url", "event"]).to_string(index=False))
    df.to_csv("report/scanner_today.csv", index=False)
    return df


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 10_000.0)

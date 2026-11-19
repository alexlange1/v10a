"""Minimal Polymarket public API client (Data API v2, Gamma, CLOB).

Only public, unauthenticated read endpoints are used. Cloudflare rejects the
default python user agent (error 1010), so a normal UA header is sent.
"""
import json
import time
import random
import threading
import requests
from requests.adapters import HTTPAdapter

DATA = "https://data-api.polymarket.com/v2/"
GAMMA = "https://gamma-api.polymarket.com/"
CLOB = "https://clob.polymarket.com/"

_local = threading.local()


def session() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0 (polymarket-research)", "Accept": "application/json"})
        ad = HTTPAdapter(pool_connections=8, pool_maxsize=8)
        s.mount("https://", ad)
        _local.s = s
    return s


class ApiError(Exception):
    pass


def get(url: str, params=None, retries: int = 6, timeout: int = 60):
    """GET with retry/backoff on 429/5xx/network errors. Returns parsed JSON."""
    last = None
    for i in range(retries):
        try:
            r = session().get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504, 520, 522, 524):
                last = f"HTTP {r.status_code}"
                time.sleep(min(30, (2 ** i) + random.random()))
                continue
            try:
                body = r.json()
            except Exception:
                body = r.text[:200]
            raise ApiError(f"HTTP {r.status_code} {url} {params} -> {body}")
        except (requests.ConnectionError, requests.Timeout, json.JSONDecodeError) as e:
            last = repr(e)
            time.sleep(min(30, (2 ** i) + random.random()))
    raise ApiError(f"giving up {url} {params}: {last}")


def paged(path: str, params: dict, max_items: int = 10_000, page: int = 1000):
    """Iterate a cursor-paginated Data API v2 list endpoint."""
    out = []
    p = dict(params)
    p["limit"] = min(page, max_items)
    while True:
        d = get(DATA + path, p)
        rows = d.get("data") or []
        out.extend(rows)
        pg = d.get("pagination") or {}
        cur = pg.get("next_cursor")
        if not pg.get("has_more") or not cur or not rows or len(out) >= max_items:
            break
        p = {"cursor": cur, "limit": min(page, max_items - len(out))}
        # cursor encodes the other params, but keep user/status filters explicit
        for k in ("user", "status", "sort_by", "type", "time_period", "category"):
            if k in params:
                p[k] = params[k]
    return out[:max_items]


def leaderboard(time_period="all", sort_by="PNL", category="overall", max_items=1000):
    params = {"time_period": time_period, "sort_by": sort_by}
    if category and category != "overall":
        params["category"] = category
    return paged("leaderboard", params, max_items=max_items)


def user_pnl(user: str, interval="all", fidelity="1d"):
    d = get(DATA + "user-pnl", {"user": user, "interval": interval, "fidelity": fidelity})
    return d.get("data") or {}


def user_stats(user: str):
    return (get(DATA + "user-stats", {"user": user}) or {}).get("data") or {}


def trades(user: str, max_items=5000):
    return paged("trades", {"user": user}, max_items=max_items)


def activity(user: str, max_items=5000, type_=None):
    p = {"user": user}
    if type_:
        p["type"] = type_
    return paged("activity", p, max_items=max_items, page=500)


def positions(user: str, status="closed", sort_by=None, max_items=5000):
    p = {"user": user, "status": status}
    if sort_by:
        p["sort_by"] = sort_by
    return paged("positions", p, max_items=max_items)


def gamma_markets(condition_ids):
    """Fetch Gamma market objects for up to ~50 condition ids per call."""
    out = []
    ids = list(condition_ids)
    for i in range(0, len(ids), 40):
        chunk = ids[i:i + 40]
        params = [("condition_ids", c) for c in chunk] + [("limit", 100)]
        d = get(GAMMA + "markets", params)
        out.extend(d if isinstance(d, list) else d.get("data", []))
    return out


def prices_history(token_id: str, interval="max", fidelity=60, start_ts=None, end_ts=None):
    p = {"market": token_id, "fidelity": fidelity}
    if start_ts or end_ts:
        if start_ts: p["startTs"] = int(start_ts)
        if end_ts: p["endTs"] = int(end_ts)
    else:
        p["interval"] = interval
    return (get(CLOB + "prices-history", p) or {}).get("history") or []

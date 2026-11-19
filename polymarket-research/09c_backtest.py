"""Step 9c: market-level backtest of the 'buy the near-certain favourite and hold
to resolution' strategy on EVERY market that resolved in the last year (not
just on the winners we studied), using hourly price paths.

Honest-execution assumptions
  * signal uses only information known at the time: the hourly price and the
    market's *scheduled* endDate (not the actual close time);
  * taker fill at mid + half a tick + 0.2c extra slippage;
  * taker fee from the fee schedule taking effect 2026-11-20:
        fee/share = rate * p * (1 - p), rate by category (see FEE_RATE);
  * hold to resolution; payoff = final price of the outcome (1, 0 or 0.5).
"""
import glob
import json
import pickle
import numpy as np
import pandas as pd

H = 3600
FEE_RATE = {  # schedule effective 2026-11-20 (docs.polymarket.com/programs/fee-and-rebate-changes-november-2026)
    "sports": 0.05, "esports": 0.05, "finance": 0.05, "tech": 0.05, "politics_election": 0.05, "other": 0.05,
    "crypto": 0.01, "politics": 0.01, "economics": 0.01, "culture": 0.01, "weather": 0.01, "mentions": 0.01,
    "geopolitics": 0.0,
}
EXTRA_SLIP = 0.002


def to_ts(s):
    return (pd.to_datetime(s, errors="coerce", utc=True, format="mixed") - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)


def category(tags, question, slug):
    import importlib
    cat = importlib.import_module("05_features").category(tags, question, slug)
    if cat == "politics" and "elections" in (tags or ""):
        return "politics_election"
    return cat


def load():
    u = pd.read_parquet("data/bt/universe.parquet")
    u = u[~u.tags.fillna("").str.contains("up-or-down")].copy()
    u["category"] = [category(a, b, c) for a, b, c in zip(u.tags, u.question, u.slug)]
    u["fee_rate"] = u.category.map(FEE_RATE).fillna(0.05)
    u["ct"] = to_ts(u.closedTime)
    u["et"] = to_ts(u.endDate)
    u["final"] = u.outcomePrices.map(lambda s: [float(x) for x in json.loads(s)] if s else None)
    u["tick"] = pd.to_numeric(u.orderPriceMinTickSize, errors="coerce").fillna(0.01)
    P = {}
    for p in sorted(glob.glob("data/bt/prices/p_*.pkl")):
        P.update(pickle.load(open(p, "rb")))
    u = u[u.conditionId.isin([k for k, v in P.items() if v is not None and len(v[0]) > 0])]
    u = u[u.final.map(lambda f: f is not None and len(f) == 2 and abs(sum(f) - 1) < 0.02)]
    return u.reset_index(drop=True), P


_BARS = {}


def bars(u, P):
    """All hourly bars of all markets in one frame (built once and cached)."""
    key = id(P)
    if key in _BARS and len(_BARS[key][1]) == len(u):
        return _BARS[key][0]
    rows_t, rows_p, rows_m = [], [], []
    for i, cid in enumerate(u.conditionId.values):
        t, p0 = P[cid]
        rows_t.append(t); rows_p.append(p0.astype(np.float64)); rows_m.append(np.full(len(t), i, dtype=np.int32))
    b = pd.DataFrame({"m": np.concatenate(rows_m), "t": np.concatenate(rows_t), "p0": np.concatenate(rows_p)})
    b["ct"] = u.ct.values[b.m.values]
    b["et"] = u.et.values[b.m.values]
    b = b[b.t < b.ct - 1800]
    b["side"] = np.where(b.p0 >= 0.5, 0, 1).astype(np.int8)
    b["pf"] = np.where(b.side == 0, b.p0, 1 - b.p0)
    b["h_end"] = (b.et - b.t) / H
    b = b.sort_values(["m", "t"], kind="stable").reset_index(drop=True)
    # entry-time features (only past information): price of *token 0* lagged, so the
    # favourite's own history can be recovered whichever side it is on.
    g = b.groupby("m", sort=False).p0
    for k in (6, 24):
        lag = g.shift(k)
        b[f"fav_chg_{k}h"] = np.where(b.side == 0, b.p0 - lag, lag - b.p0)  # + = favourite rising
    b["max_p0_before"] = g.cummax()
    b["min_p0_before"] = g.cummin()
    b["fav_max_before"] = np.where(b.side == 0, b.max_p0_before, 1 - b.min_p0_before)  # best price the fav side ever had
    d = g.diff().abs()
    b["vol_24h"] = d.groupby(b.m, sort=False).rolling(24, min_periods=6).mean().reset_index(level=0, drop=True)
    b["age_h"] = b.groupby("m", sort=False).cumcount()
    _BARS[key] = (b, u)
    return b


def entries(u, P, lo, hi, max_h_to_end, min_h_to_end=-1e9, cond=None, markets=None):
    """First hourly bar per market where the favourite's mid is in [lo, hi) and the
    *scheduled* end is within max_h_to_end hours. One trade per market.
    `markets`: optional boolean array over u's rows restricting the scope."""
    u = u.reset_index(drop=True)
    b = bars(u, P)
    m = (b.pf >= lo) & (b.pf < hi) & (b.h_end <= max_h_to_end) & (b.h_end >= min_h_to_end)
    if markets is not None:
        m &= np.asarray(markets)[b.m.values]
    if cond is not None:
        m &= cond(b).fillna(False)
    f = b[m].drop_duplicates("m", keep="first")
    if not len(f):
        return pd.DataFrame()
    mk = u.iloc[f.m.values].reset_index(drop=True)
    f = f.reset_index(drop=True)
    fill = np.minimum(f.pf.values + mk.tick.values / 2 + EXTRA_SLIP, 0.999)
    fee = mk.fee_rate.values * fill * (1 - fill)
    cost = fill + fee
    final = np.array([fl[s] for fl, s in zip(mk.final.values, f.side.values)])
    e = pd.DataFrame({
        "conditionId": mk.conditionId.values, "event_slug": mk.event_slug.values, "question": mk.question.values,
        "category": mk.category.values, "entry_ts": f.t.values, "close_ts": mk.ct.values.astype(np.int64),
        "h_to_end": f.h_end.values, "hold_h": (mk.ct.values - f.t.values) / H, "mid": f.pf.values,
        "fill": fill, "fee": fee, "cost": cost, "payoff": final, "won": final >= 0.99,
        "roi": (final - cost) / cost, "negRisk": mk.negRisk.fillna(False).astype(bool).values, "volume": mk.volumeNum.values,
        "fav_chg_6h": f.fav_chg_6h.values, "fav_chg_24h": f.fav_chg_24h.values, "fav_max_before": f.fav_max_before.values,
        "vol_24h": f.vol_24h.values, "age_h": f.age_h.values,
    })
    return e


def calibration(u, P, bands, horizons):
    rows = []
    for lo, hi in bands:
        for hz in horizons:
            e = entries(u, P, lo, hi, hz)
            if not len(e):
                continue
            rows.append({"band": f"{lo:.2f}-{hi:.2f}", "max_h_to_end": hz, "trades": len(e),
                         "avg_fill": e.fill.mean(), "win%": 100 * e.won.mean(),
                         "breakeven_win%": 100 * e.cost.mean(), "avg_roi%": 100 * e.roi.mean(),
                         "median_hold_h": e.hold_h.median(), "loss_rate%": 100 * (e.payoff < e.cost).mean()})
    return pd.DataFrame(rows)


def portfolio(e, bank=10_000.0, frac=0.02, max_event_frac=0.05, cash_floor=0.0):
    """Chronological bankroll simulation. Each trade stakes frac*equity (capped by free
    cash); total stake per event (e.g. one election's many candidate markets) <= max_event_frac*equity.
    Capital is locked until the market closes."""
    e = e.sort_values("entry_ts").reset_index(drop=True)
    cash, open_pos, curve, ev_exposure = bank, [], [], {}
    taken = []
    for r in e.itertuples(index=False):
        still = []
        for (ct, payout, ev, stake) in open_pos:
            if ct <= r.entry_ts:
                cash += payout
                ev_exposure[ev] -= stake
            else:
                still.append((ct, payout, ev, stake))
        open_pos = still
        equity = cash + sum(s for (_, _, _, s) in open_pos)
        stake = min(frac * equity, cash - cash_floor)
        room = max_event_frac * equity - ev_exposure.get(r.event_slug, 0.0)
        stake = min(stake, room)
        if stake < 1:
            continue
        shares = stake / r.cost
        cash -= stake
        ev_exposure[r.event_slug] = ev_exposure.get(r.event_slug, 0.0) + stake
        open_pos.append((r.close_ts, shares * r.payoff, r.event_slug, stake))
        taken.append((r.entry_ts, r.close_ts, stake, shares * r.payoff))
    T = pd.DataFrame(taken, columns=["entry_ts", "close_ts", "stake", "payout"])
    # daily equity marked at cost until resolution
    days = pd.date_range(pd.to_datetime(T.entry_ts.min(), unit="s").normalize(), pd.to_datetime(T.close_ts.max(), unit="s").normalize() + pd.Timedelta(days=1), freq="D")
    ts = (days - pd.Timestamp(0)) // pd.Timedelta(seconds=1)
    realized = np.array([((T.close_ts <= d) * (T.payout - T.stake)).sum() for d in ts])
    curve = pd.Series(bank + realized, index=days)
    return T, curve


if __name__ == "__main__":
    u, P = load()
    print("markets with price paths:", len(u))
    print(u.category.value_counts().to_dict())
    bands = [(0.80, 0.85), (0.85, 0.90), (0.90, 0.93), (0.93, 0.95), (0.95, 0.97), (0.97, 0.98), (0.98, 0.99), (0.99, 0.995)]
    C = calibration(u, P, bands, [6, 24, 72, 168, 360])
    C.to_csv("data/bt/calibration.csv", index=False)
    pd.set_option("display.width", 220)
    print(C.round(3).to_string())

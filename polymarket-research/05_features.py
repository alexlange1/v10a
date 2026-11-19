"""Step 5: per-trader strategy fingerprints for the selected wallets.

Combines (a) the 1Y P&L decomposition (rebates, rewards, yield, fees) with
(b) recent closed positions, open positions, trades and activity, joined to
Gamma market metadata (tags -> category, fee type, resolution).
Output: data/features.parquet (+ data/positions_enriched.parquet)
"""
import glob
import json
import numpy as np
import pandas as pd

D = "data/detail"
YEAR = 365 * 86400


def load(name):
    files = glob.glob(f"{D}/{name}_*.parquet")
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True) if files else pd.DataFrame()


def category(tags: str, question: str, slug: str) -> str:
    t = set((tags or "").split("|"))
    q = f"{question or ''} {slug or ''}".lower()
    if "up-or-down" in t or "up or down" in q:
        return "finance" if "finance-updown" in t else "crypto_updown"
    if "esports" in t or {"counter-strike-2", "league-of-legends", "dota-2", "valorant"} & t:
        return "esports"
    if "sports" in t:
        return "sports"
    if "weather" in t or {"daily-temperature", "highest-temperature", "lowest-temperature"} & t:
        return "weather"
    if "crypto" in t or "crypto-prices" in t:
        return "crypto"
    if {"mention-markets", "mentions", "tweets-markets"} & t:
        return "mentions"
    if {"economy", "economics", "fed", "fed-rates", "inflation"} & t:
        return "economics"
    if {"finance", "equities", "stocks", "commodities", "finance-updown"} & t:
        return "finance"
    if {"tech", "ai", "big-tech"} & t:
        return "tech"
    if {"geopolitics", "world", "middle-east", "iran", "ukraine", "israel"} & t:
        return "geopolitics"
    if {"politics", "elections", "trump", "midterms", "global-elections", "us-presidential-election"} & t:
        return "politics"
    if {"pop-culture", "culture", "movies", "awards", "music", "celebrities"} & t:
        return "culture"
    if "games" in t:
        return "sports"
    return "other"


def to_ts(series):
    """Datetime strings -> unix seconds (pandas 3 parses at microsecond resolution,
    so never rely on astype(int64) // 1e9)."""
    dt = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    return (dt - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)


def market_table():
    mk = pd.read_parquet(f"{D}/markets.parquet")
    mk["category"] = [category(a, b, c) for a, b, c in zip(mk.tags, mk.question, mk.slug)]
    def fee_rate(s):
        try:
            return float(json.loads(s).get("rate", 0)) if s else 0.0
        except Exception:
            return 0.0
    mk["fee_rate"] = mk.feeSchedule.map(fee_rate)
    mk["end_ts"] = to_ts(mk.endDate)
    mk["closed_ts"] = to_ts(mk.closedTime)
    def final_prices(s):
        try:
            return [float(x) for x in json.loads(s)]
        except Exception:
            return None
    mk["final"] = mk.outcomePrices.map(final_prices)
    return mk.set_index("conditionId")


def bucket(p):
    return pd.cut(p, [-0.001, 0.10, 0.30, 0.70, 0.90, 0.97, 1.001],
                  labels=["<0.10", "0.10-0.30", "0.30-0.70", "0.70-0.90", "0.90-0.97", ">=0.97"])


def main():
    cm = pd.read_parquet("data/curve_metrics.parquet").set_index("user_id")
    sel = pd.read_csv("data/selected.csv")["user_id"].tolist()
    mk = market_table()
    now = int(cm.last_ts.max())
    closed, opn, trades, act, stats = load("closed"), load("open"), load("trades"), load("activity"), load("stats")

    # ---- positions enriched with market info -----------------------------------------
    pos = closed.copy()
    pos["cost"] = pos.avg_price * pos.total_size
    pos = pos[pos.last_event_at >= now - YEAR]
    pos = pos.join(mk[["category", "fee_rate", "feeType", "holdingRewardsEnabled", "rewardsMinSize", "end_ts",
                       "closed_ts", "final", "negRisk", "sportsMarketType", "volumeNum"]], on="condition_id")
    pos["category"] = pos.category.fillna("unknown")
    def won(r):
        f = r["final"]
        if f is None or not isinstance(f, (list, np.ndarray)) or len(f) <= int(r["outcome_index"]):
            return np.nan
        v = f[int(r["outcome_index"])]
        return 1.0 if v >= 0.99 else (0.0 if v <= 0.01 else np.nan)
    pos["won"] = pos.apply(won, axis=1)
    pos["roi"] = pos.realized_pnl / pos.cost.where(pos.cost > 0)
    pos["hold_h"] = (pos.last_event_at - pos.first_entry_at) / 3600
    pos["entry_to_end_h"] = (pos.end_ts - pos.first_entry_at) / 3600
    pos["entry_to_close_h"] = (pos.closed_ts - pos.first_entry_at) / 3600
    pos["pbucket"] = bucket(pos.avg_price)
    pos.drop(columns=["final"]).to_parquet("data/positions_enriched.parquet")

    # wallets holding both outcomes of the same market (market making / arbitrage signature)
    both = pd.concat([closed[["proxy_wallet", "condition_id", "outcome_index"]], opn[["proxy_wallet", "condition_id", "outcome_index"]]])
    nb = both.groupby(["proxy_wallet", "condition_id"]).outcome_index.nunique()
    both_share = (nb > 1).groupby(level=0).mean()

    feats = []
    for u in sel:
        f = {"user_id": u}
        c = cm.loc[u]
        g = c.y1_gain
        vol = max(c.y1_d_volume_usdc, 1.0)
        span = max(c.y1_eff_span_days, 1)
        f.update(user_name=c.user_name, y1_gain=g, y1_volume_usdc=c.y1_d_volume_usdc, y1_trades=c.y1_d_trade_count,
                 y1_r2=c.y1_eff_r2, y1_dd_ratio=c.y1_dd_ratio, y1_month_pos_frac=c.y1_month_pos_frac,
                 y1_sharpe=c.y1_sharpe_ann, eff_span_days=c.y1_eff_span_days, hist_days=c.hist_days,
                 alltime_pnl=c.alltime_econ_pnl)
        f["edge_bps"] = g / vol * 1e4
        f["trades_per_day"] = c.y1_d_trade_count / span
        f["avg_trade_usdc"] = c.y1_d_volume_usdc / max(c.y1_d_trade_count, 1)
        f["profit_per_day"] = g / span
        for k in ("wallet_income", "reward_income", "maker_rebate", "taker_rebate", "yield_income", "fees_paid", "position_pnl"):
            f[f"{k}_share"] = c[f"y1_d_{k}"] / g if g else np.nan
        f["fee_drag_bps"] = -c.y1_d_fees_paid / vol * 1e4
        p = pos[pos.proxy_wallet == u]
        f["n_pos"] = len(p)
        if len(p):
            cost = p.cost.clip(lower=0)
            tc = cost.sum()
            f["pos_cost_total"] = tc
            f["median_pos_cost"] = cost.median()
            f["win_rate"] = (p.realized_pnl > 0).mean()
            f["avg_entry_price_w"] = (p.avg_price * cost).sum() / tc if tc else np.nan
            for b in bucket(pd.Series([0.05, 0.2, 0.5, 0.8, 0.93, 0.98])).cat.categories:
                pb = p[p.pbucket == b]
                f[f"cost_share_{b}"] = pb.cost.sum() / tc if tc else np.nan
                f[f"pnl_{b}"] = pb.realized_pnl.sum()
            for cat, pc in p.groupby("category"):
                f[f"cat_{cat}"] = pc.cost.sum() / tc if tc else np.nan
                f[f"catpnl_{cat}"] = pc.realized_pnl.sum()
            f["n_markets"] = p.condition_id.nunique()
            f["n_events"] = p.event_slug.nunique()
            f["median_hold_h"] = p.hold_h.median()
            f["median_entry_to_end_h"] = p.entry_to_end_h.median()
            f["median_entry_to_close_h"] = p.entry_to_close_h.median()
            f["top5_pos_pnl_share"] = p.realized_pnl.nlargest(5).sum() / p.realized_pnl.sum() if p.realized_pnl.sum() > 0 else np.nan
            f["worst_pos_loss_vs_gain"] = -p.realized_pnl.min() / g if g else np.nan
            f["fee_market_share"] = (p.fee_rate > 0).mean()
            f["holding_reward_mkt_share"] = p.holdingRewardsEnabled.fillna(False).astype(float).mean()
            f["negrisk_share"] = p.negRisk.fillna(False).astype(float).mean()
        f["both_sides_share"] = both_share.get(u, np.nan)
        t = trades[trades.proxy_wallet == u]
        if len(t):
            f["sell_share"] = (t.side == "SELL").mean()
            b = t[t.side == "BUY"]
            f["buy_px_ge_090"] = (b.price >= 0.90).mean() if len(b) else np.nan
            f["buy_px_le_010"] = (b.price <= 0.10).mean() if len(b) else np.nan
            f["median_trade_usdc"] = (t["size"] * t["price"]).median()
            hrs = pd.to_datetime(t.timestamp, unit="s").dt.hour
            f["hour_entropy"] = float(-(hrs.value_counts(normalize=True) * np.log(hrs.value_counts(normalize=True))).sum() / np.log(24))
            f["trade_sample_days"] = (t.timestamp.max() - t.timestamp.min()) / 86400
        a = act[act.proxy_wallet == u]
        if len(a):
            vc = a.type.value_counts(normalize=True)
            for typ in ("TRADE", "REDEEM", "MERGE", "SPLIT", "CONVERSION", "REWARD", "MAKER_REBATE", "TAKER_REBATE", "YIELD"):
                f[f"act_{typ.lower()}"] = vc.get(typ, 0.0)
        o = opn[opn.proxy_wallet == u]
        f["open_value"] = o.current_value.sum() if len(o) else 0.0
        f["n_open"] = len(o)
        feats.append(f)
    F = pd.DataFrame(feats)
    st = stats.drop_duplicates("user").set_index("user")
    F["join_date"] = F.user_id.map(st.join_date)
    F.to_parquet("data/features.parquet")
    print(F.shape)


if __name__ == "__main__":
    main()

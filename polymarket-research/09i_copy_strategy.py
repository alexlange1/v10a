"""Step 9i: the final rule, tested out of sample (May 19 - Nov 18 2026):
copy high-probability positions (0.85-0.98) of first-half steady wallets in
non-sports markets, only when >= 2 of them hold the same side; enter after a
delay at the then-current price (taker + new fees); hold to resolution."""
import glob
import importlib
import numpy as np
import pandas as pd

bt = importlib.import_module("09c_backtest")
from curve_metrics import metrics, steady_metrics
SPLIT, H = 1779148800, 3600
S = set(pd.read_csv("data/is_steady.csv").user_id)
cl = pd.concat([pd.read_parquet(p) for p in glob.glob("data/detail/closed_*.parquet") + glob.glob("data/detail_is/closed_*.parquet")], ignore_index=True)
cl = cl[cl.proxy_wallet.isin(S) & (cl.first_entry_at >= SPLIT)].drop_duplicates(["proxy_wallet", "condition_id", "outcome_index"])
u, P = bt.load(); u = u.reset_index(drop=True); ui = u.set_index("conditionId")
cl = cl[cl.condition_id.isin(ui.index)].copy()
cl["category"] = cl.condition_id.map(ui.category)
EXCL = {"sports", "esports", "crypto_updown"}


def signals(delay_h, min_wallets=2, lo=0.85, hi=0.98):
    """One copy trade per (market, outcome): triggered when the min_wallets-th steady wallet
    has entered; we buy delay_h hours later at the hourly price."""
    c = cl[~cl.category.isin(EXCL)].sort_values("first_entry_at")
    c["k"] = c.groupby(["condition_id", "outcome_index"]).cumcount() + 1
    trig = c[c.k == min_wallets]
    rows = []
    for r in trig.itertuples(index=False):
        m = ui.loc[r.condition_id]
        t, p0 = P[r.condition_id]
        tgt = r.first_entry_at + delay_h * H
        j = np.searchsorted(t, tgt)
        if j >= len(t) or t[j] >= m.ct - 1800 or t[j] - tgt > 6 * H:
            continue
        oi = int(r.outcome_index)
        px = float(p0[j]) if oi == 0 else 1 - float(p0[j])
        if not (lo <= px < hi):
            continue
        fill = min(px + m.tick / 2 + bt.EXTRA_SLIP, 0.999)
        cost = fill + m.fee_rate * fill * (1 - fill)
        pay = m.final[oi]
        rows.append({"condition_id": r.condition_id, "question": m.question, "category": m.category, "event_slug": m.event_slug,
                     "entry_ts": int(t[j]), "close_ts": int(m.ct), "px": px, "cost": cost, "payoff": pay,
                     "won": pay >= 0.99, "roi": (pay - cost) / cost, "hold_h": (m.ct - t[j]) / H})
    return pd.DataFrame(rows)


def show(e, label):
    print(f"{label:38s} trades {len(e):5d}  win {100 * e.won.mean():6.2f}%  avg px {e.px.mean():.3f}  ROI/trade {100 * e.roi.mean():6.3f}% "
          f"(se {100 * e.roi.std() / np.sqrt(max(len(e), 1)):.2f})  median hold {e.hold_h.median() / 24:.1f}d")


pd.set_option("display.width", 220)
for d in (1, 6, 24):
    show(signals(d), f"delay {d}h, >=2 wallets, 0.85-0.98")
for k in (1, 3):
    show(signals(6, min_wallets=k), f"delay 6h, >={k} wallets, 0.85-0.98")
for lo, hi in ((0.85, 0.90), (0.90, 0.95), (0.95, 0.98)):
    show(signals(6, 2, lo, hi), f"delay 6h, >=2 wallets, {lo}-{hi}")

E = signals(6)
E.to_parquet("data/bt/copy_rule_entries.parquet")
print("\nby category:\n", E.groupby("category").apply(lambda g: pd.Series({"n": len(g), "win%": 100 * g.won.mean(), "roi%": 100 * g.roi.mean()})).round(2))
print("\nlosing trades (largest):\n", E[~E.won].sort_values("px", ascending=False)[["question", "category", "px", "payoff"]].head(12).to_string())
for frac, evf in ((0.02, 0.05), (0.03, 0.06), (0.05, 0.10)):
    T, curve = bt.portfolio(E, bank=10_000, frac=frac, max_event_frac=evf)
    curve = curve[curve.index <= pd.Timestamp("2026-11-19")]
    t = ((curve.index - pd.Timestamp(0)) // pd.Timedelta(seconds=1)).values.astype(float)
    m = metrics(t, curve.values - 10_000); m.update(steady_metrics(t, curve.values - 10_000))
    peak = np.maximum.accumulate(curve.values); yrs = (curve.index[-1] - curve.index[0]).days / 365
    print(f"stake {frac:.0%} / event cap {evf:.0%}: trades {len(T)}, final ${curve.iloc[-1]:,.0f} ({100 * (curve.iloc[-1] / 1e4 - 1):.1f}% in {yrs * 12:.1f} months), "
          f"max DD {100 * ((peak - curve.values) / peak).max():.1f}%, R2 {m.get('r2', np.nan):.2f}, months up {m.get('month_pos_frac', np.nan)}")
    curve.to_csv(f"data/bt/copy_curve_{int(frac * 100)}pct.csv")

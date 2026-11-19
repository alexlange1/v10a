"""Step 7: assign each steady climber a primary strategy archetype (rule based,
first match wins) and summarize what each archetype looks like."""
import numpy as np
import pandas as pd

F = pd.read_parquet("data/features.parquet")
cats = [c for c in F.columns if c.startswith("cat_")]
F[cats] = F[cats].fillna(0)
F["primary_cat"] = F[cats].idxmax(axis=1).str[4:]
F["primary_share"] = F[cats].max(axis=1)
F["hp_share"] = F["cost_share_0.90-0.97"].fillna(0) + F["cost_share_>=0.97"].fillna(0)
F["mid_share"] = F["cost_share_0.30-0.70"].fillna(0)
F["long_share"] = F["cost_share_<0.10"].fillna(0) + F["cost_share_0.10-0.30"].fillna(0)
F["lp_income_share"] = F.reward_income_share.fillna(0) + F.maker_rebate_share.fillna(0)


def archetype(r):
    if r.lp_income_share >= 0.25 or (r.both_sides_share >= 0.30 and r.trades_per_day >= 50):
        return "Market maker / liquidity provider"
    if r.cat_crypto_updown >= 0.5:
        return "Crypto up/down short-term bot"
    if r.hp_share >= 0.5:
        return "High-probability 'bond' buyer"
    if r.cat_weather >= 0.5:
        return "Weather specialist"
    if r.cat_sports + r.cat_esports >= 0.5:
        return "Sports/esports trader"
    if r.long_share >= 0.4:
        return "Longshot buyer"
    return "Discretionary / event trader"


F["archetype"] = F.apply(archetype, axis=1)
F.to_parquet("data/features_arch.parquet")

agg = F.groupby("archetype").agg(
    traders=("user_id", "size"),
    total_1y_profit_M=("y1_gain", lambda x: x.sum() / 1e6),
    median_1y_profit=("y1_gain", "median"),
    median_edge_bps=("edge_bps", "median"),
    median_win_rate=("win_rate", "median"),
    median_entry_px=("avg_entry_price_w", "median"),
    median_trades_day=("trades_per_day", "median"),
    median_trade_usd=("median_trade_usdc", "median"),
    median_hours_entry_to_close=("median_entry_to_close_h", "median"),
    median_dd_ratio=("y1_dd_ratio", "median"),
    median_fee_drag_bps=("fee_drag_bps", "median"),
    median_lp_income_share=("lp_income_share", "median"),
).sort_values("traders", ascending=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(agg.round(3).T)
agg.to_csv("data/archetype_summary.csv")
print()
for a, g in F.groupby("archetype"):
    print(a, "-> primary categories:", g.primary_cat.value_counts().head(5).to_dict())

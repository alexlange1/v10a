"""Definition of a 'beautiful chart' (steady climber) over the last 365 days.

All thresholds are on the trader's economic P&L (position P&L incl. unrealized,
plus rebates/rewards/yield), measured from the day the curve started moving.
"""
import numpy as np
import pandas as pd

RULES = {
    "y1_gain": (">=", 3000),             # made at least $3k over the last year
    "y1_eff_span_days": (">=", 150),     # climbed for at least ~5 months
    "days_since_trade": ("<=", 30),      # still active
    "y1_eff_r2": (">=", 0.80),           # close to a straight-ish line (from the day it started moving)
    "y1_r2": (">=", 0.80),               # ...and over the whole 1Y chart (rejects flat-then-surge)
    "y1_month_pos_frac": (">=", 0.70),   # >=70% of 30-day blocks were up
    "y1_max_month_share": ("<=", 0.40),  # no single month made >40% of the profit
    "y1_dd_ratio": ("<=", 0.25),         # worst drawdown <= 25% of the total gain
    "y1_top5_day_share": ("<=", 0.50),   # the 5 best days made <= 50% of the gain
    "y1_min_third_share": (">=", 0.10),  # every third of the climb added >=10% (no flat-then-surge / plateau)
    "y1_last90_gain": (">", 0),          # still making money in the last 90 days
}


def add_derived(df):
    df = df.copy()
    df["days_since_trade"] = (df.last_ts.max() - df.last_trade_ts) / 86400
    return df


def mask(df, rules=RULES):
    m = pd.Series(True, index=df.index)
    for col, (op, v) in rules.items():
        x = df[col].replace([np.inf, -np.inf], np.nan)
        m &= {">=": x >= v, "<=": x <= v, ">": x > v}[op]
    return m.fillna(False)

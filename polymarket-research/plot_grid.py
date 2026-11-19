"""Render grids of 1Y P&L curves for visual validation of the selection."""
import sys, glob, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from curve_metrics import window_series, DAY

_cache = {}
def load_series(users):
    need = set(users)
    out = {}
    for p in sorted(glob.glob("data/pnl/*.pkl")):
        with open(p, "rb") as f:
            d = pickle.load(f)
        for u in need & d.keys():
            out[u] = d[u]
    return out

def grid(df, path, title, ncol=6, nrow=5, field="economic_pnl"):
    df = df.head(ncol * nrow)
    S = load_series(df.user_id.tolist())
    now = max(int(r["t"][-1]) for r in S.values())
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.6, nrow * 1.9), facecolor="#15171c")
    for ax, (_, r) in zip(axes.flat, df.iterrows()):
        rec = S[r.user_id]
        tt, yy = window_series(rec["t"], rec[field], now - 365 * DAY, now)
        ax.set_facecolor("#1e2128")
        ax.plot((tt - tt[0]) / DAY, yy, color="#7c6cf0", lw=1.3)
        ax.fill_between((tt - tt[0]) / DAY, yy, yy.min(), color="#7c6cf0", alpha=0.15)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        name = str(r.get("user_name") or r.user_id[:10])[:16]
        ax.set_title(f"{name}  +${r.y1_gain/1000:,.0f}k  R²={r.y1_r2:.2f}", color="#c9ccd3", fontsize=7)
    for ax in list(axes.flat)[len(df):]:
        ax.axis("off")
    fig.suptitle(title, color="white", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)

if __name__ == "__main__":
    df = pd.read_parquet("data/curve_metrics.parquet")
    now = df.last_ts.max()
    df["days_since_trade"] = (now - df.last_trade_ts) / 86400
    b = df[(df.y1_gain >= 5000) & (df.y1_span_days >= 180) & (df.days_since_trade <= 30)].copy()
    b = b.sort_values("y1_beauty", ascending=False).reset_index(drop=True)
    print("eligible:", len(b))
    for lo, hi in [(0, 30), (300, 330), (600, 630), (900, 930), (1200, 1230)]:
        grid(b.iloc[lo:hi], f"data/charts/beauty_rank_{lo:04d}.png", f"Beauty rank {lo}-{hi}")
    print(b.loc[[0, 300, 600, 900, 1200], ["y1_beauty", "y1_r2", "y1_dd_ratio", "y1_max_day_share", "y1_pos_week_frac"]])

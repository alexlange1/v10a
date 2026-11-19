"""Step 9m: fair portfolio evaluation. Many same-day opportunities compete for the
same cash, so which ones a trader takes is effectively random. Run the
capital-limited simulation 300 times with random same-day ordering."""
import importlib
import numpy as np
import pandas as pd

bt = importlib.import_module("09c_backtest")
from curve_metrics import metrics, steady_metrics
e0 = pd.read_parquet("data/bt/neh_final_entries.parquet")
rng = np.random.default_rng(11)


def run(e, frac=0.02, evf=0.05, reps=300):
    res, curves = [], []
    for k in range(reps):
        x = e.assign(_r=rng.random(len(e))).sort_values(["entry_ts", "_r"]).drop(columns="_r")
        x = x.assign(entry_ts=x.entry_ts + np.arange(len(x)) % 3600)  # keep the shuffled order inside the day
        T, curve = bt.portfolio(x, bank=10_000, frac=frac, max_event_frac=evf)
        curve = curve[curve.index <= pd.Timestamp("2026-11-19")]
        yrs = (curve.index[-1] - curve.index[0]).days / 365
        peak = np.maximum.accumulate(curve.values)
        t = ((curve.index - pd.Timestamp(0)) // pd.Timedelta(seconds=1)).values.astype(float)
        m = metrics(t, curve.values - 10_000)
        res.append({"cagr": (curve.iloc[-1] / 1e4) ** (1 / yrs) - 1, "maxdd": ((peak - curve.values) / peak).max(), "r2": m.get("r2", np.nan), "trades": len(T)})
        curves.append(curve.rename(k))
    R = pd.DataFrame(res)
    C = pd.concat(curves, axis=1).ffill()
    return R, C


def report(e, label, frac, evf, reps):
    R, C = run(e, frac=frac, evf=evf, reps=reps)
    q = R.quantile([.1, .5, .9])
    print(f"{label} stake {frac:.1%} / event cap {evf:.1%}: CAGR p10 {100*q.cagr[.1]:.1f}% median {100*q.cagr[.5]:.1f}% p90 {100*q.cagr[.9]:.1f}% "
          f"| P(loss) {100*(R.cagr < 0).mean():.0f}% | max DD median {100*q.maxdd[.5]:.1f}% | R2 {q.r2[.5]:.2f} | trades {R.trades.median():.0f}")
    C.quantile(.5, axis=1).to_csv(f"data/bt/final_median_curve_{label}_{int(frac * 1000)}.csv")
    R.to_csv(f"data/bt/final_mc_runs_{label}_{int(frac * 1000)}.csv", index=False)


if __name__ == "__main__":
    for label in ("taker", "maker"):
        e = e0.copy()
        if label == "maker":
            e["cost"] = e.mid - e.tick / 2
        # the recommended sizing (0.5% per market, 1.5% per event) and, for contrast, 2% / 5%
        report(e, label, 0.005, 0.015, 60)
        report(e, label, 0.02, 0.05, 60)

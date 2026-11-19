"""Step 9n: the final Deadline Fade rule = core cell (favourite 0.95-0.98, 7-30 days
before the deadline) restricted to single yes/no markets (no negRisk events)."""
import importlib
import pandas as pd

neh = importlib.import_module("09f_neh_backtest")
u = neh.universe().set_index("conditionId")
e = pd.read_parquet("data/bt/neh_core_entries.parquet")
e["negRisk"] = e.conditionId.map(u.negRisk).fillna(False).astype(bool)
e["tick"] = e.conditionId.map(u.tick)
e["roi_maker"] = (e.payoff - (e.mid - e.tick / 2)) / (e.mid - e.tick / 2)
for flag, g in e.groupby("negRisk"):
    a, b = g[g.entry_ts < neh.SPLIT], g[g.entry_ts >= neh.SPLIT]
    print(f"negRisk={flag}: n {len(g)} taker {100*g.roi.mean():.3f}% (IS {100*a.roi.mean():.3f}, OOS {100*b.roi.mean():.3f}) "
          f"maker {100*g.roi_maker.mean():.3f}%")
e[~e.negRisk].to_parquet("data/bt/neh_final_entries.parquet")

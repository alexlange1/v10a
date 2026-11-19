"""Step 3b: apply the 'beautiful chart' rules (selection.py) -> data/selected.csv."""
import pandas as pd
from selection import add_derived, mask

df = add_derived(pd.read_parquet("data/curve_metrics.parquet"))
sel = df[mask(df)].sort_values("y1_steady", ascending=False)
sel[["user_id"]].to_csv("data/selected.csv", index=False)
print("selected:", len(sel))

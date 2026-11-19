"""Step 6b: wallets that were steady in the FIRST half only (for the copy test),
and the ones whose positions still need downloading."""
import pandas as pd

P = pd.read_parquet("data/persistence.parquet")
S = P[P.is_steady & P.active_end_is & (P.is_gain >= 1500)]
S[["user_id"]].to_csv("data/is_steady.csv", index=False)
have = set(pd.read_csv("data/selected.csv").user_id)
pd.DataFrame({"user_id": sorted(set(S.user_id) - have)}).to_csv("data/is_steady_missing.csv", index=False)
print("first-half steady:", len(S), " to fetch:", len(set(S.user_id) - have))
print("then run: DETAIL_OUT=data/detail_is python3 04_fetch_trader_detail.py data/is_steady_missing.csv")

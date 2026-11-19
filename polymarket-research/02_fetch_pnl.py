"""Step 2: download the full daily P&L history for every candidate wallet.

Stores compact float32 series per wallet in data/pnl/chunk_XXXX.pkl so the
scoring step can be iterated on without re-downloading.
"""
import os
import sys
import time
import pickle
import concurrent.futures as cf
import numpy as np
import pandas as pd
import pm_api as A

FIELDS = ["economic_pnl", "position_pnl", "realized_pnl", "unrealized_pnl", "wallet_income",
          "fees_paid", "fees_refunded", "maker_rebate", "taker_rebate", "reward_income",
          "yield_income", "sponsored_income", "volume", "volume_usdc", "trade_count"]
OUT = "data/pnl"
CHUNK = 2000


def fetch(u):
    try:
        d = A.user_pnl(u, interval="all", fidelity="1d")
    except A.ApiError as e:
        return u, {"error": str(e)[:200]}
    pts = d.get("points") or []
    if not pts:
        return u, {"t": np.zeros(0, dtype=np.int64)}
    rec = {"t": np.array([p["timestamp"] for p in pts], dtype=np.int64)}
    for f in FIELDS:
        rec[f] = np.array([(p.get(f) or 0.0) for p in pts], dtype=np.float64 if f in ("volume", "volume_usdc") else np.float32)
    return u, rec


def main(cand_path="data/candidates.csv", prefix="chunk"):
    os.makedirs(OUT, exist_ok=True)
    users = pd.read_csv(cand_path)["user_id"].tolist()
    chunks = [users[i:i + CHUNK] for i in range(0, len(users), CHUNK)]
    t0 = time.time()
    for ci, chunk in enumerate(chunks):
        path = f"{OUT}/{prefix}_{ci:04d}.pkl"
        if os.path.exists(path):
            continue
        res = {}
        with cf.ThreadPoolExecutor(32) as ex:
            for u, rec in ex.map(fetch, chunk):
                res[u] = rec
        with open(path + ".tmp", "wb") as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(path + ".tmp", path)
        errs = sum(1 for r in res.values() if "error" in r)
        print(f"chunk {ci + 1}/{len(chunks)} done, errors={errs}, elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])

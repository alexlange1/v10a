"""Step 9b: hourly price path (token 0) for the 15 days before each market closed.
NO/second-outcome price = 1 - p for binary markets."""
import os
import json
import time
import pickle
import concurrent.futures as cf
import numpy as np
import pandas as pd
import pm_api as A

OUT = "data/bt/prices"


def to_ts(s):
    return (pd.to_datetime(s, errors="coerce", utc=True, format="mixed") - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)


def one(row):
    cid, tok, ct = row
    for attempt in range(8):
        try:
            h = A.prices_history(tok, fidelity=60, start_ts=ct - 14 * 86400, end_ts=ct + 3600)
            return cid, (np.array([x["t"] for x in h], dtype=np.int64), np.array([x["p"] for x in h], dtype=np.float32))
        except A.ApiError as e:
            if "429" in str(e) or "giving up" in str(e):
                time.sleep(10 + 10 * attempt)
                continue
            return cid, None
    return cid, None


def main():
    os.makedirs(OUT, exist_ok=True)
    import glob as _g
    if not os.path.exists("data/bt/universe.parquet"):
        parts = [pd.read_parquet(f) for f in sorted(_g.glob("data/bt/universe_[0-9].parquet"))]
        pd.concat(parts, ignore_index=True).drop_duplicates("conditionId").to_parquet("data/bt/universe.parquet")
    u = pd.read_parquet("data/bt/universe.parquet")
    u = u[~u.tags.fillna("").str.contains("up-or-down")]
    u = u[u.clobTokenIds.notna()]
    # stratified sample: keep every market in the smaller categories, sample sports/esports
    import importlib
    cat = importlib.import_module("05_features").category
    u["category"] = [cat(a, b, c) for a, b, c in zip(u.tags, u.question, u.slug)]
    caps = {"sports": 40000, "esports": 12000}
    u = pd.concat([g.sample(min(len(g), caps.get(k, 10**9)), random_state=42) for k, g in u.groupby("category")])
    u = u.sort_values("conditionId")
    u["tok0"] = u.clobTokenIds.map(lambda s: json.loads(s)[0] if s else None)
    u["ct"] = to_ts(u.closedTime)
    jobs = list(zip(u.conditionId, u.tok0, u.ct.astype(int)))
    print("markets:", len(jobs), flush=True)
    B = 4000
    t0 = time.time()
    for bi in range(0, len(jobs), B):
        path = f"{OUT}/p_{bi // B:04d}.pkl"
        if os.path.exists(path):
            continue
        res = {}
        with cf.ThreadPoolExecutor(32) as ex:
            for cid, arr in ex.map(one, jobs[bi:bi + B]):
                res[cid] = arr
        pickle.dump(res, open(path, "wb"))
        print(f"{min(bi + B, len(jobs))}/{len(jobs)} {time.time() - t0:.0f}s, empty={sum(1 for v in res.values() if v is None or len(v[0]) == 0)}", flush=True)


if __name__ == "__main__":
    main()

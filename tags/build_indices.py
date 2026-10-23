"""index/t1_index_all.jsonl = train + validation rows (3,192 cases). Run after build_t1_index.py."""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
rows = [l for p in ("t1_index_train.jsonl", "t1_index_val.jsonl") for l in open(H / "index" / p)]
with open(H / "index" / "t1_index_all.jsonl", "w") as f:
    f.writelines(rows)
print(f"[index] t1_index_all.jsonl: {len(rows)} rows")

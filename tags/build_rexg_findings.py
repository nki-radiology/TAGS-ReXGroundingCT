"""Input for the LLM lobe mapping: every unique finding sentence of the challenge dataset.

Writes rexg_ts/rexg_findings.jsonl ({"key": "<split>|<case>|<finding>", "text", "scan_ids": []}), one row per
unique sentence in train -> val -> test first-occurrence order. Then run llm_mapping/map_categories_llm.py
--input-jsonl tags/rexg_ts/rexg_findings.jsonl --output tags/rexg_ts/rexg_ts_labels.jsonl (scripts/00_prepare_data.sh).
eval_t1_val200.named_region() reads both files; sentences missing from them fall back to a regex parser.
"""
import json, sys
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
from paths import CHALLENGE_JSON  # noqa: E402

d = json.load(open(CHALLENGE_JSON))
out = H / "rexg_ts"; out.mkdir(exist_ok=True)
seen, n = set(), 0
with open(out / "rexg_findings.jsonl", "w") as f:
    for split in ("train", "val", "test"):
        items = list(d[split].values()) if isinstance(d[split], dict) else d[split]
        for e in items:
            case = e["name"][:-7] if e["name"].endswith(".nii.gz") else e["name"]
            for k in sorted(e["findings"], key=int):
                t = e["findings"][k]
                if t in seen:
                    continue
                seen.add(t); n += 1
                f.write(json.dumps({"key": f"{split}|{case}|{k}", "text": t, "scan_ids": []}) + "\n")
print(f"[rexg] {n} unique finding sentences -> {out / 'rexg_findings.jsonl'}")

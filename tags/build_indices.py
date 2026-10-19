"""Derived training indices (run after build_t1_index.py, and after prep_aux_texts.py for --aux).

    index/t1_index_all.jsonl        train + val rows (3,192 cases)
    index/t1_all_valx5.jsonl        train + val x5; copies 2-5 get case keys '<case>__x1'..'__x4' so the
                                    case-level sampler sees them as distinct cases (val = 25% of case draws)
    aux_bank/index_2e_only.jsonl    the pleural-effusion (2e) rows of aux_bank/index_pilot.jsonl, in order

Reproduces the files used for the submission (there they were called t1_index_all_local.jsonl and
t1_all_valx5_loc.jsonl, i.e. the same rows with node-local image paths).
"""
import json, sys
from pathlib import Path

H = Path(__file__).resolve().parent
rd = lambda p: [json.loads(l) for l in open(H / p)]


def write(p, rows):
    with open(H / p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[index] {p}: {len(rows)} rows / {len({r['case'] for r in rows})} cases")


if "--aux" in sys.argv:
    write("aux_bank/index_2e_only.jsonl", [r for r in rd("aux_bank/index_pilot.jsonl") if r["category"] == "2e"])
else:
    tr, va = rd("index/t1_index_train.jsonl"), rd("index/t1_index_val.jsonl")
    write("index/t1_index_all.jsonl", tr + va)
    dup = []
    for rep in range(5):
        for r in va:
            q = dict(r)
            if rep:
                q["case"] = f"{r['case']}__x{rep}"
            dup.append(q)
    write("index/t1_all_valx5.jsonl", tr + dup)

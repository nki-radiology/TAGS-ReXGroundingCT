"""Rebuild the two CT-RATE auxiliary pools WITHOUT any challenge test (or val) patient.

The original pools (aux_bank/index_pilot.jsonl, aux_bank/index_2e_only.jsonl) excluded val cases
but not test cases: 80 of the 300 test studies (44 identical scans + 36 other reconstructions of
the same study, same report) were in them. This script removes every row whose PATIENT appears in
the challenge test or val split, refills each category to its original count by sampling (seed 0)
from index/aux_ctrate_full.jsonl rows of non-challenge-test/val patients not already in the pool,
and writes a matching text bank (old embeddings reused, new sentences embedded with the same
Qwen3-Embedding-4B pooling as the original bank).

    python build_clean_aux.py            # needs a GPU for the new embeddings
Use aux_bank_clean/index_pilot_clean.jsonl / index_2e_only_clean.jsonl with --aux-bank aux_bank_clean.
Outputs: aux_bank_clean/{index_pilot_clean.jsonl, index_2e_only_clean.jsonl, texts.json, embeddings.npy}
"""
import json, random, collections, sys
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
OUT = H / "aux_bank_clean"; OUT.mkdir(exist_ok=True)
pat = lambda n: "_".join(n.split("_")[:2])
strip = lambda n: n[:-7] if n.endswith(".nii.gz") else n

from paths import CHALLENGE_JSON  # noqa: E402
d = json.load(open(CHALLENGE_JSON))
BAN = {pat(strip(i["name"])) for i in d["test"]} | {pat(strip(i["name"])) for i in d["val"]}

pilot = [json.loads(l) for l in open(H / "aux_bank/index_pilot.jsonl")]
full = [json.loads(l) for l in open(H / "index/aux_ctrate_full.jsonl")]
keep = [r for r in pilot if pat(r["case"]) not in BAN]
need = collections.Counter(r["category"] for r in pilot) - collections.Counter(r["category"] for r in keep)
used = {(r["case"], r["text"]) for r in pilot}
rng = random.Random(0)
added = []
for cat, n in sorted(need.items()):
    cand = [r for r in full if r["category"] == cat and pat(r["case"]) not in BAN and (r["case"], r["text"]) not in used]
    rng.shuffle(cand); added += cand[:n]
    print(f"[clean] category {cat}: removed {n} rows of test/val patients, refilled {min(n, len(cand))} from {len(cand)} candidates")
clean = keep + added
clean_2e = [r for r in clean if r["category"] == "2e"]
assert not any(pat(r["case"]) in BAN for r in clean), "test/val patient left in pool"
print(f"[clean] pilot {len(pilot)} -> {len(clean)} rows | 2e-only {sum(r['category']=='2e' for r in pilot)} -> {len(clean_2e)} rows")

for name, rows in [("index_pilot_clean.jsonl", clean), ("index_2e_only_clean.jsonl", clean_2e)]:
    with open(OUT / name, "w") as fh:
        for r in rows: fh.write(json.dumps(r) + "\n")

old_t = json.load(open(H / "aux_bank/texts.json")); old_e = np.load(H / "aux_bank/embeddings.npy")
t2v = {t: old_e[i] for i, t in enumerate(old_t)}
texts = sorted({r["text"] for r in clean})
new = [t for t in texts if t not in t2v]
print(f"[bank] {len(texts)} texts, {len(new)} new to embed")
if new:
    import torch
    from textsens import embed_texts   # same Qwen3-Embedding-4B pooling as the original bank
    vecs = embed_texts(new, "cuda" if torch.cuda.is_available() else "cpu")
    for t, v in zip(new, vecs): t2v[t] = v
    # sanity: re-embed a few OLD texts and compare to the stored vectors (must be ~1.0 cosine)
    chk = old_t[:8]; cv = embed_texts(chk, "cuda" if torch.cuda.is_available() else "cpu")
    cos = [float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) for a, b in zip(cv, old_e[:8])]
    print(f"[bank] re-embed check vs stored bank: min cosine {min(cos):.4f}")
    assert min(cos) > 0.99, "new embeddings are not in the same space as the old bank"
json.dump(texts, open(OUT / "texts.json", "w"))
np.save(OUT / "embeddings.npy", np.stack([t2v[t] for t in texts]).astype(old_e.dtype))
print("[done]", OUT)

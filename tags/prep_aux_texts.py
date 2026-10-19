"""Embed the auxiliary CT-RATE finding texts so they can condition the model.

The aux findings come from real CT-RATE reports, so their strings are NOT in
`shared_deps/voxtell_text_bank` (which only holds ReXGroundingCT's 6,926 findings). This
embeds them with the SAME encoder and pooling as the bank (Qwen3-Embedding-4B via VoxTell's
`last_token_pool` + `wrap_with_instruction`), otherwise the aux vectors would live in a
different space from the ones the model was trained on.

Masks are NOT filtered here. ~15% of aux findings have an empty TotalSegmentator mask (the
report says "nodule", the segmenter found none) and training those would teach the model to
suppress nodules — but checking requires loading every mask (~2 s each, hours for 20k), so the
dataloader skips them at load time instead.

    sbatch run/py.sh prep_aux_texts.py --index index/aux_ctrate_full.jsonl --cap 6000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path.insert(0, str(HERE))
from textsens import embed_texts  # noqa: E402  (VoxTell-consistent Qwen pooling)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(HERE / "index" / "aux_ctrate_full.jsonl"))
    ap.add_argument("--out-dir", default=str(HERE / "aux_bank"))
    ap.add_argument("--cap", type=int, default=6000,
                    help="max aux FINDINGS to keep for the pilot (0 = all)")
    ap.add_argument("--cats", default="2d,2e")
    a = ap.parse_args()
    keep_cats = set(a.cats.split(","))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    with open(a.index) as fh:
        for line in fh:
            r = json.loads(line)
            if r["category"] in keep_cats:
                rows.append(r)
    # Cap by taking a spread across the file rather than the first N, so the pilot is not all
    # from one alphabetical slice of studies.
    if a.cap and len(rows) > a.cap:
        idx = np.linspace(0, len(rows) - 1, a.cap).astype(int)
        rows = [rows[i] for i in idx]
    texts = sorted({r["text"] for r in rows})
    print(f"[aux] {len(rows)} findings, {len(texts)} unique texts "
          f"(2d={sum(1 for r in rows if r['category']=='2d')}, "
          f"2e={sum(1 for r in rows if r['category']=='2e')})", flush=True)

    vecs = embed_texts(texts, dev).astype(np.float32)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", vecs)
    json.dump(texts, open(out / "texts.json", "w"))
    with open(out / "index_pilot.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[aux] saved {vecs.shape} + {len(rows)}-row pilot index -> {out}", flush=True)


if __name__ == "__main__":
    main()

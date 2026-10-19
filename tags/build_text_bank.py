"""Qwen3-Embedding-4B bank of the train + val finding sentences (6,467 unique), used for training.

Same encoder, instruction and last-token pooling as VoxTell (textsens.embed_texts). For the submission the
bank was built by an equivalent stand-alone script; its vectors match textsens.embed_texts at cosine 1.00000.
Stored as float16, like the original. Test sentences go in a separate bank (prep_test_texts.py).
"""
import json, sys
from pathlib import Path
import numpy as np, torch

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
from paths import CHALLENGE_JSON, TEXT_BANK_DIR  # noqa: E402
from textsens import embed_texts  # noqa: E402

d = json.load(open(CHALLENGE_JSON))
texts = sorted({t for s in ("train", "val") for e in (d[s].values() if isinstance(d[s], dict) else d[s])
                for t in e["findings"].values()})
print(f"[bank] {len(texts)} unique train+val sentences", flush=True)
vecs = embed_texts(texts, "cuda" if torch.cuda.is_available() else "cpu").astype(np.float16)
TEXT_BANK_DIR.mkdir(parents=True, exist_ok=True)
np.save(TEXT_BANK_DIR / "embeddings.npy", vecs); json.dump(texts, open(TEXT_BANK_DIR / "texts.json", "w"))
print(f"[bank] saved {vecs.shape} -> {TEXT_BANK_DIR}")

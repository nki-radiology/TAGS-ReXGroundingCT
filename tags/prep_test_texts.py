"""Embed the TEST findings so the model can be prompted with them.

Only 115 of the 582 test findings appear in `shared_deps/voxtell_text_bank` (that bank holds
the train/val strings). The rest must be embedded with the SAME encoder and pooling
(Qwen3-Embedding-4B, VoxTell's last_token_pool + wrap_with_instruction), otherwise the test
vectors would live in a different space from the ones the model was trained on.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from textsens import embed_texts

from paths import CHALLENGE_JSON  # noqa: E402
d = json.load(open(CHALLENGE_JSON))
items = list(d['test'].values()) if isinstance(d['test'], dict) else d['test']
texts = sorted({t for e in items for t in e['findings'].values()})
print(f"[test] {len(items)} cases, {sum(len(e['findings']) for e in items)} findings, "
      f"{len(texts)} unique texts", flush=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
vecs = embed_texts(texts, dev).astype(np.float32)
out = HERE / "test_bank"; out.mkdir(exist_ok=True)
np.save(out / "embeddings.npy", vecs); json.dump(texts, open(out / "texts.json", "w"))
print(f"[test] saved {vecs.shape} -> {out}", flush=True)

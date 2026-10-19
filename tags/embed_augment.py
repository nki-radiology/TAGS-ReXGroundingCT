"""Embed the augmentation variant strings with the SAME encoder+pooling as the main bank."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from textsens import embed_texts                                    # noqa: E402

def main():
    texts = json.load(open(HERE / "region_prior" / "augment_new_texts.json"))
    out = HERE / "augment_bank"; out.mkdir(exist_ok=True)
    print(f"[embed] {len(texts)} variant texts", flush=True)
    import torch
    vecs = embed_texts(texts, "cuda" if torch.cuda.is_available() else "cpu")
    json.dump(texts, open(out / "texts.json", "w"))
    np.save(out / "embeddings.npy", np.asarray(vecs, np.float16))
    print(f"[embed] -> {out}  shape {np.asarray(vecs).shape}", flush=True)

if __name__ == "__main__":
    main()

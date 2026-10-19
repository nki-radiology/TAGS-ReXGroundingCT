"""Embed anatomical lobe phrases into a small bank the training sampler can draw on.

WHY: our fine-tunes ignore the location word in the prompt (textsens Dice(A,B)=1.000). The one
location signal we can add SAFELY on a report-derived (sparsely annotated) dataset comes from
the DENSELY-labelled TotalSegmentator lobes -- prompt "left upper lobe" -> LUL mask (positive),
and "left upper lobe" on a crop with no LUL -> empty (a TRUE negative, because TotalSeg labels
every lobe voxel). See train_t1_full sampler for how these are mixed in.

Multiple phrasings per lobe so the model binds the *concept* of the lobe, not one string; the
textsens diagnostic uses a single canonical phrase, and generalising to it is the thing we want
to measure, so the training set deliberately spans wordings around it.

    sbatch run/py.sh build_lobe_bank.py     # GPU node (Qwen cached -> works offline)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from textsens import embed_texts  # noqa: E402  (VoxTell-consistent Qwen pooling)

OUT = HERE / "lobe_bank"
# TotalSegmentator lung-lobe label ids -> varied natural phrasings.
LOBE_PHRASES = {
    10: ["left upper lobe", "the upper lobe of the left lung", "LUL",
         "left upper zone", "upper lobe of the left lung"],
    11: ["left lower lobe", "the lower lobe of the left lung", "LLL",
         "left lower zone", "lower lobe of the left lung"],
    12: ["right upper lobe", "the upper lobe of the right lung", "RUL",
         "right upper zone", "upper lobe of the right lung"],
    13: ["right middle lobe", "the middle lobe of the right lung", "RML",
         "right middle zone", "middle lobe of the right lung"],
    14: ["right lower lobe", "the lower lobe of the right lung", "RLL",
         "right lower zone", "lower lobe of the right lung"],
}


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    texts, lobe_ids = [], []
    for lobe, phrases in LOBE_PHRASES.items():
        for p in phrases:
            texts.append(p)
            lobe_ids.append(int(lobe))
    print(f"[lobe-bank] embedding {len(texts)} phrases for {len(LOBE_PHRASES)} lobes", flush=True)
    vecs = embed_texts(texts, dev).astype(np.float32)          # (P, 2560)
    OUT.mkdir(exist_ok=True)
    np.save(OUT / "embeddings.npy", vecs)
    json.dump(texts, open(OUT / "texts.json", "w"), indent=0)
    json.dump(lobe_ids, open(OUT / "lobe_ids.json", "w"))
    print(f"[lobe-bank] saved {vecs.shape} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

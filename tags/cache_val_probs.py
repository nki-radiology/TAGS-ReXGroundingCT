"""Run the model over val-200 ONCE and cache the probability maps.

Every post-processing idea tested today -- restriction, the body mask, the spatial prior, the
threshold sweep, per-category thresholds -- re-ran the identical sliding-window inference to
regenerate identical probability maps: ~20 s/case x 200 cases = ~90 min of GPU per experiment,
to evaluate something that is then seconds of numpy. `CACHE_val200_normalized_ct` caches the
normalised CT but NOT the model output, so the expensive half was never cached.

With this, a post-processing experiment is minutes and needs no GPU at all.

Stored as uint16 (prob * 65535) in a 4D .nii.gz, one file per case, channels in index order.
uint16 not uint8: quantisation error is 1.5e-5 instead of 2e-3, and 2e-3 is the same order as
the differences these experiments are trying to resolve (the aux-data effect was -0.0005).
gzip handles the mostly-zero volumes, so the cache stays small despite full resolution.

    sbatch run/py.sh cache_val_probs.py
    python eval_cached.py --restrict --body        # then this, on CPU, in minutes
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_t1_full import (BANK_DIR, CAT2IDX, CODES14, LOBES, TEXT_DIM, T1Model,  # noqa: E402
                           build_model, ts_path, zscore)
from eval_t1_val200 import predict_case  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "checkpoints" / "SUBMISSION_1ch.pt"))
    ap.add_argument("--index", default="t1_index_val.jsonl")
    ap.add_argument("--out", default=str(HERE / "CACHE_val200_probs"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cases", default="",
                    help="json list of case ids to cache (from build_cache_list.py). A random "
                         "prefix of the train split spends the whole budget on the common "
                         "categories -- one pneumothorax, one honeycombing -- which is useless "
                         "for anything conditioned on category.")
    ap.add_argument("--priority", default="",
                    help="json list of cases to cache FIRST. Everything else still gets cached, "
                         "just later -- so a category-covering subset is usable within the hour "
                         "while the long tail fills in behind it.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1,
                    help="split the case list across N jobs; each writes its own files, and the "
                         "cache is keyed by case so shards never collide")
    ap.add_argument("--floor", type=float, default=0.05,
                    help="zero probabilities below this before saving. No sweep goes under 0.10, "
                         "so sub-floor values are never read -- but stored faithfully they cost "
                         "most of the disk, because gzip compresses exact zeros far better than "
                         "low-entropy noise. Lossless for every use we have; do NOT lower a "
                         "threshold sweep past this without rebuilding.")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--step-frac", type=float, default=0.5)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rows = [json.loads(l) for l in open(HERE / "index" / a.index)]
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case"]].append(r)
    cases = sorted(by_case)
    if a.cases:
        want = set(json.load(open(a.cases)))
        cases = [c for c in cases if c in want]
    if a.limit:
        cases = cases[:a.limit]
    if a.priority:
        pri = set(json.load(open(a.priority)))
        cases = [c for c in cases if c in pri] + [c for c in cases if c not in pri]
    if a.nshards > 1:
        cases = cases[a.shard::a.nshards]

    bank = json.load(open(BANK_DIR / "texts.json"))
    emb = np.load(BANK_DIR / "embeddings.npy")
    text2vec = {t: emb[i] for i, t in enumerate(bank)}

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    n_in = int(ck["network_weights"]["encoder.stem.convs.0.conv.weight"].shape[1])
    net, patch = build_model(dev, in_channels=n_in, deep_supervision=True)
    model = T1Model(net, len(CODES14), TEXT_DIM).to(dev)
    from fusion_res import align_to_checkpoint
    if align_to_checkpoint(model.net, ck["network_weights"], patch=int(patch[0])):
        model.to(dev)
    model.net.load_state_dict(ck["network_weights"])
    model.cat_embed.load_state_dict(ck["cat_embed"])
    model.eval()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = {}
    print(f"[cache] {a.ckpt} epoch {ck.get('epoch')} -> {out}  ({len(cases)} cases)", flush=True)
    for i, case in enumerate(cases, 1):
        dst = out / f"{case}.nii.gz"
        rs = sorted(by_case[case], key=lambda r: r["fidx"])
        meta[case] = [{"fidx": r["fidx"], "cat": r["category"], "text": r["text"]} for r in rs]
        if dst.exists():                       # resumable: rerunning skips finished cases
            continue
        im = nib.load(rs[0]["image"])
        vol = zscore(np.asanyarray(im.dataobj).astype(np.float32))
        if n_in == 1:
            img = vol[None].astype(np.float32)
        else:
            p = ts_path(case)
            ts = np.asanyarray(nib.load(str(p)).dataobj) if p.exists() else None
            anat = (np.stack([(ts == l).astype(np.float32) for l in LOBES]) if ts is not None
                    else np.zeros((len(LOBES),) + vol.shape, np.float32))
            img = np.concatenate([vol[None], anat], 0).astype(np.float32)
        embs = np.stack([text2vec[r["text"]] for r in rs])
        cats = [CAT2IDX[r["category"]] for r in rs]
        probs = predict_case(model, img, embs, cats, patch, dev, batch=a.batch,
                             return_prob=True, step_frac=a.step_frac)
        probs = np.where(probs < a.floor, 0.0, probs)
        q = np.clip(probs * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
        nib.save(nib.Nifti1Image(q, im.affine), dst)
        if i % 10 == 0:
            mb = sum(f.stat().st_size for f in out.glob("*.nii.gz")) / 1e6
            print(f"  {i}/{len(cases)}  cache {mb:.0f} MB", flush=True)

    json.dump(meta, open(out / "meta.json", "w"))
    mb = sum(f.stat().st_size for f in out.glob("*.nii.gz")) / 1e6
    print(f"\n  {len(list(out.glob('*.nii.gz')))} cases cached, {mb:.0f} MB total")
    print(f"  -> {out}\n  -> {out/'meta.json'}")


if __name__ == "__main__":
    main()

"""One-time staging: write z-scored fp16 .npy siblings next to every training CT on
/processing, so the trainer's fast-path (train_t1_full.py:593) skips gunzip+zscore.
Run ON the target node. Usage: python stage_npy.py <index.jsonl> [workers]"""
import json, sys, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, nibabel as nib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_t1_full import zscore

SRC_ROOT = os.environ.get("NPY_SRC_ROOT", "")   # e.g. a CT-RATE copy on local disk
DST_ROOT = os.environ.get("NPY_DST_ROOT", "")   # e.g. a shared npy_cache directory
def one(img):
    src = img
    if SRC_ROOT and img.startswith(SRC_ROOT):
        # read the SAME scan from rhea instead of node-local (for staging on gaia)
        src = img.replace(SRC_ROOT, os.environ["NPY_READ_ROOT"], 1)
    out = (img.replace(SRC_ROOT, DST_ROOT, 1) if DST_ROOT else img) + ".norm.npy"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        return 0
    try:
        v = zscore(np.asanyarray(nib.load(src).dataobj).astype(np.float32)).astype(np.float16)
        np.save(out + ".tmp.npy", v)
        os.replace(out + ".tmp.npy", out)
        return 1
    except Exception as e:
        print(f"  FAIL {img}: {e}", flush=True)
        return 0

if __name__ == "__main__":
    idx, workers = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8
    imgs = sorted({json.loads(l)["image"] for l in open(idx)})
    print(f"{len(imgs)} volumes", flush=True)
    done = 0
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(one, imgs), 1):
            done += r
            if i % 200 == 0:
                print(f"  [{i}/{len(imgs)}] {done} written", flush=True)
    print(f"[stage] {done} new .norm.npy files", flush=True)

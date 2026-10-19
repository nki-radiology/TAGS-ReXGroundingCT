"""Build CACHE_val200_normalized_ct: decompressed + z-scored val CTs as float16 .npy.

Evaluation re-runs the same 200 val cases every time, and profiling showed ~20 s of gzip
decompression plus ~7 s of z-scoring per case -- more than the ~25 s of actual inference.
Caching that once turns it into a ~1 s memmap read on every subsequent eval.

Idempotent: cases already cached are skipped, so this can be re-run to top up.

    python build_val_cache.py [--limit N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_t1_base import zscore  # noqa: E402

CACHE = HERE / "CACHE_val200_normalized_ct"


def cache_path(case: str) -> Path:
    return CACHE / f"{case}.npy"


def load_normalized_ct(case: str, nii_path: str) -> np.ndarray:
    """Cached z-scored CT if present, else decompress+normalize (and this is the ONLY place
    that decides, so eval and cache-builder can never disagree)."""
    p = cache_path(case)
    if p.exists():
        return np.load(p, mmap_mode="r")
    return zscore(np.asanyarray(nib.load(nii_path).dataobj).astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(HERE / "index" / "t1_index_val.jsonl")]
    cases = {}
    for r in rows:
        cases.setdefault(r["case"], r["image"])
    items = sorted(cases.items())
    if a.limit:
        items = items[: a.limit]

    done = skip = 0
    t_start = time.time()
    for i, (case, nii) in enumerate(items, 1):
        out = cache_path(case)
        if out.exists():
            skip += 1
            continue
        t0 = time.time()
        vol = zscore(np.asanyarray(nib.load(nii).dataobj).astype(np.float32)).astype(np.float16)
        # write-then-rename so an interrupted build never leaves a half-written .npy.
        # NB: the temp name must itself end in .npy -- np.save silently APPENDS ".npy" to any
        # path that doesn't, which would write "<case>.npy.tmp.npy" and break the rename.
        tmp = out.with_suffix(".tmp.npy")
        np.save(tmp, vol)
        tmp.rename(out)
        done += 1
        if i % 20 == 0 or done <= 3:
            mb = out.stat().st_size / 1e6
            print(f"[{i}/{len(items)}] {case} {vol.shape} {mb:.0f}MB  ({time.time()-t0:.1f}s)",
                  flush=True)

    total_gb = sum(f.stat().st_size for f in CACHE.glob("*.npy")) / 1e9
    print(f"\n[done] cached={done} already_present={skip}  total={total_gb:.1f} GB  "
          f"elapsed={(time.time()-t_start)/60:.1f} min -> {CACHE}", flush=True)


if __name__ == "__main__":
    main()

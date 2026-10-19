"""Average several of our probability-map sets (uint8/255, same grid and finding order) into one uint8 set.
Used to turn N seeds into one ensemble member for the fusion with Rachel's maps.

    python primus/mean_probs.py --dirs a,b,c --out <dir> [--shard i/n]
"""
import argparse
import json
import os
import time

import numpy as np
import nibabel as nib

REX_JSON = os.environ.get("TAGS_CHALLENGE_JSON", "data/ReXGroundingCT/MICCAI_challenge_dataset.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='', help='val|test: restrict to that split of the challenge json (default: all files of the first dir)')
    ap.add_argument('--shard', default='')
    a = ap.parse_args()
    dirs = a.dirs.split(',')
    if a.split:
        d = json.load(open(REX_JSON))[a.split]; names = sorted(e['name'] for e in (d.values() if isinstance(d, dict) else d))
    else:
        names = sorted(f for f in os.listdir(dirs[0]) if f.endswith('.nii.gz'))
    if a.shard:
        i, n = map(int, a.shard.split('/')); names = names[i::n]
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    for k, name in enumerate(names):
        dst = os.path.join(a.out, name)
        if os.path.exists(dst):
            continue
        acc = None; ref = None
        for d in dirs:
            im = nib.load(os.path.join(d, name)); q = np.asarray(im.dataobj, np.float32)
            acc = q if acc is None else acc + q
            if ref is None:
                ref = im
        acc /= len(dirs)
        assert acc.ndim == 4, (name, acc.shape)
        h = ref.header.copy(); h.set_data_dtype(np.uint8); h['scl_slope'], h['scl_inter'] = 1.0, 0.0
        nib.save(nib.Nifti1Image(np.clip(acc + 0.5, 0, 255).astype(np.uint8), ref.affine, h), dst)
        if k % 20 == 0:
            print(f'[{k + 1}/{len(names)}] {(time.time() - t0) / (k + 1):.1f}s/scan', flush=True)
    print(f'done {len(names)} scans -> {a.out}')


if __name__ == '__main__':
    main()

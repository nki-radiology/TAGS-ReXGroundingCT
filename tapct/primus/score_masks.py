"""Fast per-finding Dice of challenge-format mask dirs against the released GT on the original grid, with the
official evaluator's minimum component size — the 'global Dice' of rexrank_eval.py without its instance matching,
so a comparison of several mask dirs takes minutes instead of hours. Paired bootstrap CIs between dirs.

    python primus/score_masks.py --dirs <maskdirA>,<maskdirB>[,...] [--names A,B] [--out json]
"""
import argparse
import json
import os
import time

import numpy as np
import nibabel as nib
from scipy.ndimage import label

REX_JSON = os.environ.get("TAGS_CHALLENGE_JSON", "data/ReXGroundingCT/MICCAI_challenge_dataset.json")
GT_DIR = os.path.join(os.environ.get("TAGS_REXG_ROOT", "data/ReXGroundingCT"), "segmentations")


def min_size(m, k):
    if k <= 0 or not m.any():
        return m
    cc, n = label(m, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(cc.ravel()); keep = sizes >= k; keep[0] = False
    return keep[cc]


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else 0.0


def boot(x, n=3000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', required=True)
    ap.add_argument('--names', default=None)
    ap.add_argument('--min-size', type=int, default=10)
    ap.add_argument('--out', default=None)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    dirs = a.dirs.split(','); names = a.names.split(',') if a.names else [os.path.basename(d.rstrip('/')) for d in dirs]
    entries = json.load(open(REX_JSON))['val']; entries.sort(key=lambda e: e['name'])
    if a.limit:
        entries = entries[:a.limit]
    rows = []; t0 = time.time()
    for i, e in enumerate(entries):
        if not all(os.path.exists(os.path.join(d, e['name'])) for d in dirs):
            print('missing', e['name']); continue
        G = np.asanyarray(nib.load(os.path.join(GT_DIR, e['name'])).dataobj) > 0
        if G.ndim == 3:
            G = G[None]
        Ms = [np.asanyarray(nib.load(os.path.join(d, e['name'])).dataobj) > 0 for d in dirs]
        for f in range(G.shape[0]):
            g = G[f]; row = dict(name=e['name'], fid=str(f), cat=e['categories'][str(f)], gt=int(g.sum()))
            roi = g.copy()
            for M in Ms:
                roi |= M[f]
            if roi.any():
                nz = np.nonzero(roi); sl = tuple(slice(max(0, int(n.min()) - 2), int(n.max()) + 3) for n in nz)
            for nm, M in zip(names, Ms):
                row[nm] = dice(min_size(M[f][sl], a.min_size), g[sl]) if roi.any() else 0.0
            rows.append(row)
        if i % 25 == 0:
            print(f'[{i + 1}/{len(entries)}] {(time.time() - t0) / (i + 1):.1f}s/scan', flush=True)
    print(f'\n{len(rows)} findings')
    for nm in names:
        m, lo, hi = boot([r[nm] for r in rows]); hit = np.mean([r[nm] >= 0.1 for r in rows])
        print(f'  {nm:<28} Dice {m:.4f} [{lo:.4f}, {hi:.4f}]  hit {hit:.3f}')
    for nm in names[1:]:
        m, lo, hi = boot([r[nm] - r[names[0]] for r in rows])
        print(f'  {nm} - {names[0]}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]' + ('  *' if lo > 0 or hi < 0 else ''))
        for lo_v, hi_v, lab in [(0, 1e3, 'gt<1e3 vox'), (1e3, 1e4, '1e3-1e4'), (1e4, 1e12, '>=1e4')]:
            sel = [r for r in rows if lo_v <= r['gt'] < hi_v]
            if sel:
                d = np.mean([r[nm] - r[names[0]] for r in sel])
                print(f'      {lab:<12} n={len(sel):3d} {names[0]} {np.mean([r[names[0]] for r in sel]):.3f} -> {nm} {np.mean([r[nm] for r in sel]):.3f} ({d:+.3f})')
    if a.out:
        json.dump(dict(names=names, rows=rows), open(a.out, 'w'))


if __name__ == '__main__':
    main()

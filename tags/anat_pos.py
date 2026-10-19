"""Anatomy-relative positional encoding: tell the model WHERE IN THE PATIENT it is.

The problem, measured 2026-07-30. VoxTell fuses text and image at a 12^3 = 1728-token bottleneck
and adds `self.pos_embed` -- a FIXED sinusoidal grid registered as a buffer, identical for every
patch and every scan. So when the sliding window sits at the apex, token (0,0,0) receives exactly
the same positional signal as when it sits at the base. At the point where the text attends to the
image, the model has NO signal for where in the patient it is. "Apical" is not representable there;
it can only be inferred indirectly from image content.

That matches every symptom: location-SENSITIVE (textsens 0.246 -- within a patch the text picks
among candidates) but not location-ACCURATE (frac-in-named-lobe 0.15-0.21 against 0.20 chance;
70% of false-positive volume inside the CORRECT lobe, 56% of it >20mm away).

The fix here replaces that fixed grid with coordinates measured from ANATOMY, per scan:

    radius    distance to the nearer hilum      -> perihilar / central / peripheral
    elevation signed superior-inferior offset   -> apical / basilar
    azimuth   signed anterior-posterior offset  -> anterior / posterior
    lr        signed left-right offset          -> side
    dpl       normalised distance to the pleura -> subpleural / peripheral

Hila are the origin because radiological location language is largely hilum-relative, and because
bronchopulmonary SEGMENTS -- named in 39% of findings, and the granularity our errors actually
live at -- are defined by airway branching from the hilum.

Additive, not true RoPE. The transformer already adds `pos` to the keys; making that addend
anatomy-relative tests the hypothesis without reimplementing attention. If it helps, rotary is a
refinement; if it does nothing, rotary would not have saved it.

    python anat_pos.py --self-test
"""
import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LOBES = [10, 11, 12, 13, 14]
LEFT, RIGHT = [10, 11], [12, 13, 14]
MEDIASTINUM = [51, 52, 53, 62, 63]        # heart, aorta, pulmonary vein, vena cavae
N_FEAT = 5                                 # radius, elev, ap, lr, dpl


def axis_map(affine):
    """Which array axis carries L, A, S, and its sign. Read from the affine, never assumed --
    CT-RATE is not consistently oriented and a hardcoded axis silently swaps apex and base."""
    out = {}
    for ax, code in enumerate(nib.aff2axcodes(affine)):
        if code in "LR":
            out["lr"] = (ax, 1 if code == "L" else -1)
        elif code in "AP":
            out["ap"] = (ax, 1 if code == "A" else -1)
        elif code in "SI":
            out["si"] = (ax, 1 if code == "S" else -1)
    return out


def find_hila(ts, spacing):
    """One origin per lung: the lung voxel closest to the mediastinum.

    Crude but stable -- it is the medial root of each lung, which is where the bronchi and
    vessels enter. A precise hilum would need airway-tree segmentation; this is the cheapest
    approximation that is anatomically the right point.
    """
    med = np.isin(ts, MEDIASTINUM)
    if not med.any():
        return None
    dmed = ndimage.distance_transform_edt(~med, sampling=spacing)
    out = {}
    for side, labs in (("L", LEFT), ("R", RIGHT)):
        lung = np.isin(ts, labs)
        if not lung.any():
            return None
        d = np.where(lung, dmed, np.inf)
        out[side] = np.array(np.unravel_index(int(np.argmin(d)), d.shape), np.float32)
    return out


def anat_fields(ts, affine, stride=1):
    """(N_FEAT, X, Y, Z) of anatomy-relative coordinates, on a `stride`-subsampled grid.

    Units: radius/elev/ap/lr in CENTIMETRES (so the numbers sit in a sane range for a sinusoidal
    encoding), dpl normalised to [0,1] inside the lung.
    """
    spacing = np.abs(nib.affines.voxel_sizes(affine))
    sl = tuple(slice(None, None, stride) for _ in range(3))
    t = np.ascontiguousarray(ts[sl])
    sp = spacing * stride
    h = find_hila(t, sp)
    lung = np.isin(t, LOBES)
    shape = t.shape
    if h is None or not lung.any():
        return np.zeros((N_FEAT,) + shape, np.float32), lung

    am = axis_map(affine)
    grid = np.stack(np.meshgrid(*[np.arange(s, dtype=np.float32) for s in shape],
                                indexing="ij"))                      # (3, X, Y, Z)
    # distance to each hilum, then take the nearer one per voxel
    dists, vecs = [], []
    for side in ("L", "R"):
        v = (grid - h[side][:, None, None, None]) * sp[:, None, None, None]
        vecs.append(v)
        dists.append(np.linalg.norm(v, axis=0))
    dists = np.stack(dists)                                          # (2, X, Y, Z)
    near = np.argmin(dists, axis=0)
    radius = np.take_along_axis(dists, near[None], 0)[0]
    vec = np.where(near[None] == 0, vecs[0], vecs[1])                # (3, X, Y, Z)

    sa, ss = am["si"]; aa, asg = am["ap"]; la, lsg = am["lr"]
    dpl = ndimage.distance_transform_edt(lung, sampling=sp).astype(np.float32)
    hi = float(np.percentile(dpl[lung], 99)) or 1.0

    f = np.stack([radius / 10.0,               # cm
                  vec[sa] * ss / 10.0,
                  vec[aa] * asg / 10.0,
                  vec[la] * lsg / 10.0,
                  np.clip(dpl / hi, 0, 1)]).astype(np.float32)
    f[:, ~lung] = 0.0                          # outside the lung the frame is undefined
    return f, lung


def sinusoidal(feats, dim):
    """Encode F continuous coordinates into `dim` channels, F*2*K sinusoids.

    Same family as the fixed encoding it replaces, so the model sees a familiar signal type --
    only the CONTENT changes from patch-relative to patient-relative.
    """
    F = feats.shape[0]
    k = max(dim // (2 * F), 1)
    # Geometric wavelengths from 1 cm to 64 cm. NOT 2**arange(k): with k=204 that is 2**204,
    # which overflows float32 and yields NaN. The band is chosen physically -- 1 cm resolves a
    # small nodule, 64 cm spans the whole thorax, and everything the location words distinguish
    # (segments ~3 cm, lobes ~10 cm) sits inside it.
    lam = 1.0 * (64.0 / 1.0) ** (np.arange(k, dtype=np.float32) / max(k - 1, 1))
    freqs = (2.0 * np.pi / lam)[:, None]
    out = []
    for i in range(F):
        x = feats[i].reshape(1, -1) * freqs                          # (k, n)
        out.append(np.sin(x)); out.append(np.cos(x))
    e = np.concatenate(out, 0)                                       # (F*2*k, n)
    if e.shape[0] < dim:
        e = np.concatenate([e, np.zeros((dim - e.shape[0], e.shape[1]), np.float32)], 0)
    return e[:dim].T.reshape(feats.shape[1:] + (dim,)).astype(np.float32)


def token_pos(fields, fstride, origin, patch, grid, dim):
    """Positional encoding for the bottleneck token grid of ONE crop.

    `fields` is the per-CASE coordinate stack from `anat_fields(..., stride=fstride)`, computed
    ONCE and reused for every crop of that scan. Recomputing it per crop would run three
    full-volume distance transforms per training sample -- seconds each, and the loader is
    already the bottleneck (measured: 33% GPU utilisation on the 1-GPU configs).

    `origin` is the crop's corner in FULL-resolution voxels; `grid` the bottleneck side (12), so
    each token covers patch/grid voxels. Returns (grid^3, dim), ready to add to the keys.
    """
    step = patch // grid
    cen = [(origin[d] + (np.arange(grid) + 0.5) * step) / fstride for d in range(3)]
    idx = [np.clip(c, 0, fields.shape[d + 1] - 1).astype(int) for d, c in enumerate(cen)]
    sub = fields[:, idx[0][:, None, None], idx[1][None, :, None], idx[2][None, None, :]]
    return sinusoidal(sub, dim).reshape(grid ** 3, dim)


def _self_test():
    from train_t1_full import ts_path
    rows = [json.loads(l) for l in open(HERE / "index" / "t1_index_val.jsonl")]
    seen = 0
    for r in rows:
        p = ts_path(r["case"])
        if not p.exists():
            continue
        im = nib.load(r["image"])
        ts = np.asanyarray(nib.load(str(p)).dataobj)
        f, lung = anat_fields(ts, im.affine, stride=4)
        if not lung.any():
            continue
        gt = np.asanyarray(nib.load(r["seg"]).dataobj)
        gt = (gt[r["fidx"]] if gt.ndim == 4 else gt)[::4, ::4, ::4] > 0
        g = gt[:f.shape[1], :f.shape[2], :f.shape[3]] & lung
        if g.sum() < 5:
            continue
        names = ["radius(cm)", "elev(cm)", "ap(cm)", "lr(cm)", "dpl"]
        print(f"\n[{seen+1}] {r['category']}  {r['text'][:88]}")
        for i, n in enumerate(names):
            print(f"    {n:11s} finding {f[i][g].mean():+7.2f}   lung {f[i][lung].mean():+7.2f}")
        seen += 1
        if seen >= 4:
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()

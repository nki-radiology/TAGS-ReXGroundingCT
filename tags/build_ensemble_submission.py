"""Average several members' TEST probability maps, restrict, threshold once, write the submission.

Three things this must get right, each of which has already gone wrong once:

* **Average as PROBABILITIES, threshold once.** Thresholding each member and voting discards the
  confidence that makes averaging work -- a false positive is usually one model's idiosyncratic
  error, and it vanishes because the others contribute near-zero to the MEAN, not because they
  outvote it.
* **Apply the anatomical restriction HERE.** `predict_test.py --save-prob` returns before the
  block that applies it, so the cached test probabilities are UNRESTRICTED -- while every val
  number quoted for these ensembles (0.3473 for four members, 0.3506 for six) was measured WITH
  it. It is the single largest post-processing gain in the project: +0.0124 overall and +0.0687
  on category-ambiguous findings, where it supplies the location the model does not. A zip built
  without it validates cleanly and simply scores worse.
* **Do not re-tune the threshold.** It comes from the val subset search; picking it on test would
  be selection on the thing being scored.

    python build_ensemble_submission.py --members aux_2e,aux_dil2,cautious,fusion24 \
        --thr 0.35 --out submission/ENS4
"""
import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_t1_val200 import named_region, LEFT, RIGHT                    # noqa: E402


def side_region(text):
    """SIDE-level restriction instead of lobe-level.

    Lobe restriction clips true positives at the fissures. Measured on val, the better-placed the
    model the more that costs: side-lobe is -0.013 on `nograft` (worst model) but +0.007 on
    `cautious` and +0.008..+0.013 on the 5-member ENSEMBLE, which is what we ship. A finding whose
    text names one side is restricted to that whole lung; anything else keeps the lobe rule.
    """
    reg = named_region(text)
    if not reg:
        return None
    s_ = set(reg); l, r = s_ & set(LEFT), s_ & set(RIGHT)
    if l and not r:
        return LEFT
    if r and not l:
        return RIGHT
    return reg                                   # bilateral / unresolved -> unchanged
from train_t1_full import ts_path                                       # noqa: E402

from paths import CHALLENGE_JSON  # noqa: E402
DS = str(CHALLENGE_JSON)


def _region_mask(ts, text, side, dilate_k, cache, side_k=0):
    """Per-finding restriction mask. side_k>0: named SIDE dilated in-plane (the 2026-08-23
    winner). dilate_k>0: named lobes dilated (side fallback).
    LOBE_K env (>0): if the text names a lobe (word 'lobe' present) restrict to the named lobe(s)
    dilated in-plane by LOBE_K, else fall through to the side rule (§71: +0.0038 val at k=16)."""
    import os as _os, re as _re
    _lk = int(_os.environ.get("LOBE_K", "0"))
    if _lk > 0 and _re.search(r"\blobe\b", text, _re.I):
        lob = named_region(text)
        if lob:
            key = ("lobek", tuple(lob), _lk)
            if key not in cache:
                from scipy import ndimage
                st = ndimage.generate_binary_structure(2, 1)
                m = np.isin(ts, lob)
                cache[key] = np.stack([ndimage.binary_dilation(m[:, :, z], st, iterations=_lk)
                                       for z in range(m.shape[2])], axis=2)
            return cache[key]
    if side_k > 0:
        reg = side_region(text)
        if reg is None:
            return None
        key = ("sdil", tuple(reg), side_k)
        if key not in cache:
            from scipy import ndimage
            st = ndimage.generate_binary_structure(2, 1)
            m = np.isin(ts, reg)
            cache[key] = np.stack([ndimage.binary_dilation(m[:, :, z], st, iterations=side_k)
                                   for z in range(m.shape[2])], axis=2)
        return cache[key]
    if dilate_k > 0:
        lob = named_region(text)
        if lob:
            key = (tuple(lob), dilate_k)
            if key not in cache:
                from scipy import ndimage
                st = ndimage.generate_binary_structure(2, 1)
                m = np.isin(ts, lob)
                cache[key] = np.stack([ndimage.binary_dilation(m[:, :, z], st, iterations=dilate_k)
                                       for z in range(m.shape[2])], axis=2)
            return cache[key]
    reg = side_region(text) if side else named_region(text)
    if reg is None:
        return None
    key = ("flat", tuple(reg))
    if key not in cache:
        cache[key] = np.isin(ts, reg)
    return cache[key]


_THRM = None

def _thr_features(q, text, category, regvox, lungvox):
    """MUST match thr_predictor2.feats() exactly -- any drift silently degrades the prediction."""
    import re
    from scipy import ndimage as _nd
    from eval_t1_val200 import named_region as _nr
    CATS = ["1a","1b","1c","1d","1e","1f","2a","2b","2c","2d","2e","2f","2g","2h"]
    PLUR = re.compile(r"\b(nodules|opacities|densities|lesions|areas|multiple|several|few|scattered|"
                      r"numerous|both|bilateral|patchy|multifocal|diffuse|foci)\b", re.I)
    SEG = re.compile(r"\b(posterobasal|laterobasal|mediobasal|apicoposterior|apical|lingul|subpleural|"
                     r"peripheral|central|basal|segment)\b", re.I)
    SEV = re.compile(r"\b(minimal|mild|slight|subtle|faint)\b", re.I)
    EXT = re.compile(r"\b(extensive|widespread|prominent|marked|severe|large)\b", re.I)
    SMALL = re.compile(r"\b(subcentimeter|millimetric|tiny|small|punctate)\b", re.I)
    v = {t: int((q > t).sum()) for t in (0.15, 0.25, 0.35, 0.5, 0.65)}
    if v[0.35] == 0:
        return None
    lab, n = _nd.label(q > 0.35)
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
    pz = np.where((q > 0.35).any(axis=(0, 1)))[0]
    inp = np.sqrt(max(int((q > 0.35).any(axis=2).sum()), 1))
    f = dict(mass=float(q.sum()), peak=float(q.max()),
             v15=v[0.15], v25=v[0.25], v35=v[0.35], v50=v[0.5], v65=v[0.65],
             r_15_35=v[0.15]/max(v[0.35],1), r_50_35=v[0.5]/max(v[0.35],1),
             r_65_35=v[0.65]/max(v[0.35],1), mass_over_v35=float(q.sum())/max(v[0.35],1),
             mean_in_mask=float(q[q > 0.35].mean()),
             ncomp=n, big_frac=float(sizes.max()/max(sizes.sum(),1)),
             med_comp=float(np.median(sizes)), zext=len(pz), inplane=float(inp),
             z_over_in=len(pz)/max(inp,1), v35_over_reg=v[0.35]/max(regvox,1),
             reg_over_lung=regvox/max(lungvox,1),
             plural=int(bool(PLUR.search(text))), seg=int(bool(SEG.search(text))),
             lobe=int(bool(_nr(text))), sev=int(bool(SEV.search(text))),
             ext=int(bool(EXT.search(text))), small=int(bool(SMALL.search(text))),
             nwords=len(text.split()))
    for c in CATS:
        f[f"cat_{c}"] = int(category == c)
    return f


def _predict_thr(q, text, category, regvox, lungvox, path):
    global _THRM
    if _THRM is None:
        import pickle
        _THRM = pickle.load(open(path, "rb"))
    f = _thr_features(q, text, category, regvox, lungvox)
    if f is None:
        return None
    x = np.array([[f[k] for k in _THRM["keys"]]], np.float32)
    t = _THRM["w_gbm"] * _THRM["gbm"].predict(x)[0] + _THRM["w_ridge"] * _THRM["ridge"].predict(x)[0]
    g = _THRM["grid"]
    return float(np.clip(t, g.min(), g.max()))


def _one_case(args):
    """One case, identical to the serial loop body. Returns the counters it would have updated."""
    e, mem, dirs, out, thr, side, no_restrict, dilate_k, side_k, thr_model, tap_dir, tap_w = args
    from pathlib import Path as _P
    out = _P(out)
    stem = e["name"][:-7] if e["name"].endswith(".nii.gz") else e["name"]
    acc, ref = None, None
    for m in mem:
        im = nib.load(_P(dirs[m]) / f"{stem}.nii.gz")
        q = np.asarray(im.dataobj, np.float32) / 65535.0
        acc = q if acc is None else acc + q
        if ref is None:
            ref = im
    acc /= len(mem)
    import os as _os
    _twm = _os.environ.get("TAPW_MODEL", "")            # per-finding TAP weight model (pkl); empty = fixed tap_w
    ours = acc.copy() if (tap_dir and _twm) else None
    if tap_dir:
        tq = np.asarray(nib.load(_P(tap_dir) / f"{stem}.nii.gz").dataobj, np.float32) / 255.0
        acc = (1.0 - tap_w) * acc + tap_w * tq          # TAGS+TAP weighted fusion

    n_restricted = vox_before = vox_after = 0
    regs = {}                       # finding channel -> TRUE region size (mask voxels)
    _rmask = {}; _lobemask = {}
    keys = sorted(e["findings"], key=int)
    if not no_restrict:
        tsp = ts_path(stem)
        ts = np.asanyarray(nib.load(str(tsp)).dataobj) if tsp.exists() else None
        if ts is not None and ts.shape == acc.shape[1:]:
            _mc = {}
            _lts = _os.environ.get("LOBE_THR_SIDE", "") == "1"; _lobemask = {}
            for j, k in enumerate(keys):
                msk = _region_mask(ts, e["findings"][k], side, dilate_k, _mc, side_k)
                if msk is None:
                    continue
                if _lts:                                   # thresholds from the side map; lobe mask applied at the end
                    _lk_save = _os.environ.pop("LOBE_K", None); msk_side = _region_mask(ts, e["findings"][k], side, dilate_k, _mc, side_k); _os.environ["LOBE_K"] = _lk_save or "0"
                    _lobemask[j] = msk; msk = msk_side
                b = int((acc[j] >= thr).sum())
                acc[j] = np.where(msk, acc[j], 0.0)
                regs[j] = max(int(msk.sum()), 1)
                if ours is not None: _rmask[j] = msk
                vox_before += b
                vox_after += int((acc[j] >= thr).sum())
            n_restricted = 1

    if ours is not None:
        import pickle as _pk
        from thr_predictor2 import CATS as _CATS
        _W = _pk.load(open(_twm, "rb")); lungv0 = max(int(np.isin(ts, [10, 11, 12, 13, 14]).sum()), 1) if ts is not None else 1
        for j, k in enumerate(keys):
            m_ = _rmask.get(j); o_ = np.where(m_, ours[j], 0.0) if m_ is not None else ours[j]; t_ = np.where(m_, tq[j], 0.0) if m_ is not None else tq[j]
            f = _thr_features(acc[j], e["findings"][k], (e.get("categories") or {}).get(k, ""), regs.get(j, lungv0), lungv0)
            if f is None: continue
            f = dict(f); f["tap_mass"] = float(t_.sum()); f["tap_v35"] = int((t_ > 0.35).sum()); f["abs_diff_mass"] = float(np.abs(o_ - t_).sum()); f["ours_v35"] = int((o_ > 0.35).sum())
            x = np.array([[f[kk] for kk in _W["keys"]] + [int((e.get("categories") or {}).get(k, "") == c) for c in _CATS]], np.float32)
            w_ = float(np.clip(_W["gbm"].predict(x)[0], _W["wmin"], _W["wmax"]))
            acc[j] = (1.0 - w_) * o_ + w_ * t_
    if thr_model:
        lungv = max(int(np.isin(ts, [10, 11, 12, 13, 14]).sum()), 1) if ts is not None else 1
        chans, n_pred, t_sum = [], 0, 0.0
        for j, k in enumerate(keys):
            t_ = _predict_thr(acc[j], e["findings"][k],
                              (e.get("categories") or {}).get(k, ""),
                              regs.get(j, lungv), lungv, thr_model)
            if t_ is not None:
                n_pred += 1; t_sum += t_
            ch = (acc[j] >= (t_ if t_ is not None else thr))
            if j in _lobemask: ch = ch & _lobemask[j]
            chans.append(ch.astype(np.uint8))
        msk = np.stack(chans)
    else:
        msk = (acc >= thr).astype(np.uint8)
    nf = len(e["findings"])
    if msk.shape[0] != nf:
        raise SystemExit(f"{stem}: {msk.shape[0]} channels vs {nf} findings")
    empty = int(sum(1 for j in range(nf) if msk[j].sum() == 0))

    h = ref.header.copy()
    h.set_data_dtype(np.uint8)
    h["scl_slope"], h["scl_inter"] = 1.0, 0.0
    nib.save(nib.Nifti1Image(msk, ref.affine, h), out / f"{stem}.nii.gz")
    return dict(empty=empty, tot=nf, n_restricted=n_restricted,
                vox_before=vox_before, vox_after=vox_after,
                thr_pred=(n_pred if thr_model else 0),
                thr_sum=(t_sum if thr_model else 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True)
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--side", action="store_true",
                    help="restrict to the named SIDE rather than the named LOBE (+0.008 on the "
                         "ensemble; lobe clips true positives at the fissures)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallelise over cases; the work is gzip decompression, not arithmetic")
    ap.add_argument("--thr-model", default="",
                    help="pickle from thr_predictor2.py: predict a PER-FINDING threshold from "
                         "GT-free features of the probability map instead of using --thr. "
                         "Out-of-fold val gain +0.0072 (2026-08-25).")
    ap.add_argument("--dilate-side", type=int, default=0,
                    help="restrict to the named SIDE dilated this many voxels in-plane. Sweep "
                         "2026-08-23: sdil12 0.3650 vs side 0.3547 vs dilated-lobe 0.3626 -- the "
                         "pleural tolerance band beats lobe specificity (our hypothesis).")
    ap.add_argument("--dilate-lobe", type=int, default=0,
                    help="restrict to the NAMED LOBES dilated this many voxels in-plane (fold-"
                         "validated +0.007 over side at k=12..26; sweep results/restrict_dilate)."
                         " Findings naming only a side keep the side mask; needs --side too.")
    ap.add_argument("--no-restrict", action="store_true",
                    help="skip the anatomical restriction (for measuring what it is worth)")
    ap.add_argument("--tap", default="",
                    help="directory of an external member's prob maps (uint8/255, same grid/order); "
                         "fused as (1-w)*mean(members) + w*TAP. Kevin's TAP-CT: submitted w=0.2.")
    ap.add_argument("--tap-w", type=float, default=0.2)
    ap.add_argument("--start", type=int, default=0, help="case range for sharded builds")
    ap.add_argument("--end", type=int, default=0)
    a = ap.parse_args()

    mem = [m.strip() for m in a.members.split(",") if m.strip()]
    dirs = {m: HERE / f"TESTPROB_{m}" for m in mem}
    for m, d in dirs.items():
        n = len(list(d.glob("*.nii.gz")))
        if n != 300:
            raise SystemExit(f"{m}: {n}/300 test probability maps -- incomplete, refusing to build")
    if a.tap:
        _tn = len(list(Path(a.tap).glob("*.nii.gz")))
        if _tn != 300:
            raise SystemExit(f"--tap: {_tn}/300 maps -- incomplete, refusing to build")
        print(f"[ens-sub] TAP fusion: w={a.tap_w} dir={a.tap}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[ens-sub] {len(mem)} members {mem}", flush=True)
    print(f"[ens-sub] thr {a.thr}  restriction "
          f"{'OFF' if a.no_restrict else (f'DILATED-SIDE k={a.dilate_side}' if a.dilate_side else (f'DILATED-LOBE k={a.dilate_lobe} (side fallback)' if a.dilate_lobe else ('SIDE' if a.side else 'LOBE')))}"
          f"  -> {out}", flush=True)

    d = json.load(open(DS))
    items = list(d["test"].values()) if isinstance(d["test"], dict) else d["test"]
    if a.start or a.end:
        items = items[a.start:(a.end or len(items))]
        print(f"[ens-sub] SHARD cases {a.start}:{a.end or 'end'} ({len(items)} cases)", flush=True)
    empty = tot = n_restricted = 0
    vox_before = vox_after = 0

    # One case = 6 gzipped probability maps of ~100 MB each. The work is gzip decompression, not
    # arithmetic, so cases go to a process pool; the per-case body below is unchanged.
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        _tapd = str(Path(a.tap)) if a.tap else ""
        args = [(e, mem, {m: str(v) for m, v in dirs.items()}, str(out),
                 a.thr, a.side, a.no_restrict, a.dilate_lobe, a.dilate_side, a.thr_model, _tapd, a.tap_w)
                for e in items]
        done = 0
        thr_pred, thr_sum = 0, 0.0
        with ProcessPoolExecutor(a.workers) as ex:
            futs = [ex.submit(_one_case, x) for x in args]
            for f in as_completed(futs):
                r = f.result()
                empty += r["empty"]; tot += r["tot"]; n_restricted += r["n_restricted"]
                vox_before += r["vox_before"]; vox_after += r["vox_after"]
                thr_pred += r.get("thr_pred", 0); thr_sum += r.get("thr_sum", 0.0)
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(items)}", flush=True)
        items_done = done
        print(f"\n  {items_done} cases, {tot} findings, {empty} empty masks "
              f"({100*empty/max(tot,1):.1f}%)")
        if not a.no_restrict:
            print(f"  restriction applied on {n_restricted}/{items_done} cases; it removed "
                  f"{vox_before-vox_after:,} of {vox_before:,} predicted voxels "
                  f"({100*(vox_before-vox_after)/max(vox_before,1):.1f}%)")
        if a.thr_model:
            print(f"  thr-model: predicted threshold on {thr_pred}/{tot} findings "
                  f"(mean {thr_sum/max(thr_pred,1):.3f}); fallback {a.thr} on the rest")
            if thr_pred == 0:
                raise SystemExit("--thr-model given but 0 predictions -- refusing")
        _stems = {(e["name"][:-7] if e["name"].endswith(".nii.gz") else e["name"]) for e in items}
        n = sum(1 for p_ in out.glob('*.nii.gz') if p_.name[:-7] in _stems)   # count THIS shard only
        if n != len(items):
            raise SystemExit(f"wrote {n}/{len(items)} masks -- refusing to report success")
        print(f"  -> {out}")
        return

    if a.thr_model:
        raise SystemExit("--thr-model requires --workers > 1 (serial path does not apply it)")

    for i, e in enumerate(items, 1):
        stem = e["name"][:-7] if e["name"].endswith(".nii.gz") else e["name"]
        acc, ref = None, None
        for m in mem:
            im = nib.load(dirs[m] / f"{stem}.nii.gz")
            q = np.asarray(im.dataobj, np.float32) / 65535.0
            acc = q if acc is None else acc + q
            if ref is None:
                ref = im
        acc /= len(mem)
        if a.tap:
            tq = np.asarray(nib.load(Path(a.tap) / f"{stem}.nii.gz").dataobj, np.float32) / 255.0
            acc = (1.0 - a.tap_w) * acc + a.tap_w * tq

        keys = sorted(e["findings"], key=int)
        if not a.no_restrict:
            tsp = ts_path(stem)
            ts = np.asanyarray(nib.load(str(tsp)).dataobj) if tsp.exists() else None
            if ts is not None and ts.shape == acc.shape[1:]:
                _mc = {}
                for j, k in enumerate(keys):
                    msk = _region_mask(ts, e["findings"][k], a.side, a.dilate_lobe, _mc,
                                       a.dilate_side)
                    if msk is None:
                        continue
                    b = int((acc[j] >= a.thr).sum())
                    acc[j] = np.where(msk, acc[j], 0.0)
                    vox_before += b
                    vox_after += int((acc[j] >= a.thr).sum())
                n_restricted += 1

        msk = (acc >= a.thr).astype(np.uint8)
        nf = len(e["findings"])
        if msk.shape[0] != nf:
            raise SystemExit(f"{stem}: {msk.shape[0]} channels vs {nf} findings")
        empty += int(sum(1 for j in range(nf) if msk[j].sum() == 0))
        tot += nf

        h = ref.header.copy()
        h.set_data_dtype(np.uint8)
        h["scl_slope"], h["scl_inter"] = 1.0, 0.0     # a label map must not be rescaled
        nib.save(nib.Nifti1Image(msk, ref.affine, h), out / f"{stem}.nii.gz")
        if i % 50 == 0:
            print(f"  {i}/{len(items)}", flush=True)

    print(f"\n  {len(items)} cases, {tot} findings, {empty} empty masks "
          f"({100*empty/max(tot,1):.1f}%)")
    if not a.no_restrict:
        print(f"  restriction applied on {n_restricted}/{len(items)} cases; it removed "
              f"{vox_before-vox_after:,} of {vox_before:,} predicted voxels "
              f"({100*(vox_before-vox_after)/max(vox_before,1):.1f}%)")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()

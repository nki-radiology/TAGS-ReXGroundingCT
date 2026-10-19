"""Threshold predictor, take 2: 8x the data, smoothed target, stronger model.

Take 1 (thr_predictor.py) got +0.0014 out-of-fold from 379 examples / 41 features, and I flagged
two possible causes: too little data, and a noisy target (argmax of a per-finding Dice curve fits
noise). Both are addressed here at zero GPU cost:

  DATA   every cached MEMBER model gives a different probability map for the same finding, each
         with its own oracle threshold. Features are map-intrinsic, so those are valid training
         examples. 8 members x ~380 findings ~= 3000, and the model is still EVALUATED on the
         ENS5 ensemble map -- the thing we actually threshold at submission time.
  TARGET instead of the argmax, a Dice-weighted average of thresholds (soft target), which is
         stable when the curve is flat near its peak.
  MODEL  bigger GBM + a Ridge fallback, averaged.

Evaluation is unchanged and honest: case-grouped 5-fold CV, score = the Dice the predicted
threshold ACTUALLY achieves on held-out ENS5 findings.
"""
import json, sys, re, collections
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, nibabel as nib
from scipy import ndimage

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
from train_t1_full import ts_path                                   # noqa: E402
from build_ensemble_submission import side_region                   # noqa: E402
from eval_t1_val200 import named_region                             # noqa: E402
DK = 12
GRID = np.round(np.arange(0.10, 0.86, 0.05), 2)
CATS = ["1a","1b","1c","1d","1e","1f","2a","2b","2c","2d","2e","2f","2g","2h"]
MEMBERS = ["aux_2e", "aux_dil2", "cautious", "fusion24", "nograft", "cleangraft",
           "textaug", "copypaste"]
TARGET = "ENS5"

# Env overrides so the SAME extraction runs on the train-1000 caches:
#   TP2_INDEX=index/t1_train1000_local.jsonl TP2_PREFIX=CACHE_train_ TP2_MEMBERS=aux_2e_vonly,...
#   TP2_TARGET=__MEAN__ (per-finding mean of the members = the ensemble map)
#   TP2_DUMP=results/thr_examples_train.pkl (extract only, no training)  TP2_WORKERS=32
import os
INDEX = os.environ.get("TP2_INDEX", "index/t1_index_val.jsonl")
CACHE_PREFIX = os.environ.get("TP2_PREFIX", "CACHE_sf25_")
if os.environ.get("TP2_MEMBERS"):
    MEMBERS = os.environ["TP2_MEMBERS"].split(",")
TARGET = os.environ.get("TP2_TARGET", TARGET)
DUMP = os.environ.get("TP2_DUMP", "")
WORKERS = int(os.environ.get("TP2_WORKERS", "6"))
CROPS = bool(int(os.environ.get("TP2_CROPS", "0")))
TAP = os.environ.get("TP2_TAP", "")          # dir of external uint8/255 val maps; __MEAN__ becomes (1-w)*mean+w*TAP
TAPW = float(os.environ.get("TP2_TAPW", "0.3"))
OUT = os.environ.get("TP2_OUT", "results/thr_model.pkl")
BASE = os.environ.get("TP2_BASE", "")        # existing pkl to score on the same TARGET records (baseline)

PLUR = re.compile(r"\b(nodules|opacities|densities|lesions|areas|multiple|several|few|scattered|"
                  r"numerous|both|bilateral|patchy|multifocal|diffuse|foci)\b", re.I)
SEG = re.compile(r"\b(posterobasal|laterobasal|mediobasal|apicoposterior|apical|lingul|subpleural|"
                 r"peripheral|central|basal|segment)\b", re.I)
SEV = re.compile(r"\b(minimal|mild|slight|subtle|faint)\b", re.I)
EXT = re.compile(r"\b(extensive|widespread|prominent|marked|severe|large)\b", re.I)
SMALL = re.compile(r"\b(subcentimeter|millimetric|tiny|small|punctate)\b", re.I)

def feats(q, gt, r, regvox, lungvox):
    v = {t: int((q > t).sum()) for t in (0.15, 0.25, 0.35, 0.5, 0.65)}
    if v[0.35] == 0:
        return None, None
    lab, n = ndimage.label(q > 0.35)
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
    pz = np.where((q > 0.35).any(axis=(0, 1)))[0]
    inp = np.sqrt(max(int((q > 0.35).any(axis=2).sum()), 1))
    txt = r["text"]
    f = dict(mass=float(q.sum()), peak=float(q.max()),
             v15=v[0.15], v25=v[0.25], v35=v[0.35], v50=v[0.5], v65=v[0.65],
             r_15_35=v[0.15]/max(v[0.35],1), r_50_35=v[0.5]/max(v[0.35],1),
             r_65_35=v[0.65]/max(v[0.35],1), mass_over_v35=float(q.sum())/max(v[0.35],1),
             mean_in_mask=float(q[q > 0.35].mean()),
             ncomp=n, big_frac=float(sizes.max()/max(sizes.sum(),1)),
             med_comp=float(np.median(sizes)), zext=len(pz), inplane=float(inp),
             z_over_in=len(pz)/max(inp,1), v35_over_reg=v[0.35]/regvox,
             reg_over_lung=regvox/lungvox,
             plural=int(bool(PLUR.search(txt))), seg=int(bool(SEG.search(txt))),
             lobe=int(bool(named_region(txt))), sev=int(bool(SEV.search(txt))),
             ext=int(bool(EXT.search(txt))), small=int(bool(SMALL.search(txt))),
             nwords=len(txt.split()))
    for c in CATS:
        f[f"cat_{c}"] = int(r["category"] == c)
    gts = int(gt.sum())
    curve = []
    for t in GRID:
        mm = q > t
        s = int(mm.sum()) + gts
        curve.append(2.0*int((mm & gt).sum())/s if s else 0.0)
    return f, np.array(curve)

def _one(args):
    case, rs = args
    tp = ts_path(case)
    if not Path(tp).exists():
        return []
    caches = {}
    want = MEMBERS if TARGET == "__MEAN__" else MEMBERS + [TARGET]
    for m in want:
        p = H / f"{CACHE_PREFIX}{m}" / f"{case}.nii.gz"
        if p.exists():
            caches[m] = p
    if TARGET == "__MEAN__":
        if len(caches) < len(MEMBERS):        # ensemble target needs EVERY member
            return []
    elif TARGET not in caches:
        return []
    if TAP and not (Path(TAP) / f"{case}.nii.gz").exists():
        return []
    tapimg = nib.load(str(Path(TAP) / f"{case}.nii.gz")) if TAP else None
    ts = np.asanyarray(nib.load(str(tp)).dataobj)
    lungvox = max(int(np.isin(ts, [10,11,12,13,14]).sum()), 1)
    segf = nib.load(rs[0]["seg"]).dataobj
    ens_q = {}                                # fidx -> running sum for __MEAN__ target
    st = ndimage.generate_binary_structure(2, 1)
    out = []
    for r in rs:
        try:
            gt = np.asanyarray(segf[r["fidx"]] if len(segf.shape) == 4 else segf) > 0
            if gt.sum() < 20:
                continue
            mask, regvox = None, lungvox
            if int(os.environ.get("LOBE_K", "0")) > 0:
                from restrict_rule import restrict_mask
                mask = restrict_mask(ts, r["text"], side_k=DK)
            else:
                sid = side_region(r["text"])
                if sid is not None:
                    mm = np.isin(ts, sid)
                    mask = np.stack([ndimage.binary_dilation(mm[:, :, z], st, iterations=DK)
                                     for z in range(mm.shape[2])], axis=2)
            if mask is not None:
                regvox = max(int(mask.sum()), 1)
            qsum = None
            for m, p in caches.items():
                q = np.asanyarray(nib.load(p).dataobj[r["fidx"]]).astype(np.float32)/65535.0
                if q.shape != gt.shape:
                    continue
                if mask is not None:
                    q = np.where(mask, q, 0.0)
                if TARGET == "__MEAN__":
                    qsum = q.copy() if qsum is None else qsum + q
                f, curve = feats(q, gt, r, regvox, lungvox)
                if f is None:
                    continue
                rec = dict(case=case, cat=r["category"], model=m, fidx=r["fidx"],
                           feat=f, curve=curve.tolist())
                if CROPS:
                    nz = np.argwhere(q > 0.15)
                    if len(nz):
                        lo, hi = nz.min(0), nz.max(0) + 1
                        sub = q[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
                        zf = [48 / max(d, 1) for d in sub.shape]
                        rec["crop"] = ndimage.zoom(sub, zf, order=1).astype(np.float16)
                        rec["bbox_shape"] = [int(x) for x in sub.shape]
                out.append(rec)
            if TARGET == "__MEAN__" and qsum is not None:
                qm = qsum/len(caches)
                if tapimg is not None:
                    t = np.asanyarray(tapimg.dataobj[r["fidx"]]).astype(np.float32)/255.0
                    if t.shape != qm.shape:
                        continue
                    if mask is not None:
                        t = np.where(mask, t, 0.0)
                    qm = (1.0-TAPW)*qm + TAPW*t
                f, curve = feats(qm, gt, r, regvox, lungvox)
                if f is not None:
                    out.append(dict(case=case, cat=r["category"], model="__MEAN__",
                                    feat=f, curve=curve.tolist()))
        except Exception:
            continue
    return out

def main():
    rows = [json.loads(l) for l in open(H / INDEX)]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["case"]].append(r)
    res = []
    with ProcessPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(_one, it) for it in by.items()]
        for i, f in enumerate(as_completed(futs), 1):
            res.extend(f.result())
            if i % 25 == 0:
                print(f"  [{i}/{len(by)}] {len(res)} examples", flush=True)
    print(f"[data] {len(res)} examples ({len(MEMBERS)+1} maps x findings)", flush=True)
    if DUMP:
        import pickle
        pickle.dump(res, open(H / DUMP, "wb"))
        print(f"[dump] -> {DUMP}  (no training in dump mode)", flush=True)
        return
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    keys = sorted(res[0]["feat"])
    X = np.array([[r["feat"][k] for k in keys] for r in res], np.float32)
    C = np.array([r["curve"] for r in res], np.float32)
    isT = np.array([r["model"] == TARGET for r in res])
    cases = np.array([r["case"] for r in res])
    ucase = sorted(set(cases)); fold = {c: i % 5 for i, c in enumerate(ucase)}
    fo = np.array([fold[c] for c in cases])
    # SOFT target: Dice-weighted mean threshold (stable where the curve is flat)
    w = np.clip(C - C.max(1, keepdims=True) + 0.05, 0, None)
    y = (w * GRID[None, :]).sum(1) / np.maximum(w.sum(1), 1e-9)
    i35 = int(np.where(GRID == 0.35)[0][0])
    def dice_at(t, sel):
        idx = np.abs(GRID[None, :] - t[:, None]).argmin(1)
        return C[np.where(sel)[0], idx]
    print(f"\n=== take 2: {int(isT.sum())} ENS5 findings evaluated, {int((~isT).sum())} member examples for training ===")
    print(f"  fixed 0.35   {C[isT, i35].mean():.4f}")
    print(f"  ORACLE       {C[isT].max(1).mean():.4f}")
    if BASE:
        import pickle as _pk
        B = _pk.load(open(H / BASE, "rb")); Xb = np.array([[r["feat"][k] for k in B["keys"]] for r in res], np.float32)
        pb = np.clip(B["w_gbm"]*B["gbm"].predict(Xb[isT]) + B["w_ridge"]*B["ridge"].predict(Xb[isT]), GRID.min(), GRID.max())
        db = dice_at(pb, isT)
        print(f"  BASELINE {BASE}: {db.mean():.4f}  ({db.mean()-C[isT, i35].mean():+.4f})")
    for label, use_members in (("ENS5 only", False), ("ENS5 + 8 members", True)):
        oof = np.zeros(int(isT.sum()), np.float32); pos = 0
        for k in range(5):
            tr = (fo != k) & (isT | use_members)
            te = (fo == k) & isT
            g = GradientBoostingRegressor(n_estimators=600, max_depth=4, learning_rate=0.03,
                                          subsample=0.8, random_state=0).fit(X[tr], y[tr])
            rg = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X[tr], y[tr])
            pred = np.clip(0.7*g.predict(X[te]) + 0.3*rg.predict(X[te]), GRID.min(), GRID.max())
            d = dice_at(pred, te)
            oof[pos:pos+len(d)] = d; pos += len(d)
        print(f"  LEARNED ({label}): {oof.mean():.4f}  ({oof.mean()-C[isT, i35].mean():+.4f})")
    # FIT ON EVERYTHING and persist, for use on the test set by build_ensemble_submission.py
    import pickle
    gA = GradientBoostingRegressor(n_estimators=600, max_depth=4, learning_rate=0.03,
                                   subsample=0.8, random_state=0).fit(X, y)
    rA = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X, y)
    pickle.dump({"keys": keys, "gbm": gA, "ridge": rA, "grid": GRID,
                 "w_gbm": 0.7, "w_ridge": 0.3},
                open(H / OUT, "wb"))
    print(f"\n  persisted -> {OUT}  ({len(keys)} features, "
          f"{len(res)} training examples)")

if __name__ == "__main__":
    main()

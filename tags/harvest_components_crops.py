"""Component records WITH CROPS for a neural component verifier (§56).
Map = production-like: (1-w)*mean(9 train-only members) + w*Kevin train-only val TAP, w=0.2,
dilated-side-12 restriction, threshold THR (0.45 ~ mean predicted thr). Per component:
hand features (as harvest_components) + ddice label + 3x32^3 crop (CT window, prob, comp mask)
around the padded bbox. Env: HC_THR, HC_TAPW, HC_OUT, HC_WORKERS, HC_INDEX."""
import json, os, sys, collections, pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, nibabel as nib
from scipy import ndimage
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H))
from paths import TAPCT_VAL_MAPS  # noqa: E402
from train_t1_full import ts_path                                   # noqa: E402
from build_ensemble_submission import side_region                   # noqa: E402
from eval_t1_val200 import named_region                             # noqa: E402
from thr_predictor2 import PLUR, SEG, SEV, EXT, SMALL, CATS         # noqa: E402

THR = float(os.environ.get("HC_THR", "0.45")); DK = 12; LOBE_IDS = [10, 11, 12, 13, 14]
INDEX = os.environ.get("HC_INDEX", "index/t1_index_val.jsonl")
MEMBERS = open(H / "subsets_afternoon.txt").readline().strip().split(",")
TAP = TAPCT_VAL_MAPS
TAPW = float(os.environ.get("HC_TAPW", "0.2"))
OUT = os.environ.get("HC_OUT", "results/comp_crops_val.pkl")
WORKERS = int(os.environ.get("HC_WORKERS", "12")); CS = 32
TAPCH = bool(int(os.environ.get("HC_TAPCH", "0")))   # store the raw TAP prob crop as a 4th channel
STDCH = bool(int(os.environ.get("HC_STD", "0")))     # + member std and vote-count channels

def crop3(vol, slc, order):
    sub = vol[slc]
    zf = [CS / max(d, 1) for d in sub.shape]
    return ndimage.zoom(sub, zf, order=order)

def _one(args):
    case, rs = args
    tp = ts_path(case)
    if not Path(tp).exists() or not (TAP / f"{case}.nii.gz").exists(): return []
    paths = [H / f"CACHE_sf25_{m}" / f"{case}.nii.gz" for m in MEMBERS]
    if not all(p.exists() for p in paths): return []
    ts = np.asanyarray(nib.load(str(tp)).dataobj); segf = nib.load(rs[0]["seg"]).dataobj
    ct = np.asanyarray(nib.load(rs[0]["image"]).dataobj).astype(np.float32)
    ct = np.clip((ct + 1000.0) / 1400.0, 0, 1)                      # window [-1000, 400]
    if ct.shape != ts.shape: return []
    imgs = [nib.load(p).dataobj for p in paths]; tapd = nib.load(TAP / f"{case}.nii.gz").dataobj
    st2 = ndimage.generate_binary_structure(2, 1); out = []
    for r in rs:
        try:
            gt = np.asanyarray(segf[r["fidx"]] if len(segf.shape) == 4 else segf) > 0
            if gt.sum() < 20 or gt.shape != ts.shape: continue
            mem = [np.asanyarray(im[r["fidx"]]).astype(np.float32) / 65535.0 for im in imgs]
            q = sum(mem) / len(mem)
            if STDCH:
                sdev = np.std(np.stack(mem), axis=0); vote = np.mean(np.stack([m_ > 0.45 for m_ in mem]), axis=0).astype(np.float32)
            del mem
            t = np.asanyarray(tapd[r["fidx"]]).astype(np.float32) / 255.0
            if t.shape != q.shape: continue
            traw = t
            q = (1 - TAPW) * q + TAPW * t
            from restrict_rule import restrict_mask
            mm = restrict_mask(ts, r["text"], side_k=DK)
            if mm is not None:
                q = np.where(mm, q, 0.0); traw = np.where(mm, traw, 0.0)
                if STDCH: sdev = np.where(mm, sdev, 0.0); vote = np.where(mm, vote, 0.0)
            pred = q > THR; npred = int(pred.sum())
            if npred == 0: continue
            lab, n = ndimage.label(pred); gts = int(gt.sum())
            inter_full = int((pred & gt).sum()); dice_full = 2.0 * inter_full / (npred + gts)
            reg = named_region(r["text"]); regmask = np.isin(ts, reg) if reg else None; txt = r["text"]
            tf = dict(plural=int(bool(PLUR.search(txt))), seg_w=int(bool(SEG.search(txt))), lobe_w=int(reg is not None),
                      sev=int(bool(SEV.search(txt))), ext_w=int(bool(EXT.search(txt))), small_w=int(bool(SMALL.search(txt))), nwords=len(txt.split()))
            sizes = ndimage.sum_labels(np.ones_like(lab), lab, range(1, n + 1)).astype(int); order = np.argsort(-sizes)
            objs = ndimage.find_objects(lab); comps = []
            for rank, ci in enumerate(order):
                c = ci + 1; slc = objs[c - 1]; sub = lab[slc] == c; csz = int(sizes[ci])
                inter_c = int((sub & gt[slc]).sum()); d_wo = 2.0 * (inter_full - inter_c) / max(npred - csz + gts, 1)
                qq = q[slc][sub]; zs = slc[2]
                # padded bbox for the crop: 25% + 4 voxels each side
                pad = [(s.stop - s.start) // 4 + 4 for s in slc]
                cslc = tuple(slice(max(s.start - p, 0), min(s.stop + p, sh)) for s, p, sh in zip(slc, pad, q.shape))
                chans = [crop3(ct, cslc, 1), crop3(q, cslc, 1), crop3((lab == c).astype(np.float32), cslc, 0)] + ([crop3(traw, cslc, 1)] if TAPCH else []) + ([crop3(sdev, cslc, 1), crop3(vote, cslc, 1)] if STDCH else [])
                crop = np.stack(chans).astype(np.float16)
                comps.append(dict(size=csz, size_frac=csz / npred, rank=rank, nsib=n, peak=float(qq.max()), mean=float(qq.mean()),
                                  zext=int(zs.stop - zs.start), inlobe=float(np.isin(ts[slc][sub], LOBE_IDS).mean()),
                                  innamed=float(regmask[slc][sub].mean()) if regmask is not None else -1.0,
                                  bbox=[int(s.start) for s in slc] + [int(s.stop) for s in slc],
                                  inter=inter_c, ddice=dice_full - d_wo, crop=crop))
            out.append(dict(case=case, fidx=r["fidx"], cat=r["category"], text=txt, tf=tf, gts=gts, npred=npred, dice=dice_full, comps=comps))
        except Exception as e:
            print(f"[warn] {case}/{r.get('fidx')}: {e}", flush=True); continue
    return out

def main():
    rows = [json.loads(l) for l in open(H / INDEX)]; by = collections.defaultdict(list)
    for r in rows: by[r["case"]].append(r)
    print(f"[harvest-crops] {len(by)} cases, members={len(MEMBERS)} + TAP w={TAPW}, thr={THR}", flush=True)
    res = []
    with ProcessPoolExecutor(WORKERS) as ex:
        for i, f in enumerate(as_completed([ex.submit(_one, it) for it in by.items()]), 1):
            res.extend(f.result())
            if i % 25 == 0: print(f"  [{i}/{len(by)}] {len(res)} findings, {sum(len(r['comps']) for r in res)} comps", flush=True)
    pickle.dump(res, open(H / OUT, "wb"), protocol=4)
    ncomp = sum(len(r["comps"]) for r in res); nharm = sum(1 for r in res for c in r["comps"] if c["ddice"] < 0)
    print(f"[harvest-crops] -> {OUT}: {len(res)} findings, {ncomp} comps ({nharm} harmful, {100*nharm/max(ncomp,1):.1f}%), base dice {np.mean([r['dice'] for r in res]):.4f}", flush=True)
if __name__ == "__main__": main()

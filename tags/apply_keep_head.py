"""Apply the §56 4-channel component verifier (checkpoints/comp_head4.pt) to a production mask VARIANT dir.
Per test case: fused map = (1-w)*mean(12 members) + w*TAP (as build_ensemble_submission), restricted per finding
(dilated-side-12, same as the harvest); components of the PRODUCTION mask channel; per component the same crop
(CT window, fused prob, comp mask, raw TAP) + hand features as harvest_components_crops.py; drop if p(harmful) > TAU.
Env: KH_SRC, KH_DST, KH_TAU, KH_WORKERS, KH_START/KH_END. Members/TAP fixed to the 0.368 recipe."""
import os, sys, json, math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, nibabel as nib, torch
from scipy import ndimage
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H))
from paths import TAPCT_TEST_MAPS, CTRATE_ROOT  # noqa: E402
from train_t1_full import ts_path                                   # noqa: E402
from build_ensemble_submission import side_region, DS               # noqa: E402
from eval_t1_val200 import named_region                             # noqa: E402
from thr_predictor2 import PLUR, SEG, SEV, EXT, SMALL, CATS         # noqa: E402
import torch.nn as nn
class Head(nn.Module):                                              # identical to comp_head.Head (not imported: that module trains on import)
    def __init__(s, nf, cin=3):
        super().__init__()
        s.cnn = nn.Sequential(nn.Conv3d(cin, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2),
                              nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2),
                              nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(), nn.AdaptiveAvgPool3d(1), nn.Flatten())
        s.mlp = nn.Sequential(nn.Linear(64 + nf, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(s, x, f): return s.mlp(torch.cat([s.cnn(x), f], 1)).squeeze(1)

SRC = os.environ.get("KH_SRC", "submission/VARIANT_XGEN12_TAP_w020"); DST = os.environ.get("KH_DST", "submission/VARIANT_XGEN12_TAP_w020_HEAD4")
TAU = float(os.environ.get("KH_TAU", "0.7")); WORKERS = int(os.environ.get("KH_WORKERS", "6"))
START, END = int(os.environ.get("KH_START", "0")), int(os.environ.get("KH_END", "0"))
MEMBERS = os.environ.get("KH_MEMBERS", "cautious_aug33_sf125;aux2e_aug33_sf125;auxdil2_aug33_sf125;fusion24_aug33_sf125;lobeaux_aug33_sf125;fleet_plain_sf125;fleet_tvA_sf125;fleet_tvB_sf125;fleet_seed2_sf125;mb_2dboost_sf125;mb_catbal_sf125;mb_boost20_sf125").replace(",", ";").split(";")
assert len(MEMBERS) >= 12, MEMBERS
TAPD = Path(os.environ.get("KH_TAPD", str(TAPCT_TEST_MAPS))); TAPW = 0.2
CKPTS = [H / c for c in os.environ.get("KH_CKPTS", "checkpoints/comp_head4.pt").split(";")]; DK = 12; LOBE_IDS = [10, 11, 12, 13, 14]; CS = 32
CK = ["size", "size_frac", "rank", "nsib", "peak", "mean", "zext", "inlobe", "innamed"]; TK = ["plural", "seg_w", "lobe_w", "sev", "ext_w", "small_w", "nwords"]
CT_ROOT = CTRATE_ROOT

def ct_path(stem):
    sub = "valid_fixed" if stem.startswith("valid") else "train_fixed"; pid = "_".join(stem.split("_")[:2]); acc = "_".join(stem.split("_")[:3])
    return CT_ROOT / sub / pid / acc / f"{stem}.nii.gz"

def crop3(vol, slc, order):
    sub = vol[slc]; zf = [CS / max(d, 1) for d in sub.shape]; return ndimage.zoom(sub, zf, order=order)

_model = None
def model():
    global _model
    if _model is None:
        ms = []
        for cp in CKPTS:
            ck = torch.load(cp, map_location="cpu", weights_only=False); m = Head(ck["nf"], ck["cin"]); m.load_state_dict(ck["state"]); m.eval(); ms.append(m)
        _model = (ms, np.asarray(ck["fmean"], np.float32), np.asarray(ck["fstd"], np.float32))
    return _model

def _one(e):
    torch.set_num_threads(2)
    stem = e["name"][:-7] if e["name"].endswith(".nii.gz") else e["name"]
    m, fmean, fstd = model()
    mimg = nib.load(Path(SRC) / f"{stem}.nii.gz"); mask = np.asanyarray(mimg.dataobj).astype(np.uint8)
    keys = sorted(e["findings"], key=int); assert mask.shape[0] == len(keys)
    acc = None
    for mem in MEMBERS:
        q = np.asarray(nib.load(H / f"TESTPROB_{mem}" / f"{stem}.nii.gz").dataobj, np.float32) / 65535.0
        acc = q if acc is None else acc + q
    acc /= len(MEMBERS)
    tapq = np.asarray(nib.load(TAPD / f"{stem}.nii.gz").dataobj, np.float32) / 255.0
    fused = (1.0 - TAPW) * acc + TAPW * tapq
    ts = np.asanyarray(nib.load(str(ts_path(stem))).dataobj)
    ct = np.clip((np.asanyarray(nib.load(ct_path(stem)).dataobj).astype(np.float32) + 1000.0) / 1400.0, 0, 1)
    st2 = ndimage.generate_binary_structure(2, 1); n_sing = n_drop = vox_rm = 0; ncomp_tot = 0
    for j, k in enumerate(keys):
        txt = e["findings"][k]; cat = (e.get("categories") or {}).get(k, "")
        pm = mask[j] > 0; npred = int(pm.sum())
        if npred == 0: continue
        q = fused[j]; traw = tapq[j]; sid = side_region(txt)
        if sid is not None:
            mm = np.isin(ts, sid); mm = np.stack([ndimage.binary_dilation(mm[:, :, z], st2, iterations=DK) for z in range(mm.shape[2])], axis=2)
            q = np.where(mm, q, 0.0); traw = np.where(mm, traw, 0.0)
        lab, n = ndimage.label(pm); ncomp_tot += n
        if n <= 1: continue
        reg = named_region(txt); regmask = np.isin(ts, reg) if reg else None
        tf = [int(bool(PLUR.search(txt))), int(bool(SEG.search(txt))), int(reg is not None), int(bool(SEV.search(txt))), int(bool(EXT.search(txt))), int(bool(SMALL.search(txt))), len(txt.split())]
        sizes = ndimage.sum_labels(np.ones_like(lab), lab, range(1, n + 1)).astype(int); order = np.argsort(-sizes); objs = ndimage.find_objects(lab)
        X, F, cids = [], [], []
        for rank, ci in enumerate(order):
            c = ci + 1; slc = objs[c - 1]; sub = lab[slc] == c; csz = int(sizes[ci]); qq = q[slc][sub]; zs = slc[2]
            hand = dict(size=csz, size_frac=csz / npred, rank=rank, nsib=n, peak=float(qq.max()), mean=float(qq.mean()), zext=int(zs.stop - zs.start),
                        inlobe=float(np.isin(ts[slc][sub], LOBE_IDS).mean()), innamed=float(regmask[slc][sub].mean()) if regmask is not None else -1.0)
            pad = [(s.stop - s.start) // 4 + 4 for s in slc]
            cslc = tuple(slice(max(s.start - p, 0), min(s.stop + p, sh)) for s, p, sh in zip(slc, pad, q.shape))
            X.append(np.stack([crop3(ct, cslc, 1), crop3(q, cslc, 1), crop3((lab == c).astype(np.float32), cslc, 0), crop3(traw, cslc, 1)]).astype(np.float32))
            F.append([math.log1p(hand[kk]) if kk in ("size", "zext") else hand[kk] for kk in CK] + tf + [int(cat == cc) for cc in CATS]); cids.append(c)
        Fn = (np.array(F, np.float32) - fmean) / fstd
        with torch.no_grad(): p = np.mean([torch.sigmoid(mm_(torch.from_numpy(np.stack(X)), torch.from_numpy(Fn))).numpy() for mm_ in m], axis=0)
        drop = [c for c, pp in zip(cids, p) if pp > TAU]
        if drop and len(drop) < n:                     # never empty a finding
            rm = np.isin(lab, drop); vox_rm += int(rm.sum()); mask[j][rm] = 0; n_drop += len(drop)
    h = mimg.header.copy(); h.set_data_dtype(np.uint8); h["scl_slope"], h["scl_inter"] = 1.0, 0.0
    nib.save(nib.Nifti1Image(mask, mimg.affine, h), Path(DST) / f"{stem}.nii.gz")
    return dict(comps=ncomp_tot, drop=n_drop, vox_rm=vox_rm)

def main():
    Path(DST).mkdir(parents=True, exist_ok=True)
    d = json.load(open(DS)); items = list(d["test"].values()) if isinstance(d["test"], dict) else d["test"]
    if START or END: items = items[START:(END or len(items))]
    print(f"[keep-head] {len(items)} cases, tau={TAU}, {len(MEMBERS)} members, src={SRC} -> {DST}", flush=True); agg = dict(comps=0, drop=0, vox_rm=0)
    with ProcessPoolExecutor(WORKERS) as ex:
        for i, f in enumerate(as_completed([ex.submit(_one, e) for e in items]), 1):
            r = f.result()
            for k in agg: agg[k] += r[k]
            if i % 25 == 0: print(f"  [{i}/{len(items)}] {agg}", flush=True)
    print(f"[keep-head] done: components {agg['comps']}, dropped {agg['drop']} ({100*agg['drop']/max(agg['comps'],1):.1f}%), voxels removed {agg['vox_rm']}", flush=True)

if __name__ == "__main__": main()

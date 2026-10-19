"""§81 case-level filters on a VARIANT mask dir (test). Fused map per finding = 0.8*mean(12 members) + 0.2*TAP, restricted
(side-12 / lobe-16) exactly as production. Modes (AC_MODE, ';'-separated, applied in order):
  vox:M   voxels claimed by >=2 findings go to the finding with the higher fused prob (keep if q >= best - M); never empties
  comp    whole component of finding i dropped if >50% of it is claimed by another finding with higher mean prob; never empties
  bilat:T text bilateral/both and one-sided mask -> add largest other-side blob with q > T
Env: AC_SRC, AC_DST, AC_MODE, AC_WORKERS, AC_MEMBERS (';'), AC_TAPD."""
import os, sys, json, re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, nibabel as nib
from scipy import ndimage
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H))
from paths import TAPCT_TEST_MAPS  # noqa: E402
from train_t1_full import ts_path; from build_ensemble_submission import DS; from restrict_rule import restrict_mask; from eval_t1_val200 import LEFT, RIGHT
SRC = os.environ["AC_SRC"]; DST = os.environ["AC_DST"]; MODES = os.environ.get("AC_MODE", "comp").split(";")
MEMBERS = os.environ.get("AC_MEMBERS", "cautious_aug33_sf125;aux2e_aug33_sf125;auxdil2_aug33_sf125;fusion24_aug33_sf125;lobeaux_aug33_sf125;fleet_plain_sf125;fleet_tvA_sf125;fleet_tvB_sf125;fleet_seed2_sf125;mb_2dboost_sf125;mb_catbal_sf125;mb_boost20_sf125").split(";")
assert len(MEMBERS) >= 12, MEMBERS
TAPD = Path(os.environ.get("AC_TAPD", str(TAPCT_TEST_MAPS)))
BILAT = re.compile(r"\b(bilateral|both lungs|both lower|both upper|both sides|in both|bilaterally)\b", re.I)
def _one(e):
    stem = e["name"][:-7] if e["name"].endswith(".nii.gz") else e["name"]
    mimg = nib.load(Path(SRC) / f"{stem}.nii.gz"); mask = np.asanyarray(mimg.dataobj).astype(np.uint8); keys = sorted(e["findings"], key=int); assert mask.shape[0] == len(keys)
    acc = None
    for mem in MEMBERS:
        q = np.asarray(nib.load(H / f"TESTPROB_{mem}" / f"{stem}.nii.gz").dataobj, np.float32) / 65535.0; acc = q if acc is None else acc + q
    fused = 0.8 * (acc / len(MEMBERS)) + 0.2 * np.asarray(nib.load(TAPD / f"{stem}.nii.gz").dataobj, np.float32) / 255.0
    ts = np.asanyarray(nib.load(str(ts_path(stem))).dataobj); n = len(keys); texts = [e["findings"][k] for k in keys]
    Q = []
    for j in range(n):
        mm = restrict_mask(ts, texts[j], side_k=12, lobe_k=16) if ts.shape == fused.shape[1:] else None
        Q.append(np.where(mm, fused[j], 0.0) if mm is not None else fused[j])
    P = [mask[j] > 0 for j in range(n)]; stats = dict(vox_moved=0, comp_drop=0, bilat_add=0)
    for mode in MODES:
        name, _, arg = mode.partition(":")
        if name in ("vox", "comp") and n >= 2:
            S = np.zeros(P[0].shape, np.int16)
            for p in P: S += p
            multi = S >= 2
            if not multi.any(): continue
            best = np.max(np.stack([np.where(P[i], Q[i], 0.0) for i in range(n)]), 0)
            for i in range(n):
                if not (P[i] & multi).any(): continue
                if name == "vox":
                    keep = P[i] & (~multi | (Q[i] >= best - float(arg or 0)))
                else:
                    lab, nc = ndimage.label(P[i]); keep = P[i].copy()
                    for c in range(1, nc + 1):
                        cm = lab == c
                        if (cm & multi).sum() / cm.sum() > 0.5:
                            others = [j for j in range(n) if j != i and (P[j] & cm).any()]
                            if others and max(float(Q[j][cm].mean()) for j in others) > float(Q[i][cm].mean()): keep &= ~cm; stats["comp_drop"] += 1
                if keep.any():
                    stats["vox_moved"] += int((P[i] & ~keep).sum()); P[i] = keep
        elif name == "bilat":
            left = np.isin(ts, LEFT); right = np.isin(ts, RIGHT)
            for i in range(n):
                p = P[i]
                if not (BILAT.search(texts[i]) and p.any()): continue
                onL, onR = (p & left).any(), (p & right).any()
                if onL == onR: continue
                cand = (Q[i] > float(arg)) & (right if onL else left) & ~p
                if cand.any():
                    lab, nc = ndimage.label(cand); sizes = ndimage.sum_labels(np.ones_like(lab), lab, range(1, nc + 1)); P[i] = p | (lab == int(np.argmax(sizes)) + 1); stats["bilat_add"] += 1
    for j in range(n): mask[j] = P[j].astype(np.uint8)
    h = mimg.header.copy(); h.set_data_dtype(np.uint8); h["scl_slope"], h["scl_inter"] = 1.0, 0.0
    nib.save(nib.Nifti1Image(mask, mimg.affine, h), Path(DST) / f"{stem}.nii.gz"); return stats
def main():
    Path(DST).mkdir(parents=True, exist_ok=True); d = json.load(open(DS)); items = list(d["test"].values()) if isinstance(d["test"], dict) else d["test"]
    print(f"[case] {len(items)} cases modes={MODES} {SRC} -> {DST}", flush=True); agg = dict(vox_moved=0, comp_drop=0, bilat_add=0)
    with ProcessPoolExecutor(int(os.environ.get("AC_WORKERS", "8"))) as ex:
        for i, f in enumerate(as_completed([ex.submit(_one, e) for e in items]), 1):
            for k, v in f.result().items(): agg[k] += v
            if i % 25 == 0: print(f"  [{i}/{len(items)}] {agg}", flush=True)
    print(f"[case] done: {agg}", flush=True)
if __name__ == "__main__": main()

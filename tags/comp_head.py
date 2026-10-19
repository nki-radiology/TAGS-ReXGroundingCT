"""§56 component VERIFIER head: small 3D CNN on (CT, prob, comp-mask) 32^3 crops + MLP on hand
features/text flags/category -> p(harmful) where harmful := dropping the component raises Dice.
5-fold OOF by case on results/comp_crops_val.pkl; drop threshold tau tuned on the TRAIN folds only;
exact Dice after dropping via closed form (inter/size stored per component). Prints OOF gain vs base.
Env: CH_PKL, CH_EPOCHS, CH_DEVICE (cpu|cuda), CH_OUT (state dict of a full-data model, optional)."""
import os, sys, pickle, math
from pathlib import Path
import numpy as np, torch, torch.nn as nn
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H))
from thr_predictor2 import CATS  # noqa: E402
PKL = os.environ.get("CH_PKL", "results/comp_crops_val.pkl"); EPOCHS = int(os.environ.get("CH_EPOCHS", "12")); NSEED = int(os.environ.get("CH_NSEED", "1")); INNER = bool(int(os.environ.get("CH_INNER", "0"))); SHUF = bool(int(os.environ.get("CH_SHUFFLE", "0")))
DEV = os.environ.get("CH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"); OUTP = os.environ.get("CH_OUT", "")
SEED = int(os.environ.get("CH_SEED", "0")); torch.manual_seed(SEED); np.random.seed(SEED)
CK = ["size", "size_frac", "rank", "nsib", "peak", "mean", "zext", "inlobe", "innamed"]; TK = ["plural", "seg_w", "lobe_w", "sev", "ext_w", "small_w", "nwords"]

recs = pickle.load(open(H / PKL, "rb"))
REL = bool(int(os.environ.get("CH_REL", "0"))); REG = bool(int(os.environ.get("CH_REG", "0")))
X, F, y, meta = [], [], [], []
yg = []
for fi, r in enumerate(recs):
    big = max(r["comps"], key=lambda c: c["size"]) if r["comps"] else None
    for ci, c in enumerate(r["comps"]):
        X.append(c["crop"]); f = [math.log1p(c[k]) if k in ("size", "zext") else c[k] for k in CK] + [r["tf"][k] for k in TK] + [int(r["cat"] == cc) for cc in CATS]
        if REL and big is not None and "bbox" in c:
            cb = [(c["bbox"][i] + c["bbox"][i + 3]) / 2 for i in range(3)]; bb = [(big["bbox"][i] + big["bbox"][i + 3]) / 2 for i in range(3)]
            f += [c["peak"] / max(big["peak"], 1e-3), c["mean"] / max(big["mean"], 1e-3), math.log1p(c["size"]) - math.log1p(big["size"]), math.log1p(math.dist(cb, bb)), int(c is big)]
        yg.append(max(0.0, -c["ddice"]))
        F.append(f); y.append(int(c["ddice"] < 0)); meta.append((fi, ci))
X = np.stack(X).astype(np.float32); F = np.array(F, np.float32); y = np.array(y, np.float32); yg = np.array(yg, np.float32)
if SHUF:
    rs = np.random.RandomState(1); y = y[rs.permutation(len(y))]; print("[control] labels SHUFFLED", flush=True)
FM, FS = F.mean(0), F.std(0) + 1e-6
F = (F - FM) / FS
cases = np.array([recs[fi]["case"] for fi, _ in meta]); uc = sorted(set(cases)); fold = {c: i % 5 for i, c in enumerate(uc)}
fo = np.array([fold[c] for c in cases]); base = np.array([r["dice"] for r in recs])
print(f"[data] {len(recs)} findings, {len(y)} comps, harmful {y.mean():.3f}, base dice {base.mean():.4f}, device {DEV}", flush=True)

class Head(nn.Module):
    def __init__(s, nf, cin=3):
        super().__init__()
        s.cnn = nn.Sequential(nn.Conv3d(cin, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2),
                              nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2),
                              nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(), nn.AdaptiveAvgPool3d(1), nn.Flatten())
        s.mlp = nn.Sequential(nn.Linear(64 + nf, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(s, x, f): return s.mlp(torch.cat([s.cnn(x), f], 1)).squeeze(1)

def fit(idx):
    m = Head(F.shape[1], X.shape[1]).to(DEV); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    pw = torch.tensor([(1 - y[idx].mean()) / max(y[idx].mean(), 1e-3)], device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw); bs = 64
    if REG:
        lossf = lambda out, yb: nn.functional.smooth_l1_loss(out, yb)
        yt_ = torch.from_numpy(yg * 100.0)   # gain in Dice*100 for scale
    else: yt_ = torch.from_numpy(y)
    for ep in range(EPOCHS):
        m.train(); perm = np.random.permutation(idx); tot = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]; xb = torch.from_numpy(X[b]).to(DEV); fb = torch.from_numpy(F[b]).to(DEV); yb = yt_[b].to(DEV)
            if np.random.rand() < 0.5: xb = xb.flip(2)          # z-flip aug only (L/R matters)
            opt.zero_grad(); l = lossf(m(xb, fb), yb); l.backward(); opt.step(); tot += float(l) * len(b)
        print(f"    ep {ep+1}/{EPOCHS} loss {tot/len(perm):.4f}", flush=True)
    return m

@torch.no_grad()
def predict(m, idx):
    m.eval(); out = np.zeros(len(idx), np.float32)
    for i in range(0, len(idx), 256):
        b = idx[i:i + 256]; o_ = m(torch.from_numpy(X[b]).to(DEV), torch.from_numpy(F[b]).to(DEV)); out[i:i + len(b)] = (o_ if REG else torch.sigmoid(o_)).cpu().numpy()
    return out

def dice_after(r, drop):
    inter = sum(c["inter"] for c in r["comps"]) - sum(r["comps"][i]["inter"] for i in drop)
    den = r["npred"] - sum(r["comps"][i]["size"] for i in drop) + r["gts"]
    return 2.0 * inter / den if den > 0 else 0.0

def apply(prob, idx, tau):
    by = {}
    for p, j in zip(prob, idx): by.setdefault(meta[j][0], []).append((meta[j][1], p))
    d = {}
    for fi, lst in by.items(): d[fi] = dice_after(recs[fi], [ci for ci, p in lst if p > tau])
    return d

oof = base.copy(); TAUS = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0) if REG else (0.5, 0.6, 0.7, 0.8, 0.9, 0.95); OOFP = np.full(len(y), np.nan, np.float32); DUMP = os.environ.get("CH_DUMP", "")
SKIP = bool(int(os.environ.get("CH_SKIP_OOF", "0")))
for k in (range(5) if not SKIP else []):
    tr = np.where(fo != k)[0]; te = np.where(fo == k)[0]; print(f"[fold {k}] train {len(tr)} test {len(te)}", flush=True)
    def ens_fit(idx):
        ms = []
        for s_ in range(NSEED):
            torch.manual_seed(s_); np.random.seed(s_); ms.append(fit(idx))
        return lambda ids: np.mean([predict(m_, ids) for m_ in ms], axis=0)
    if INNER:                                   # tau from an inner held-out 20% of train CASES
        ic = np.array([hash(cases[j]) % 5 == 0 for j in tr]); pin = ens_fit(tr[~ic]); ptr = pin(tr[ic]); tune_ids = tr[ic]
    else:
        tune_ids = tr
    pred_f = ens_fit(tr); ptr = ptr if INNER else pred_f(tr)
    best_tau, best_gain = None, 0.0
    for tau in TAUS:
        d = apply(ptr, tune_ids, tau); gain = sum(d[fi] - recs[fi]["dice"] for fi in d)
        if gain > best_gain: best_gain, best_tau = gain, tau
    print(f"  {'inner-holdout' if INNER else 'train-fold'} best tau {best_tau} (gain {best_gain:+.3f} summed)", flush=True)
    if best_tau is None: continue
    pte_ = pred_f(te); OOFP[te] = pte_
    d = apply(pte_, te, best_tau)
    for fi, v in d.items(): oof[fi] = v
    print(f"  fold {k}: OOF {np.mean([oof[fi] for fi in d]):.4f} vs base {np.mean([base[fi] for fi in d]):.4f}", flush=True)
print(f"\n=== §56 verifier head OOF: {oof.mean():.4f} vs base {base.mean():.4f}  ({oof.mean()-base.mean():+.4f}) ===", flush=True)
for c in sorted({r["cat"] for r in recs}):
    ii = [i for i, r in enumerate(recs) if r["cat"] == c]; print(f"  {c} n={len(ii):3d} {base[ii].mean():.4f} -> {oof[ii].mean():.4f} ({oof[ii].mean()-base[ii].mean():+.4f})")
if OUTP:
    m = fit(np.arange(len(y))); torch.save({"state": m.state_dict(), "nf": F.shape[1], "cin": X.shape[1], "fmean": FM, "fstd": FS}, H / OUTP); print(f"[save] {OUTP}")

# ---- tau per plurality group (singular vs plural text), tuned per fold on the other folds' OOF probs ----
if not SKIP and not np.isnan(OOFP).all():
    plural = np.array([recs[fi]["tf"]["plural"] for fi, _ in meta]).astype(bool)
    oof2 = base.copy()
    for k in range(5):
        te = np.where(fo == k)[0]; tr = np.where(fo != k)[0]
        taus = {}
        for g, name in ((False, "sing"), (True, "plur")):
            ids = tr[plural[tr] == g]; best, bt = 0.0, 0.7
            for tau in TAUS:
                d = apply(OOFP[ids], ids, tau); gain = sum(d[fi] - recs[fi]["dice"] for fi in d)
                if gain > best: best, bt = gain, tau
            taus[g] = bt
        # apply group-specific taus on the held-out fold
        by = {}
        for j in te: by.setdefault(meta[j][0], []).append((meta[j][1], OOFP[j], plural[j]))
        for fi, lst in by.items(): oof2[fi] = dice_after(recs[fi], [ci for ci, p, g in lst if p > taus[g]])
        print(f"  [plural-tau fold {k}] sing {taus[False]} plur {taus[True]}", flush=True)
    print(f"=== tau by plurality: OOF {oof2.mean():.4f} vs single-tau OOF {oof.mean():.4f} vs base {base.mean():.4f} ===", flush=True)
    if DUMP: np.savez(H / DUMP, oofp=OOFP, y=y, fo=fo, plural=plural)

"""Evaluate a Track-1 checkpoint on val-200: per-finding Dice, optionally region-restricted.

Training uses 192^3 patches but a finding must be segmented in the whole 512x512xD volume, so
inference is sliding-window. All findings of a case are pushed through as N prompts in ONE
forward per patch -- VoxTell shares the image encode across prompts, so evaluating 3 findings
costs barely more than 1.

`--restrict` applies the post-processing the rules explicitly permit: intersect the predicted
mask with the TotalSegmentator region named in the finding. The lobe parser here is a plain
string match, which is enough because ReXGroundingCT phrasing is standardised ("right lower
lobe"); it can be swapped for the LLM mapper's ts_labels without touching anything else.

    python eval_t1_val200.py --ckpt VoxTellTrainer_T1_full.pt [--restrict] [--limit 20]
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_t1_full import (BANK_DIR, CAT2IDX, CODES14, LOBES, TEXT_DIM, T1Model,  # noqa: E402
                           build_model, ts_path, zscore)
from build_val_cache import load_normalized_ct  # noqa: E402  (cached decompress+z-score)

# TotalSegmentator lung-lobe label ids
LOBE_ID = {"lung_upper_lobe_left": 10, "lung_lower_lobe_left": 11,
           "lung_upper_lobe_right": 12, "lung_middle_lobe_right": 13,
           "lung_lower_lobe_right": 14}
RIGHT = [12, 13, 14]
LEFT = [10, 11]


TS_LABELS_JSONL = HERE / "rexg_ts" / "rexg_ts_labels.jsonl"
_LLM_REGION = None


def load_llm_regions():
    """text -> TotalSeg LOBE ids, from the LLM mapper (map_categories_llm.py).

    Replaces the regex below, which could not parse "bilateral lower lobes" or "both lungs"
    (44% of val findings unrestrictable) and bled across sides when a stray "left"/"right"
    appeared anywhere in the string.

    Only LOBE labels are kept. A finding whose ts_labels contain no lobe -- pleural effusion
    (`pleural_effusion`), pneumothorax (`[]`) -- yields no region and is therefore NOT
    restricted. That is deliberate: those sit outside the lung parenchyma, and intersecting
    them with lobe masks deletes true signal (measured: 2e Dice 0.455 -> 0.384 under the regex).
    """
    global _LLM_REGION
    if _LLM_REGION is not None:
        return _LLM_REGION
    _LLM_REGION = {}
    if not TS_LABELS_JSONL.exists():
        return _LLM_REGION
    src = {}
    findings_file = HERE / "rexg_ts" / "rexg_findings.jsonl"
    if findings_file.exists():
        for line in open(findings_file):
            r = json.loads(line)
            src[r["key"]] = r["text"]
    for line in open(TS_LABELS_JSONL):
        r = json.loads(line)
        text = src.get(r["key"])
        if text is None:
            continue
        ids = sorted({LOBE_ID[n] for f in r["findings"] for n in f["ts_labels"] if n in LOBE_ID})
        _LLM_REGION[text] = ids or None
    return _LLM_REGION


def named_region(text: str):
    """TotalSeg lobe ids named in the finding, or None if it names no lung location."""
    llm = load_llm_regions()
    if text in llm:
        return llm[text]
    t = text.lower()                                   # fallback: regex (unseen phrasing only)
    ids = []
    for side, lobes in (("right", {"upper": 12, "middle": 13, "lower": 14}),
                        ("left", {"upper": 10, "lower": 11})):
        for lobe, lid in lobes.items():
            if re.search(rf"{lobe}\s+lobe", t) and side in t:
                ids.append(lid)
    if "lingula" in t:                             # lingula is part of the left upper lobe
        ids.append(10)
    ids = sorted(set(ids))
    if not ids:                                    # side named but no lobe
        has_r, has_l = "right" in t, "left" in t
        if has_r and not has_l:
            ids = RIGHT
        elif has_l and not has_r:
            ids = LEFT
    return ids or None


def dice(a, b, eps=1e-6):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float((2 * (a & b).sum() + eps) / (s + eps))


def _gaussian_window(patch, sigma_scale=0.125, device="cpu"):
    """Gaussian importance map over a patch, peak 1 at the centre.

    A voxel at the EDGE of a 192^3 window has almost no surrounding context, so its prediction
    is systematically worse than one at the centre. Uniform averaging (the previous behaviour)
    weights them equally; nnU-Net weights each window by this Gaussian so confident central
    predictions dominate the blend. sigma = 1/8 of the patch side is nnU-Net's default.
    """
    import numpy as _np
    axes = []
    for p in patch:
        c = (p - 1) / 2.0
        x = _np.arange(p, dtype=_np.float32)
        axes.append(_np.exp(-0.5 * ((x - c) / (p * sigma_scale)) ** 2))
    g = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    g = g / g.max()
    return _np.maximum(g, 1e-3).astype(_np.float32)      # floor: never zero-weight a voxel


@torch.no_grad()
def predict_case(model, img, embs, cats, patch, dev, step_frac=0.5, thr=0.5, batch=4,
                 use_cat=1.0, return_prob=False, gaussian=False, anat=None,
                 collect_region=False):
    """Sliding-window logits -> probability maps, one per prompt. img: (6,H,W,D) numpy.

    Patches are pushed through in BATCHES rather than one at a time. A 192^3 forward with
    batch=1 badly under-feeds the GPU at inference (no activations are stored without grad),
    and a case needs ~98 windows -- which is why a 40-case eval took ~2 h. Same patches, same
    model, same arithmetic; only the number of forward calls changes.
    """
    C, H, W, D = img.shape
    ph, pw, pd = patch
    N = embs.shape[0]
    acc = np.zeros((N, H, W, D), np.float32)
    racc = np.zeros((N, H, W, D), np.float32) if collect_region else None
    cnt = np.zeros((H, W, D), np.float32)
    steps = [max(1, int(p * step_frac)) for p in patch]

    def starts(total, p, s):
        if total <= p:
            return [0]
        xs = list(range(0, total - p + 1, s))
        if xs[-1] != total - p:
            xs.append(total - p)
        return xs

    txt1 = torch.from_numpy(embs).float()[None, :, None].to(dev)      # (1,N,1,D)
    cat1 = torch.tensor(cats, dtype=torch.long)[None].to(dev)         # (1,N)
    origins = [(y, x, z) for y in starts(H, ph, steps[0])
               for x in starts(W, pw, steps[1]) for z in starts(D, pd, steps[2])]

    gw = _gaussian_window(patch) if gaussian else None

    def flush(buf):
        """Run one forward for a list of (origin, patch-array) and scatter the results back."""
        if not buf:
            return
        t = torch.from_numpy(np.stack([b[1] for b in buf])).float().to(dev)   # (B,6,ph,pw,pd)
        B = t.shape[0]
        if anat is not None:
            # A model trained with --anat-pos expects PER-WINDOW anatomy-relative coordinates.
            # Without this the wrapper falls back to VoxTell's fixed grid -- a signal the model
            # has never seen -- and it scores badly for reasons unrelated to the idea.
            from anat_pos import token_pos
            fields, fstride, grid, dim = anat
            pos = np.stack([token_pos(fields, fstride, o, ph, grid, dim) for o, _ in buf])
            model.set_pos(torch.from_numpy(pos).float().to(dev))
        with torch.amp.autocast("cuda", enabled=dev.startswith("cuda")):
            out = model(t, txt1.expand(B, -1, -1, -1), cat1.expand(B, -1),
                        torch.full((B, N), float(use_cat), device=dev))
        if not torch.is_tensor(out):
            out = sorted(out, key=lambda o: -o.shape[-1])[0]          # finest scale
        probs = torch.sigmoid(out.float()).cpu().numpy()              # (B,N,ph,pw,pd)
        rp = None
        if collect_region:
            ro = getattr(model, "region_out", None)
            rp = torch.sigmoid(ro.float()).cpu().numpy() if ro is not None else None
        for k, ((y, x, z), _) in enumerate(buf):
            hh, ww, dd = min(ph, H - y), min(pw, W - x), min(pd, D - z)
            if gw is None:
                acc[:, y:y + hh, x:x + ww, z:z + dd] += probs[k][:, :hh, :ww, :dd]
                if rp is not None:
                    racc[:, y:y + hh, x:x + ww, z:z + dd] += rp[k][:, :hh, :ww, :dd]
                cnt[y:y + hh, x:x + ww, z:z + dd] += 1
            else:
                w = gw[:hh, :ww, :dd]
                acc[:, y:y + hh, x:x + ww, z:z + dd] += probs[k][:, :hh, :ww, :dd] * w
                cnt[y:y + hh, x:x + ww, z:z + dd] += w      # weighted mean: divide by same w

    buf = []
    for (y, x, z) in origins:
        sub = img[:, y:y + ph, x:x + pw, z:z + pd]
        if sub.shape[1:] != (ph, pw, pd):                             # pad edge patches
            pad = [(0, 0)] + [(0, p - s) for p, s in zip(patch, sub.shape[1:])]
            sub = np.pad(sub, pad)
        buf.append(((y, x, z), sub))
        if len(buf) == batch:
            flush(buf); buf = []
    flush(buf)
    prob = acc / np.maximum(cnt, 1e-6)
    if collect_region:
        return prob, (racc / np.maximum(cnt, 1e-6))
    # return_prob lets a caller sweep MANY thresholds from ONE sliding-window pass; the window
    # pass is ~25 s/case, so re-running the eval per threshold would cost hours for nothing.
    return prob if return_prob else (prob >= thr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "VoxTellTrainer_T1_full.pt"))
    ap.add_argument("--restrict", action="store_true", help="intersect with the named lobe(s)")
    ap.add_argument("--bank", default="",
                    help="text bank dir (texts.json + embeddings.npy). Empty = the finding bank. "
                         "The class-agnostic lesion detector uses its own single-prompt bank.")
    ap.add_argument("--index", default="t1_index_val.jsonl",
                    help="val index to score. Use t1_val50_hold.jsonl for the 50 cases held out "
                         "of the val-dose experiments -- the only clean measurement for a model "
                         "that trained on val.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=4, help="patches per forward")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [json.loads(l) for l in open(HERE / "index" / a.index)]
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case"]].append(r)
    cases = sorted(by_case)
    if a.limit:
        cases = cases[: a.limit]

    _bank_dir = Path(a.bank) if a.bank else BANK_DIR
    bank = json.load(open(_bank_dir / "texts.json"))
    emb = np.load(_bank_dir / "embeddings.npy")
    text2vec = {t: emb[i] for i, t in enumerate(bank)}

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    w = ck["network_weights"].get("encoder.stem.convs.0.conv.weight")
    n_in = int(w.shape[1]) if w is not None else 1 + len(LOBES)   # 1-ch (--no-anat) vs 6-ch models
    net, patch = build_model(dev, in_channels=n_in, deep_supervision=True)
    # rebuild whatever conditioning pathway the checkpoint was trained with, else an
    # encoder trained on modulated skips is evaluated without them (see EXPERIMENTS.md 4o-2)
    _fs = tuple(ck.get("film_stages", ()) or ())
    _gs = tuple(ck.get("gate_stages", ()) or ())
    model = T1Model(net, len(CODES14), TEXT_DIM, film_stages=_fs, gate_stages=_gs).to(dev)
    # FOURTH loader to need this (predict_test.py:64 documents the first three): a checkpoint
    # trained with --fusion-stage carries a bigger pos_embed and a reshaped bottleneck projection;
    # align_to_checkpoint retargets before load and is a no-op for standard checkpoints.
    from fusion_res import align_to_checkpoint
    if align_to_checkpoint(model.net, ck["network_weights"], patch=int(patch[0])):
        model.to(dev)
    if _fs or _gs:
        _missing = model.load_state_dict(ck.get("cond_extra", {}), strict=False)
        print(f"[cond] film={_fs} gate={_gs}; loaded {len(ck.get('cond_extra', {}))} extra tensors",
              flush=True)
    model.net.load_state_dict(ck["network_weights"])
    model.cat_embed.load_state_dict(ck["cat_embed"])
    model.eval()
    print(f"[ckpt] {a.ckpt} (epoch {ck.get('epoch')})  -- reporting raw AND region-restricted", flush=True)

    # Both scorings come from ONE inference pass: the sliding window is the expensive part,
    # while the region restriction is a boolean AND on the same prediction. (Running the script
    # twice -- once with --restrict -- doubled the cost and previously blew the time limit.)
    S = {"raw": {"all": [], "amb": [], "uniq": [], "cat": defaultdict(list)},
         "restricted": {"all": [], "amb": [], "uniq": [], "cat": defaultdict(list)}}
    n_restrictable = 0
    for i, case in enumerate(cases, 1):
        rs = sorted(by_case[case], key=lambda r: r["fidx"])
        vol = load_normalized_ct(case, rs[0]["image"])   # cache hit -> ~1s instead of ~27s
        p = ts_path(case)
        ts = np.asanyarray(nib.load(str(p)).dataobj) if p.exists() else np.zeros_like(vol, np.uint8)
        if n_in == 1:
            img = vol[None].astype(np.float32)                    # 1-channel (--no-anat) model
        else:
            anat = np.stack([(ts == l).astype(np.float32) for l in LOBES])
            img = np.concatenate([vol[None], anat], 0)

        embs = np.stack([text2vec[r["text"]] for r in rs])
        cats = [CAT2IDX[r["category"]] for r in rs]
        pred = predict_case(model, img, embs, cats, patch, dev, thr=a.thr, batch=a.batch)

        segobj = nib.load(rs[0]["seg"]).dataobj
        catcount = defaultdict(int)
        for r in rs:
            catcount[r["category"]] += 1
        for j, r in enumerate(rs):
            gt = (np.asanyarray(segobj[r["fidx"]] if len(segobj.shape) == 4 else segobj) > 0)
            raw = pred[j]
            reg = named_region(r["text"])
            restricted = (raw & np.isin(ts, reg)) if reg else raw
            n_restrictable += 1 if reg else 0
            amb = catcount[r["category"]] > 1
            for key, pm in (("raw", raw), ("restricted", restricted)):
                d = dice(pm, gt)
                S[key]["all"].append(d)
                S[key]["cat"][r["category"]].append(d)
                S[key]["amb" if amb else "uniq"].append(d)
        if i % 10 == 0:
            print(f"  [{i}/{len(cases)}] raw={np.mean(S['raw']['all']):.4f} "
                  f"restricted={np.mean(S['restricted']['all']):.4f}", flush=True)

    n = len(S["raw"]["all"])
    print(f"\n=== val-200: {len(cases)} cases, {n} findings "
          f"({n_restrictable} name a lung region the mask can be restricted to) ===")
    print(f"  {'':30}{'raw':>10}{'restricted':>13}{'delta':>10}")
    for label, key in (("mean Dice", "all"),
                       ("category-AMBIGUOUS (text needed)", "amb"),
                       ("category-unique", "uniq")):
        r_, x_ = np.mean(S["raw"][key]), np.mean(S["restricted"][key])
        print(f"  {label:30}{r_:>10.4f}{x_:>13.4f}{x_-r_:>+10.4f}   (n={len(S['raw'][key])})")
    print("  per category (raw -> restricted):")
    for c in CODES14:
        if S["raw"]["cat"][c]:
            print(f"    {c}: {np.mean(S['raw']['cat'][c]):.4f} -> {np.mean(S['restricted']['cat'][c]):.4f}"
                  f"  (n={len(S['raw']['cat'][c])})")


if __name__ == "__main__":
    main()

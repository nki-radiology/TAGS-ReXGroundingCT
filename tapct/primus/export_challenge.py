"""Export ReXGroundingCT-challenge predictions: one 4D uint8 NIfTI (F, X, Y, Z) per scan on the ORIGINAL CT grid.

For every scan of the requested split, every finding text is run through one or more referring2 runs (ensemble =
mean probability; the encoder is shared across findings within a run because it is text-independent), the
probability map is carried back from the 1.5 mm cache grid to the CT grid by inverting build_cache's mapping
exactly (trilinear resize to the original (z,y,x) shape, then the same axis flips and the (z,y,x)->(x,y,z)
transpose), thresholded and stacked in finding-index order — the layout of the released segmentation masks and
of the challenge submission.

    python primus/export_challenge.py --runs results/primus/rex_combo_z24_e16_50k --head v3both_cat_4_6_9_12 \
        --root <cache dir> --split val --out <dir> --win-z 24 [--save-prob]

`--win-z` must equal the window depth the run was trained with (PRIMUS_WIN_Z); it is checked against the
environment so a run is never scored at a geometry it never saw.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from primus.data import WIN, SPACING, gaussian_weight, load_meta, load_volume, normalise, sliding_windows   # noqa: E402
from primus.eval_volume import load_model                                                       # noqa: E402
from primus.referring import TokenTextBank, log                                                 # noqa: E402
from primus.referring2 import RefNet                                                            # noqa: E402
from primus.build_cache import rex_ct_path, REX_JSON, nib_to_lps_zyx                            # noqa: E402


class MergedBank:
    """Several token banks presented as one (train/val bank + a test bank)."""

    def __init__(self, paths):
        self.banks = [TokenTextBank(p) for p in paths]

    def get(self, stem, fid):
        for b in self.banks:
            r = b.get(stem, fid)
            if r is not None:
                return r
        return None


def lps_zyx_to_ct_grid(arr, affine):
    """Inverse of build_cache.nib_to_lps_zyx: (z,y,x) LPS array -> the CT file's own (x,y,z) voxel order."""
    import nibabel as nib
    codes = nib.aff2axcodes(affine)
    flips = [ax for ax, (c, w) in enumerate(zip(codes, ('L', 'P', 'S'))) if c != w]
    out = np.transpose(arr, (2, 1, 0))
    if flips:
        out = np.flip(out, axis=flips)
    return np.ascontiguousarray(out)


AUTOCAST = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': None}


@torch.no_grad()
def infer(models, x, arms, head, ac):
    """Sliding-window inference of every text arm on a normalised tensor x (D,H,W) >= WIN. Returns {fid: prob}."""
    stride = (max(1, WIN[0] // 2), 112, 112)
    batch = max(1, (8 * 12) // WIN[0])
    wins = sliding_windows(x.shape, WIN, stride)
    g = gaussian_weight(WIN).to(x.device)
    out = {fid: torch.zeros(x.shape, device=x.device) for fid in arms}
    norm = torch.zeros(x.shape, device=x.device)
    for mi, model in enumerate(models):
        for i0 in range(0, len(wins), batch):
            ws = wins[i0:i0 + batch]
            xb = torch.stack([x[a:a + WIN[0], b:b + WIN[1], c:c + WIN[2]] for a, b, c in ws])[:, None]
            with ac():
                feats = model._encode(xb, None, None)
            feats = {k: v.detach() for k, v in feats.items()}
            for fid, (tt, tmk) in arms.items():
                t = tt[None].expand(len(ws), -1, -1)
                mk = tmk[None].expand(len(ws), -1)
                with ac():
                    temb = RefNet.pool(t, mk)
                    o = model.run_head(head, feats, xb, temb)
                pr = torch.softmax(o.float(), 1)[:, 1] * g
                for j, (aa, bb, cc) in enumerate(ws):
                    out[fid][aa:aa + WIN[0], bb:bb + WIN[1], cc:cc + WIN[2]] += pr[j]
            if mi == 0:
                for aa, bb, cc in ws:
                    norm[aa:aa + WIN[0], bb:bb + WIN[1], cc:cc + WIN[2]] += g
    return {fid: v / (norm * len(models)) for fid, v in out.items()}, len(wins)


def pad_to_win(x):
    pad = [max(0, WIN[k] - x.shape[k]) for k in range(3)]
    return F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]), value=float(x.min())) if any(pad) else x


@torch.no_grad()
def predict_scan(models, root, stem, fids, bank, dev, head, dtype=torch.bfloat16):
    """Mean over models of the per-finding probability maps on the cache grid. Returns ({fid: (Dc,Hc,Wc)}, n_windows)."""
    ac = (lambda: torch.autocast('cuda', dtype=dtype)) if dtype is not None else (lambda: torch.autocast('cuda', enabled=False))
    ct, _, _ = load_volume(root, stem)
    ct = np.array(ct)
    x = pad_to_win(normalise(ct).to(dev))
    arms = {}
    for fid in fids:
        r = bank.get(stem, fid)
        if r is not None:
            arms[fid] = (r[0].to(dev), r[1].to(dev))
    probs, nwin = infer(models, x, arms, head, ac)
    crop = tuple(slice(0, ct.shape[k]) for k in range(3))
    return {fid: v[crop] for fid, v in probs.items()}, nwin, arms, ac


@torch.no_grad()
def repass(models, probs, arms, head, ac, meta, ct_native, dev, scale=2.0, thr=0.5, margin_mm=10.0, max_ml=0.0,
           mode='replace', max_comp=50, box='window'):
    """Coarse-to-fine second pass: every component of the pass-1 mask is boxed, the box is cut from the NATIVE CT and
    resampled to (cache spacing / scale) — i.e. the network sees the finding at 2x the resolution it was trained
    at, so a 4x8x8 patch covers half the anatomy — and the refined probability replaces (or is averaged with) the
    pass-1 probability inside the box. Returns {fid: refined prob on the cache grid}, n_boxes, n_windows."""
    from scipy.ndimage import label as cc_label
    sp_c = np.array(SPACING, float)                           # cache spacing (z,y,x)
    sp_n = np.array(meta['orig_spacing'], float)              # native spacing (z,y,x)
    shape_c = np.array(meta['shape']); shape_n = np.array(ct_native.shape)
    ratio = shape_n / shape_c                                 # native voxels per cache voxel, per axis
    tgt = sp_c / scale                                        # target spacing of the crop
    out, n_boxes, n_win = {}, 0, 0
    for fid, p1 in probs.items():
        p2 = p1.clone()
        m = (p1 >= thr).cpu().numpy()
        if not m.any():
            out[fid] = p2; continue
        cc, n = cc_label(m, structure=np.ones((3, 3, 3)))
        sizes = np.bincount(cc.ravel()); sizes[0] = 0
        order = [int(i) for i in np.argsort(-sizes) if sizes[i] >= 10][:max_comp]
        for ci in order:
            vol_ml = sizes[ci] * float(np.prod(sp_c)) / 1000.0
            if max_ml and vol_ml > max_ml:
                continue
            nz = np.nonzero(cc == ci)
            if box == 'window':
                # a full network window (WIN voxels at the target spacing) centred on the component: real context at
                # 2x resolution instead of an air-padded island (the crop-and-pad protocol is itself out of distribution)
                ctr = np.array([v.mean() for v in nz])
                half = np.array(WIN) * tgt / sp_c / 2.0                     # window half-size in cache voxels
                lo_c = np.clip(np.round(ctr - half).astype(int), 0, None); hi_c = np.minimum(lo_c + np.round(2 * half).astype(int), shape_c)
                lo_c = np.maximum(hi_c - np.round(2 * half).astype(int), 0)
            else:
                lo_c = np.array([int(v.min()) for v in nz]); hi_c = np.array([int(v.max()) + 1 for v in nz])
                mg = np.ceil(margin_mm / sp_c).astype(int)
                lo_c = np.maximum(lo_c - mg, 0); hi_c = np.minimum(hi_c + mg, shape_c)
            lo_n = np.floor(lo_c * ratio).astype(int); hi_n = np.minimum(np.ceil(hi_c * ratio).astype(int), shape_n)
            crop = ct_native[lo_n[0]:hi_n[0], lo_n[1]:hi_n[1], lo_n[2]:hi_n[2]]
            new = tuple(max(1, int(round(crop.shape[k] * sp_n[k] / tgt[k]))) for k in range(3))
            xc = torch.from_numpy(np.ascontiguousarray(crop))[None, None].to(dev)
            xc = F.interpolate(xc, size=new, mode='trilinear', align_corners=False)[0, 0]
            xc = pad_to_win(normalise(xc.cpu().numpy().astype(np.int16)).to(dev))
            pc, nw = infer(models, xc, {fid: arms[fid]}, head, ac)
            n_win += nw; n_boxes += 1
            pc = pc[fid][:new[0], :new[1], :new[2]][None, None]
            box = tuple(int(v) for v in (hi_c - lo_c))
            pr = F.interpolate(pc, size=box, mode='trilinear', align_corners=False)[0, 0]
            sl = (slice(lo_c[0], hi_c[0]), slice(lo_c[1], hi_c[1]), slice(lo_c[2], hi_c[2]))
            p2[sl] = pr if mode == 'replace' else 0.5 * (p1[sl] + pr)
        out[fid] = p2
    return out, n_boxes, n_win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True, help='comma list of referring2 run dirs (ensemble = mean prob)')
    ap.add_argument('--head', required=True)
    ap.add_argument('--root', required=True, help='cache dir holding <stem>_{ct,seg,inst}.npy + _meta.json')
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--dataset-json', default=REX_JSON)
    ap.add_argument('--out', required=True)
    ap.add_argument('--thr', type=float, default=0.5)
    ap.add_argument('--win-z', type=int, required=True)
    ap.add_argument('--bank', default=None, help='extra token bank npz (test findings)')
    ap.add_argument('--save-prob', action='store_true', help='also write uint8 probability maps (x255) to <out>_prob')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--dtype', default='bf16', choices=list(AUTOCAST))
    ap.add_argument('--shard', default='0/1', help='i/n: process entries i, i+n, ... (array jobs)')
    ap.add_argument('--repass', default='', help="comma list of second-pass modes to also write: 'replace', 'mean'")
    ap.add_argument('--repass-scale', type=float, default=2.0)
    ap.add_argument('--repass-margin-mm', type=float, default=10.0)
    ap.add_argument('--repass-max-ml', type=float, default=0.0, help='refine only components up to this volume (0 = all)')
    ap.add_argument('--repass-box', default='window', choices=['window', 'bbox'], help="'window': full WIN-sized box at the target spacing centred on the component; 'bbox': component bbox + margin (air-padded)")
    args = ap.parse_args()
    log('args', vars(args)); log('host', os.uname().nodename, 'WIN', WIN)
    if WIN[0] != args.win_z:
        raise SystemExit(f'PRIMUS_WIN_Z={WIN[0]} but --win-z {args.win_z}: set the environment to the run geometry')
    import nibabel as nib
    dev = 'cuda'
    models = []
    for r in args.runs.split(','):
        m, sd, steer = load_model(r, dev)
        if steer:
            raise SystemExit('SteerViT runs are not supported (encoder depends on text)')
        if args.head not in m.heads:
            raise SystemExit(f'{r}: head {args.head} not in {list(m.heads)}')
        from primus.model import enable_sdpa
        n_p, d_p = enable_sdpa(m.vit)
        log(f'SDPA attention: {n_p} modules, probe |naive - sdpa| max {d_p:.2e}')
        models.append(m)
        log(f'loaded {r}: finetune={bool(sd["args"].get("finetune"))} heads={list(m.heads)}')
    paths = [os.environ.get("TAPCT_TEXT_BANK", "cache/rexground_text_tokens_qwen06b.npz")] + ([args.bank] if args.bank else [])
    bank = MergedBank(paths)
    entries = json.load(open(args.dataset_json))[args.split]
    entries = list(entries.values()) if isinstance(entries, dict) else entries
    entries.sort(key=lambda e: e['name'])
    if args.limit:
        entries = entries[:args.limit]
    si, sn = (int(v) for v in args.shard.split('/'))
    entries = entries[si::sn]
    log(f'shard {si}/{sn}: {len(entries)} scans, dtype {args.dtype}')
    os.makedirs(args.out, exist_ok=True)
    if args.save_prob:
        os.makedirs(args.out + '_prob', exist_ok=True)
    t0 = time.time()
    n_missing_text = 0
    for i, e in enumerate(entries):
        name = e['name']; stem = name[:-7]
        dst = os.path.join(args.out, name)
        if os.path.exists(dst) and not args.overwrite:
            continue
        meta = load_meta(args.root, stem)
        fids = sorted(e['findings'], key=int)
        assert fids == [str(k) for k in range(len(fids))], f'{name}: non-contiguous finding ids {fids}'
        ctimg = nib.load(rex_ct_path(name))
        orig = tuple(meta['orig_shape'])                        # (z,y,x) after LPS reorientation
        assert tuple(ctimg.shape[:3]) == (orig[2], orig[1], orig[0]), (ctimg.shape, orig)
        probs, nwin, arms, ac = predict_scan(models, args.root, stem, fids, bank, dev, args.head, AUTOCAST[args.dtype])
        variants = {'': probs}
        if args.repass:
            ct_native = nib_to_lps_zyx(np.asanyarray(ctimg.dataobj).astype(np.float32), ctimg.affine)
            for mode in args.repass.split(','):
                pr2, nb, nw2 = repass(models, probs, arms, args.head, ac, meta, ct_native, dev, scale=args.repass_scale,
                                      thr=args.thr, margin_mm=args.repass_margin_mm, max_ml=args.repass_max_ml, mode=mode,
                                      box=args.repass_box)
                variants[f'_repass_{mode}'] = pr2
                log(f'  repass {mode}: {nb} boxes, {nw2} windows')
        for suffix, probs_v in variants.items():
            write_scan(args, ctimg, orig, fids, probs_v, name, suffix)
        del probs, variants
        torch.cuda.empty_cache()
        log(f'[{i + 1}/{len(entries)}] {name} F={len(fids)} windows={nwin} {(time.time() - t0) / (i + 1):.1f}s/scan')
    log(f'done: {len(entries)} scans, out={args.out}')


def write_scan(args, ctimg, orig, fids, probs, name, suffix=''):
    import nibabel as nib
    out_dir = args.out + suffix
    os.makedirs(out_dir, exist_ok=True)
    if args.save_prob:
        os.makedirs(out_dir + '_prob', exist_ok=True)
    masks, pmaps = [], []
    for fid in fids:
            if fid not in probs:
                masks.append(np.zeros(ctimg.shape[:3], np.uint8))
                if args.save_prob:
                    pmaps.append(np.zeros(ctimg.shape[:3], np.uint8))
                continue
            p = probs[fid][None, None]
            p = F.interpolate(p, size=orig, mode='trilinear', align_corners=False)[0, 0]
            p_np = lps_zyx_to_ct_grid(p.cpu().numpy(), ctimg.affine)
            masks.append((p_np > args.thr).astype(np.uint8))
            if args.save_prob:
                pmaps.append(np.rint(p_np * 255).astype(np.uint8))
    nib.save(nib.Nifti1Image(np.stack(masks, 0), ctimg.affine), os.path.join(out_dir, name))
    if args.save_prob:
        nib.save(nib.Nifti1Image(np.stack(pmaps, 0), ctimg.affine), os.path.join(out_dir + '_prob', name))


if __name__ == '__main__':
    main()

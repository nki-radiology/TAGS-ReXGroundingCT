"""Full-volume (scan-level) referring evaluation — the honest counterpart to the ROI metric.

The ROI evaluation in referring2.py centres a window on the GT finding, so it measures delineation given perfect
localisation ("oracle") and is blind to false positives elsewhere in the scan. This module instead runs the
sliding window over the WHOLE volume, conditioned on one finding's text at a time, and scores against the
full-volume mask. That is where per-block patch-detection deep supervision (--aux-w) is expected to pay off:
it teaches the tokens to say "the finding is NOT in this patch", which the ROI metric cannot reward.

Efficiency: for every route except SteerViT the ENCODER IS TEXT-INDEPENDENT, so each window is encoded once per
volume and the (cheap) heads are re-run per finding on the cached features. SteerViT checkpoints are rejected
rather than silently re-encoding per finding (it would be ~F times slower); pass --allow-steer to do it anyway.

Metrics per (head, finding, text arm):
    dice          volume Dice vs the finding's full mask
    recall/prec   voxel recall and precision over the whole volume
    n_fp_comp     26-connected predicted components not overlapping the GT finding (the localisation error the
                  ROI metric hides)
    fp_vol_mm3    total volume of those components
    roi_dice      the ROI-restricted number, recomputed here so both metrics come from the same forward pass
    dice_cat      volume Dice against the union of ALL annotated findings of the SAME category in that scan.
                  ~half of the "false-positive" volume lands on another annotated finding, which the
                  single-instance target scores as an error; this is the any-matching-finding upper variant.
                  Report both: `dice` is strict referring, `dice_cat` removes the annotation-protocol penalty.
Text arms: 'm' = the finding's own text, 's' = a text drawn from a different scan (fixed seed, shared across heads).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from primus.data import WIN, gaussian_weight, load_meta, load_volume, normalise, sliding_windows
from primus.referring import TokenTextBank, make_text_bank, log
from primus.referring2 import RefNet

from primus.data import VOX_MM3   # depends on PRIMUS_Z_MM; never assume 1.5 mm isotropic


def fp_components(pred_np, gt_np, other_np=None):
    """26-connected predicted components that do not touch the GT finding.

    `other_np`: union of the OTHER annotated findings in the same scan. A "false positive" that lands on one of
    them is a correct detection mis-scored by the single-instance target ("GGO in the right lower lobe" can match
    several GGOs while only one is annotated for this text), so it is counted separately.
    """
    from scipy import ndimage
    lab, n = ndimage.label(pred_np, structure=np.ones((3, 3, 3)))
    if n == 0:
        return 0, 0.0, 0, 0.0
    hit = {int(h) for h in np.unique(lab[gt_np]) if h > 0}
    on_other = {int(h) for h in np.unique(lab[other_np]) if h > 0} if other_np is not None else set()
    sizes = ndimage.sum(np.ones_like(lab, dtype=np.uint8), lab, range(1, n + 1))
    fp = [(j, float(v)) for j, v in enumerate(sizes, 1) if j not in hit]
    fp_other = [v for j, v in fp if j in on_other]
    return len(fp), float(sum(v for _, v in fp) * VOX_MM3), len(fp_other), float(sum(fp_other) * VOX_MM3)


def load_model(run, dev, allow_steer=False):
    sd = torch.load(os.path.join(run, 'last.pt'), map_location='cpu', weights_only=False)
    a, specs = sd['args'], sd['specs']
    steer = bool(a.get('steer'))
    if steer and not allow_steer:
        raise SystemExit(f'{run} is a SteerViT run: its encoder depends on the text, so features cannot be shared '
                         f'across findings. Re-run with --allow-steer if you accept the ~F x cost.')
    model = RefNet(a['variant'], specs, frozen=not a.get('finetune'), steer=steer,
                   aux=bool(a.get('aux_w', 0.0) > 0), stem=a.get('stem', 'lite')).to(dev)
    model.heads.load_state_dict(sd['heads'])
    if steer and sd.get('adapters'):
        model.adapters.load_state_dict(sd['adapters'])
    if 'vit' in sd:
        model.vit.load_state_dict(sd['vit'])
    model.eval()
    return model, sd, steer


@torch.no_grad()
def eval_volume(model, root, stems, bank, dev, heads=None, steer=False, shuffle_seed=1234, thr=0.5,
                batch=None, stride=None, max_findings=0, tag='', out=None):
    stride = stride or (max(1, WIN[0] // 2), 112, 112)
    batch = batch or max(1, (8 * 12) // WIN[0])   # deeper windows -> fewer per forward
    g = gaussian_weight(WIN).to(dev)
    keys = list(bank.idx)
    rng = np.random.default_rng(shuffle_seed)
    hnames = heads or list(model.heads)
    recs = []
    t0 = time.time()
    n_done = 0
    for s in stems:
        meta = load_meta(root, s)
        insts = [i for i in meta['instances'] if bank.get(s, str(i['id'] - 1)) is not None]
        if not insts:
            continue
        ct, seg, inst = load_volume(root, s)
        ct = np.array(ct)
        inst_np = np.array(inst)
        seg_np = np.array(seg)          # class map: 0 = bg, 1..14 = finding categories
        x = normalise(ct).to(dev)
        pad = [max(0, WIN[k] - x.shape[k]) for k in range(3)]
        if any(pad):
            x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]), value=float(x.min()))
        wins = sliding_windows(x.shape, WIN, stride)
        shape = x.shape

        # text arms for every finding of this volume, fixed across heads
        arms = {}
        for ins in insts:
            fid = str(ins['id'] - 1)
            tm, mm = bank.get(s, fid)
            k2 = keys[rng.integers(len(keys))]
            while k2.split('|')[0] == s:
                k2 = keys[rng.integers(len(keys))]
            arms[fid] = {'m': (tm.to(dev), mm.to(dev)),
                         's': (bank.emb[bank.idx[k2]].float().to(dev), bank.mask[bank.idx[k2]].to(dev))}

        acc = {(h, fid, a): torch.zeros(shape, device=dev) for h in hnames for fid in arms for a in ('m', 's')}
        norm = torch.zeros(shape, device=dev)
        dw = {}                                              # text-conditioned depth weights, per (head, finding, arm)
        for i0 in range(0, len(wins), batch):
            ws = wins[i0:i0 + batch]
            xb = torch.stack([x[a:a + WIN[0], b:b + WIN[1], c:c + WIN[2]] for a, b, c in ws])[:, None]
            feats = None
            if not steer:                                   # encoder is text-independent -> encode once per window
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    feats = model._encode(xb, None, None)
                feats = {k: v.detach() for k, v in feats.items()}
            for fid, per_arm in arms.items():
                for a, (tt, tmk) in per_arm.items():
                    t = tt[None].expand(len(ws), -1, -1)
                    mk = tmk[None].expand(len(ws), -1)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        if steer:
                            f_ = model._encode(xb, t, mk)
                        else:
                            f_ = feats
                        temb = RefNet.pool(t, mk)
                        for h in hnames:
                            o = model.run_head(h, f_, xb, temb)
                            if getattr(model.heads[h], 'last_w', None) is not None:
                                dw.setdefault((h, fid, a), []).append(model.heads[h].last_w.float().mean(0).tolist())
                            p = torch.softmax(o.float(), 1)[:, 1] * g
                            for j, (aa, bb, cc) in enumerate(ws):
                                acc[(h, fid, a)][aa:aa + WIN[0], bb:bb + WIN[1], cc:cc + WIN[2]] += p[j]
            for aa, bb, cc in ws:
                norm[aa:aa + WIN[0], bb:bb + WIN[1], cc:cc + WIN[2]] += g

        crop = tuple(slice(0, ct.shape[k]) for k in range(3))
        for ins in insts:
            fid = str(ins['id'] - 1)
            gt = torch.from_numpy((inst_np == ins['id']).astype(np.uint8)).to(dev).bool()
            gt_np = gt.cpu().numpy()
            lo = [max(0, ins['lo'][k] - (8, 48, 48)[k]) for k in range(3)]
            hi = [min(ct.shape[k], ins['hi'][k] + 1 + (8, 48, 48)[k]) for k in range(3)]
            roi = tuple(slice(lo[k], hi[k]) for k in range(3))
            rec = dict(stem=s, fid=fid, cls=ins['cls'], vol_mm3=ins['n_vox'] * VOX_MM3, m={}, s={})
            other = torch.from_numpy(((inst_np > 0) & (inst_np != ins['id']))).to(dev).bool()
            other_np = other.cpu().numpy()
            # union of every annotated finding of the SAME category, i.e. the target a referring phrase like
            # "GGO in the right lower lobe" can legitimately match when the scan holds several GGOs
            cat = torch.from_numpy(seg_np == ins['cls']).to(dev).bool()
            nrefc = float(cat.sum())
            for h in hnames:
                for a in ('m', 's'):
                    prob = (acc[(h, fid, a)] / norm)[crop]
                    sweep = {}
                    for t2 in (0.3, 0.5, 0.7, 0.9):
                        q = prob > t2
                        tp2 = float((q & gt).sum()); np2 = float(q.sum())
                        sweep[str(t2)] = dict(
                            dice=(2 * tp2 / (np2 + float(gt.sum()))) if (np2 + float(gt.sum())) > 0 else None,
                            prec=(tp2 / np2) if np2 else None, vol_mm3=np2 * VOX_MM3)
                    pr = prob > thr
                    tp = float((pr & gt).sum()); npred = float(pr.sum()); nref = float(gt.sum())
                    d = 2 * tp / (npred + nref) if (npred + nref) > 0 else None
                    pr_roi = pr[roi]; gt_roi = gt[roi]
                    dr = (2 * float((pr_roi & gt_roi).sum()) / (float(pr_roi.sum()) + float(gt_roi.sum()))
                          if (pr_roi.sum() + gt_roi.sum()) > 0 else None)
                    nfp, fpv, nfp_other, fpv_other = fp_components(pr.cpu().numpy(), gt_np, other_np)
                    tpc = float((pr & cat).sum())
                    dcat = 2 * tpc / (npred + nrefc) if (npred + nrefc) > 0 else None
                    w = dw.get((h, fid, a))
                    rec[a][h] = dict(dice=d, roi_dice=dr, sweep=sweep,
                                     depth_w=[round(x, 4) for x in np.mean(w, 0).tolist()] if w else None,
                                     recall=tp / nref if nref else None,
                                     prec=tp / npred if npred else None,
                                     n_fp_comp=nfp, fp_vol_mm3=fpv, pred_vol_mm3=npred * VOX_MM3,
                                     n_fp_on_other=nfp_other, fp_other_mm3=fpv_other,
                                     dice_cat=dcat, recall_cat=tpc / nrefc if nrefc else None,
                                     prec_cat=tpc / npred if npred else None,
                                     cat_vol_mm3=nrefc * VOX_MM3, cls=int(ins['cls']))
            recs.append(rec)
            n_done += 1
        del acc, norm
        torch.cuda.empty_cache()
        log(f'  {tag} {n_done} findings ({s}) {(time.time()-t0)/max(n_done,1):.1f}s/finding')
        if max_findings and n_done >= max_findings:
            break
    if out:
        json.dump(recs, open(out, 'w'))
    return recs


def summarise(recs, hnames):
    out = {}
    for h in hnames:
        for a in ('m', 's'):
            d = [r[a][h]['dice'] for r in recs if r[a][h]['dice'] is not None]
            rd = [r[a][h]['roi_dice'] for r in recs if r[a][h]['roi_dice'] is not None]
            fp = [r[a][h]['n_fp_comp'] for r in recs]
            pv = [r[a][h]['pred_vol_mm3'] for r in recs]
            dc = [r[a][h].get('dice_cat') for r in recs if r[a][h].get('dice_cat') is not None]
            out[f'{h}|{a}'] = dict(vol_dice=round(float(np.mean(d)), 4) if d else None,
                                   dice_cat=round(float(np.mean(dc)), 4) if dc else None,
                                   roi_dice=round(float(np.mean(rd)), 4) if rd else None,
                                   fp_comp=round(float(np.mean(fp)), 1),
                                   pred_vol_mm3=round(float(np.mean(pv)), 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='directory holding last.pt (a referring2 run)')
    ap.add_argument('--root', required=True)
    ap.add_argument('--heads', default=None, help='comma list; default = all heads in the checkpoint')
    ap.add_argument('--max-findings', type=int, default=0)
    ap.add_argument('--thr', type=float, default=0.5)
    ap.add_argument('--batch', type=int, default=0)
    ap.add_argument('--allow-steer', action='store_true')
    ap.add_argument('--tag', default='vol')
    args = ap.parse_args()
    dev = 'cuda'
    log('args', vars(args)); log('host', os.uname().nodename)
    if not os.path.isdir(args.root) or not any(n.endswith('_meta.json') for n in os.listdir(args.root)[:50]):
        raise SystemExit(f'cache not visible on {os.uname().nodename}: {args.root} — node-local /processing copies '
                         f'exist on herakles and euctemon only; pin with --nodelist or rsync the cache first.')
    model, sd, steer = load_model(args.run, dev, args.allow_steer)
    bank = make_text_bank(sd['args'].get('text_bank', 'tokens')); log('text bank', sd['args'].get('text_bank', 'tokens'))
    va = json.load(open(os.path.join(args.run, 'split.json')))['val']
    hnames = args.heads.split(',') if args.heads else list(model.heads)
    log(f'{args.run}: heads {hnames} | steer={steer} | finetune={bool(sd["args"].get("finetune"))} '
        f'| aux_w={sd["args"].get("aux_w", 0.0)} | {len(va)} val volumes')
    recs = eval_volume(model, args.root, va, bank, dev, heads=hnames, steer=steer, thr=args.thr,
                       batch=args.batch, max_findings=args.max_findings, tag=args.tag,
                       out=os.path.join(args.run, f'eval_volume_{args.tag}.json'))
    log(f'VOLUME-LEVEL n={len(recs)}')
    for k, v in summarise(recs, hnames).items():
        log(f'  {k}: {v}')


if __name__ == '__main__':
    main()

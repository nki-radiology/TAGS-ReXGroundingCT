"""Sliding-window inference for every head + volume/class/instance metrics."""
import json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from primus.data import WIN, load_meta, load_volume, normalise, sliding_windows, gaussian_weight

from primus.data import VOX_MM3   # depends on PRIMUS_Z_MM; never assume 1.5 mm isotropic


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


@torch.no_grad()
def predict_volume(model, ct, n_classes, dev, batch=16, stride=(6, 112, 112)):
    """ct int16 (D,H,W) -> dict head -> argmax label map (D,H,W) uint8 on CPU."""
    model.eval()
    shape = ct.shape
    pad = [max(0, w - s) for w, s in zip(WIN, shape)]
    x = normalise(np.asarray(ct))
    if any(pad):
        x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]), value=(-1024 - (-86.8086)) / 322.6347)
    x = x.to(dev)
    wins = sliding_windows(x.shape, WIN, stride)
    g = gaussian_weight(WIN).to(dev)
    acc = {k: torch.zeros((n_classes,) + tuple(x.shape), device=dev, dtype=torch.float32) for k in model.heads}
    norm = torch.zeros(tuple(x.shape), device=dev, dtype=torch.float32)
    for i in range(0, len(wins), batch):
        ws = wins[i:i + batch]
        xb = torch.stack([x[a:a + WIN[0], b:b + WIN[1], c:c + WIN[2]] for a, b, c in ws])[:, None]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outs = model(xb)
        for k, o in outs.items():
            p = torch.softmax(o.float(), 1) * g
            for j, (a, b, c) in enumerate(ws):
                acc[k][:, a:a + WIN[0], b:b + WIN[1], c:c + WIN[2]] += p[j]
        for (a, b, c) in ws:
            norm[a:a + WIN[0], b:b + WIN[1], c:c + WIN[2]] += g
    preds = {}
    for k in acc:
        pr = (acc[k] / norm).argmax(0)[:shape[0], :shape[1], :shape[2]]
        preds[k] = pr.to(torch.uint8)
    return preds


def dice_per_class(pred, gt, n_classes):
    """pred, gt: uint8 tensors on the same device -> list over classes 1..C-1 of Dice or None (both empty)."""
    out = []
    for c in range(1, n_classes):
        p = pred == c; t = gt == c
        np_, nt = int(p.sum()), int(t.sum())
        if np_ == 0 and nt == 0:
            out.append(None); continue
        out.append(float(2 * (p & t).sum()) / (np_ + nt))
    return out


def instance_metrics(pred, inst, seg, insts_meta, binary=False):
    """Per GT instance: voxel recall by foreground-any and by correct class; predicted-FG overlap."""
    pf = pred > 0
    recs = []
    for m in insts_meta:
        i = m['id']
        mk = inst == i
        n = int(mk.sum())
        if n == 0:
            continue
        rec_any = float((pf & mk).sum()) / n
        rec_cls = float(((pred == (1 if binary else m['cls'])) & mk).sum()) / n
        # local Dice: prediction vs this instance inside the instance's bounding box dilated by max(8 vox, 50 % extent)
        lo, hi = m['lo'], m['hi']
        sl = tuple(slice(max(0, lo[k] - max(8, (hi[k] - lo[k]) // 2)), hi[k] + 1 + max(8, (hi[k] - lo[k]) // 2)) for k in range(3))
        pl = pf[sl]; ml = mk[sl]
        den = int(pl.sum()) + int(ml.sum())
        ldice = float(2 * (pl & ml).sum()) / den if den else 0.0
        recs.append(dict(id=i, cls=m['cls'], n_vox=n, vol_mm3=n * VOX_MM3, recall_any=rec_any, recall_cls=rec_cls, local_dice=ldice))
    return recs


def fp_components(pred_np, gt_fg_np):
    from scipy import ndimage
    lab, n = ndimage.label(pred_np > 0, structure=np.ones((3, 3, 3)))
    if n == 0:
        return 0, 0, []
    hit = np.unique(lab[gt_fg_np])
    hit = set(int(h) for h in hit if h > 0)
    sizes = ndimage.sum(np.ones_like(lab, dtype=np.uint8), lab, range(1, n + 1))
    fp_sizes = [float(s) * VOX_MM3 for j, s in enumerate(sizes, 1) if j not in hit]
    return n, len(fp_sizes), fp_sizes


def evaluate_volumes(model, root, stems, n_classes, dev, tag='', out=None, stride=(6, 112, 112), components=None, binary=False):
    """Returns dict(per_head -> summary, per_volume -> records). components: count predicted components (slow);
    defaults to True when `out` is given (final evaluation)."""
    if components is None:
        components = out is not None
    per_vol = []
    t0 = time.time()
    for vi, s in enumerate(stems):
        meta = load_meta(root, s)
        ct, seg, inst = load_volume(root, s)
        preds = predict_volume(model, ct, n_classes, dev, stride=stride)
        seg_np = np.asarray(seg)
        if binary:
            seg_np = (seg_np > 0).astype(np.uint8)
        gt = torch.from_numpy(seg_np).to(dev)
        inst_t = torch.from_numpy(np.asarray(inst)).to(dev)
        rec = dict(stem=s, n_inst=len(meta['instances']), heads={})
        gt_fg_np = np.asarray(seg) > 0 if components else None
        for k, pr in preds.items():
            d = dice_per_class(pr, gt, n_classes)
            fgd = None
            if n_classes > 2:
                # binary foreground Dice as well (class-agnostic)
                p = pr > 0; t = gt > 0
                den = int(p.sum()) + int(t.sum())
                fgd = float(2 * (p & t).sum()) / den if den > 0 else None
            else:
                fgd = d[0]
            im = instance_metrics(pr, inst_t, gt, meta['instances'], binary=binary)
            r = dict(dice=d, dice_fg=fgd, inst=im)
            if components:
                n_comp, n_fp, fp_sizes = fp_components(pr.cpu().numpy(), gt_fg_np)
                r.update(n_pred_comp=n_comp, n_fp_comp=n_fp, fp_sizes_mm3=fp_sizes)
            rec['heads'][k] = r
        per_vol.append(rec)
        if (vi + 1) % 10 == 0:
            log(f'  eval {tag} {vi+1}/{len(stems)} {(time.time()-t0)/(vi+1):.1f}s/vol')
    summary = {}
    for k in model.heads:
        fg = [v['heads'][k]['dice_fg'] for v in per_vol if v['heads'][k]['dice_fg'] is not None]
        cls = [[v['heads'][k]['dice'][c] for v in per_vol if v['heads'][k]['dice'][c] is not None] for c in range(n_classes - 1)]
        insts = [i for v in per_vol for i in v['heads'][k]['inst']]
        summary[k] = dict(mean_dice_fg=float(np.mean(fg)) if fg else float('nan'), n_vol=len(fg),
                          mean_dice_per_class=[float(np.mean(c)) if c else None for c in cls],
                          mean_inst_recall_any=float(np.mean([i['recall_any'] for i in insts])) if insts else None,
                          mean_inst_recall_cls=float(np.mean([i['recall_cls'] for i in insts])) if insts else None,
                          n_inst=len(insts))
        if components:
            summary[k]['mean_fp_comp'] = float(np.mean([v['heads'][k]['n_fp_comp'] for v in per_vol]))
    res = dict(tag=tag, per_head=summary, per_volume=per_vol)
    if out:
        json.dump(res, open(os.path.join(out, f'eval_{tag}.json'), 'w'))
    return res

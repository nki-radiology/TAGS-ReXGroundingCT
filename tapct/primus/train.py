"""Train Primus-style decoders on TAP-CT block features.

Mode 'frozen': the encoder runs once per batch under no_grad and ALL heads (one per tapped block + multi-block
combos) are trained simultaneously on the same features -> one job answers "which block segments best".
Mode 'finetune': encoder truncated at the deepest tapped block and trained end-to-end with a single head.
Recipe follows Primus (AdamW 3e-4, wd 5e-2, betas (0.9,0.98), grad-clip 1, linear warmup -> poly decay,
Dice+CE, no deep supervision).
"""
import argparse, json, os, sys, time, math
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from primus.data import CropDataset, split_stems, load_meta
from primus.model import TapSeg, default_heads, parse_heads, make_loss
from primus.evaluate import evaluate_volumes


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def poly_lr(base, it, total, warmup, power=0.9):
    if it < warmup:
        return base * (it + 1) / warmup
    return base * (1 - (it - warmup) / max(1, total - warmup)) ** power


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['pancancer', 'rexground'])
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--variant', default='tap-ct-b-3d')
    ap.add_argument('--control', default='trained')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--mode', default='frozen', choices=['frozen', 'finetune'])
    ap.add_argument('--heads', default='all')
    ap.add_argument('--iters', type=int, default=25000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--enc-lr', type=float, default=5e-5)
    ap.add_argument('--wd', type=float, default=5e-2)
    ap.add_argument('--warmup', type=int, default=250)
    ap.add_argument('--drop-path', type=float, default=0.0)
    ap.add_argument('--fg-prob', type=float, default=0.5)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--val-every', type=int, default=5000)
    ap.add_argument('--val-subset', type=int, default=24)
    ap.add_argument('--val-list', default=None, help='json list of val stems (pancancer)')
    ap.add_argument('--loss', default='dc_ce', choices=['dc_ce', 'batchdc_ce', 'dc_ce_fg', 'batchdc_ce_fg'])
    ap.add_argument('--binary', action='store_true', help='collapse all foreground classes to 1 (rexground)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--init-heads', default=None, help='last.pt of a frozen run; copies the matching head weights')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log('args', vars(args)); log('host', os.uname().nodename)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = 'cuda'
    n_classes = 2 if (args.dataset == 'pancancer' or args.binary) else 15

    val_list = json.load(open(args.val_list)) if args.val_list else None
    tr, va = split_stems(args.root, args.dataset, val_list=val_list, seed=args.seed)
    log(f'train {len(tr)} val {len(va)} volumes')
    json.dump(dict(train=tr, val=va), open(os.path.join(args.out, 'split.json'), 'w'))

    heads = default_heads() if args.heads == 'all' else parse_heads(args.heads)
    if args.mode == 'finetune':
        assert len(heads) == 1, 'finetune trains one head'
    model = TapSeg(args.variant, heads, n_classes, control=args.control, frozen=(args.mode == 'frozen'),
                   drop_path=args.drop_path, ckpt=args.ckpt).to(dev)
    log(f'heads: {list(heads)}; encoder depth kept {model.depth}; tapped {model.tapped}')
    n_head = sum(p.numel() for p in model.heads.parameters()); n_enc = sum(p.numel() for p in model.vit.parameters() if p.requires_grad)
    log(f'trainable params: heads {n_head/1e6:.2f}M encoder {n_enc/1e6:.2f}M')

    # one optimizer per head so grad clipping and AdamW state are independent across heads
    opts = {k: torch.optim.AdamW(h.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98), fused=True)
            for k, h in model.heads.items()}
    enc_opt = None
    if args.mode == 'finetune':
        enc_params = [p for p in model.vit.parameters() if p.requires_grad]
        enc_opt = torch.optim.AdamW(enc_params, lr=args.enc_lr, weight_decay=args.wd, betas=(0.9, 0.98), fused=True)

    ds = CropDataset(args.root, tr, fg_prob=args.fg_prob, seed=args.seed, binary=args.binary)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, pin_memory=True, persistent_workers=True,
                    prefetch_factor=4)
    if args.init_heads:
        src = torch.load(args.init_heads, map_location='cpu', weights_only=False)['heads']
        for k in heads:
            sub = {kk[len(k) + 1:]: v for kk, v in src.items() if kk.startswith(k + '.')}
            model.heads[k].load_state_dict(sub)
            log(f'initialised head {k} from {args.init_heads} ({len(sub)} tensors)')
    start = 0
    ck = os.path.join(args.out, 'last.pt')
    if args.resume and os.path.exists(ck):
        sd = torch.load(ck, map_location='cpu')
        model.heads.load_state_dict(sd['heads'])
        if 'vit' in sd:
            model.vit.load_state_dict(sd['vit'])
        for k, o in opts.items():
            o.load_state_dict(sd['opts'][k])
        if enc_opt is not None:
            enc_opt.load_state_dict(sd['enc_opt'])
        start = sd['it'] + 1
        log(f'resumed at it {start}')

    loss_fn = make_loss(args.loss)
    hist = []
    t0 = time.time(); run_loss = {k: 0.0 for k in heads}; n_run = 0
    it = start
    for x, y in dl:
        if it >= args.iters:
            break
        lr = poly_lr(args.lr, it, args.iters, args.warmup)
        for o in opts.values():
            for g in o.param_groups:
                g['lr'] = lr
        if enc_opt is not None:
            for g in enc_opt.param_groups:
                g['lr'] = poly_lr(args.enc_lr, it, args.iters, args.warmup)
        x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
        model.train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outs = model(x)
        losses = {k: loss_fn(v, y) for k, v in outs.items()}
        for o in opts.values():
            o.zero_grad(set_to_none=True)
        if enc_opt is not None:
            enc_opt.zero_grad(set_to_none=True)
        sum(losses.values()).backward()
        for k, o in opts.items():
            torch.nn.utils.clip_grad_norm_(model.heads[k].parameters(), 1.0)
            o.step()
        if enc_opt is not None:
            torch.nn.utils.clip_grad_norm_([p for p in model.vit.parameters() if p.requires_grad], 1.0)
            enc_opt.step()
        for k, l in losses.items():
            run_loss[k] += float(l)
        n_run += 1
        if (it + 1) % 100 == 0:
            msg = ' '.join(f'{k}={v/n_run:.3f}' for k, v in run_loss.items())
            log(f'it {it+1} lr {lr:.2e} {(time.time()-t0)/(it+1-start):.2f}s/it | {msg}')
            hist.append(dict(it=it + 1, **{k: v / n_run for k, v in run_loss.items()}))
            run_loss = {k: 0.0 for k in heads}; n_run = 0
        if (it + 1) % args.val_every == 0 or (it + 1) == args.iters:
            torch.save(dict(it=it, heads=model.heads.state_dict(), opts={k: o.state_dict() for k, o in opts.items()},
                            **({'vit': model.vit.state_dict(), 'enc_opt': enc_opt.state_dict()} if enc_opt else {}),
                            args=vars(args), heads_spec=heads), ck)
            json.dump(hist, open(os.path.join(args.out, 'loss_hist.json'), 'w'))
            sub = va[:args.val_subset] if (it + 1) < args.iters else va
            res = evaluate_volumes(model, args.root, sub, n_classes, dev, tag=f'it{it+1}', out=args.out if (it + 1) == args.iters else None, binary=args.binary)
            summ = {k: round(v['mean_dice_fg'], 4) for k, v in res['per_head'].items()}
            log(f'VAL it {it+1} n={len(sub)} mean fg Dice per head: {summ}')
            json.dump(res, open(os.path.join(args.out, f'val_it{it+1}.json'), 'w'))
        it += 1
    log('done')


if __name__ == '__main__':
    main()

"""Referring segmentation on ReXGroundingCT with block-wise TAP-CT taps and FiLM text conditioning.

Question 1 (dHT follow-up): at which encoder depth do *referable* concepts live — which block supports
text-conditioned segmentation best?
Question 2 (text infusion): how much does the finding text add? Each text head is evaluated with matched AND
shuffled text; 'notxt' heads (zero embedding) are the unconditional baseline.

Item = (volume, finding). Sampling per draw: 60 % crop centred on the finding (matched text, target = finding);
20 % random in-body crop of the same volume (matched text, target = finding mask in that crop, usually empty);
20 % crop centred on the finding but with a MISMATCHED text from another volume -> target empty (teaches text
dependence). Target uses the instance map (smallest-wins), so overlapped parts of large findings are excluded —
noted as approximation. Loss: binary Dice+CE.
"""
import argparse, json, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from primus.data import WIN, load_meta, load_volume, normalise, sliding_windows, gaussian_weight, split_stems
from primus.model import TapSeg, Head, parse_heads, make_loss

TEXTS = os.environ.get('TAPCT_TEXT_POOLED', 'cache/rexground_text_qwen06b.npz')
from primus.data import VOX_MM3   # depends on PRIMUS_Z_MM; never assume 1.5 mm isotropic


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


class TextBank:
    def __init__(self, path=TEXTS):
        z = np.load(path, allow_pickle=False)
        self.emb = torch.from_numpy(z['emb'])
        self.idx = {k: i for i, k in enumerate(z['keys'].tolist())}
        self.cats = z['cats'].tolist()

    def get(self, stem, fid):
        i = self.idx.get(f'{stem}|{fid}')
        return None if i is None else self.emb[i]


class TokenTextBank:
    """Token-level text states (L,1024) fp16 + mask per finding. Several npz files may be merged; keys are
    '<stem>|<fid>' for the finding's own text, '<stem>|<fid>|p<k>' for paraphrase / template variants
    (positives) and '<stem>|<fid>|n<k>' for hard negatives. `idx` holds base keys only, so evaluation code that
    samples shuffled texts from `idx` is unaffected by the variants."""

    def __init__(self, path=os.environ.get("TAPCT_TEXT_BANK", "cache/rexground_text_tokens_qwen06b.npz")):
        paths = [path] if isinstance(path, str) else list(path)
        embs, masks, keys = [], [], []
        for pth in paths:
            z = np.load(pth, allow_pickle=False)
            embs.append(torch.from_numpy(z['emb'])); masks.append(torch.from_numpy(z['mask'])); keys += z['keys'].tolist()
        self.emb = torch.cat(embs) if len(embs) > 1 else embs[0]           # (n, L, 1024) fp16
        self.mask = torch.cat(masks) if len(masks) > 1 else masks[0]       # (n, L) bool
        self.idx, self.pos, self.neg = {}, {}, {}
        for i, k in enumerate(keys):
            parts = k.split('|')
            base = '|'.join(parts[:2])
            if len(parts) == 2:
                self.idx.setdefault(k, i)
            elif parts[2].startswith('p'):
                self.pos.setdefault(base, []).append(i)
            elif parts[2].startswith('n'):
                self.neg.setdefault(base, []).append(i)
        self.n_pos = sum(len(v) for v in self.pos.values()); self.n_neg = sum(len(v) for v in self.neg.values())

    def _row(self, i):
        return self.emb[i].float(), self.mask[i]

    def get(self, stem, fid):
        i = self.idx.get(f'{stem}|{fid}')
        return None if i is None else self._row(i)

    def get_pos(self, stem, fid, rng, p=0.0):
        """The finding's text, or with probability p one of its positive variants (if any)."""
        v = self.pos.get(f'{stem}|{fid}')
        if v and p > 0 and rng.random() < p:
            return self._row(v[rng.randrange(len(v))])
        return self.get(stem, fid)

    def get_neg(self, stem, fid, rng):
        v = self.neg.get(f'{stem}|{fid}')
        return self._row(v[rng.randrange(len(v))]) if v else None


class PooledTextBank:
    """A pooled (n,1024) bank presented through the TokenTextBank interface: one 'token' per finding with a
    one-element mask, so RefNet.pool() returns the stored vector unchanged. Used for SPECTRE's own text space
    (cache/rexground_text_spectre.npz), which is pooled by construction. SteerViT (token K/V) cannot use it."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.emb = torch.from_numpy(z['emb'])[:, None, :]       # (n, 1, 1024)
        self.mask = torch.ones(self.emb.shape[0], 1, dtype=torch.bool)
        self.idx = {k: i for i, k in enumerate(z['keys'].tolist())}
        self.path = path

    def get(self, stem, fid):
        i = self.idx.get(f'{stem}|{fid}')
        return None if i is None else (self.emb[i].float(), self.mask[i])


def make_text_bank(spec):
    """'tokens' (default Qwen3 token states), 'tokens+<npz>+<npz>' (merged with variant/extra banks), or a path
    to a pooled npz (SPECTRE text space)."""
    if spec in (None, '', 'tokens'):
        return TokenTextBank()
    if spec.startswith('tokens+'):
        return TokenTextBank([os.environ.get("TAPCT_TEXT_BANK", "cache/rexground_text_tokens_qwen06b.npz")] + spec.split('+')[1:])
    return PooledTextBank(spec)


class GatedTextCrossAttn(nn.Module):
    """SteerViT adapter (Ruthardt & Gaur et al., arXiv 2604.02327; vendored from dHT tokenizer3d/steer.py):
    x <- x + tanh(alpha) * CrossAttn(q=x, kv=text-MLP(text tokens)), alpha init 0 -> identity at step 0."""

    def __init__(self, d_img, d_txt=1024, n_heads=8):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_img)
        self.txt_mlp = nn.Sequential(nn.LayerNorm(d_txt), nn.Linear(d_txt, d_img), nn.GELU(), nn.Linear(d_img, d_img))
        self.attn = nn.MultiheadAttention(d_img, n_heads, batch_first=True)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, txt, txt_mask):
        kv = self.txt_mlp(txt)
        out, _ = self.attn(self.norm_q(x), kv, kv, key_padding_mask=~txt_mask, need_weights=False)
        return x + torch.tanh(self.alpha) * out

    def gate(self):
        return float(torch.tanh(self.alpha.detach()))


class SteerReferringModel(nn.Module):
    """Frozen TAP-CT ViT with gated text cross-attention after every other block; PLAIN heads (no FiLM) —
    text reaches the decoder only through the steered visual tokens. Heads tap several depths so the depth
    profile of referable concepts is read from the same run; gates themselves are a per-depth readout."""

    def __init__(self, variant, specs, device='cuda'):
        super().__init__()
        need = max(b for bl, _ in specs.values() for b in bl)
        base = TapSeg(variant, {f'b{need}': ((need,), 'cat')}, 2, frozen=True)
        self.vit, self.win, self.patch, self.grid = base.vit, base.win, base.patch, base.grid
        self.nreg, self.dim = base.nreg, base.dim
        self.tapped = sorted({b for bl, _ in specs.values() for b in bl})
        self.adapters = nn.ModuleDict({str(i): GatedTextCrossAttn(self.dim) for i in range(2, 13, 2)})  # after blocks 2,4,...,12
        self.heads = nn.ModuleDict({k: Head(bl, self.dim, self.patch, 2, red) for k, (bl, red) in specs.items()})

    def encode(self, x, txt, txt_mask):
        """Adapters need gradients, so no no_grad here; the ViT's own params stay frozen (requires_grad False)."""
        self.vit.eval()
        h = self.vit._prepare_tokens_with_masks(x)
        feats = {}
        if 0 in self.tapped:
            feats[0] = h[:, self.nreg:]
        for i, blk in enumerate(self.vit.blocks, 1):
            h = blk(h)
            if str(i) in self.adapters:
                pt = self.adapters[str(i)](h[:, self.nreg:], txt, txt_mask)
                h = torch.cat([h[:, :self.nreg], pt], 1)
            if i in self.tapped:
                feats[i] = h[:, self.nreg:]
            if i >= max(self.tapped):
                break
        return feats

    def forward(self, x, txt, txt_mask):
        feats = self.encode(x, txt, txt_mask)
        return {k: hd(feats, self.grid) for k, hd in self.heads.items()}

    def gates(self):
        return {k: round(a.gate(), 4) for k, a in self.adapters.items()}


class FiLM(nn.Module):
    def __init__(self, dim=768, tdim=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(tdim, 512), nn.GELU(), nn.Linear(512, 2 * dim))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)   # identity at init

    def forward(self, h, t):
        g, b = self.net(t).chunk(2, -1)
        return h * (1 + g[:, None]) + b[:, None]


class TextHead(Head):
    """Head with FiLM on the (projected) tokens. use_text=False -> plain head fed zeros (unconditional)."""

    def __init__(self, blocks, dim, patch, n_classes, reduce='cat', use_text=True):
        super().__init__(blocks, dim, patch, n_classes, reduce)
        self.use_text = use_text
        self.film = FiLM(dim) if use_text else None

    def forward(self, feats, grid, temb=None):
        hs = [n(feats[b]) for n, b in zip(self.norms, self.blocks)]
        h = hs[0] if len(hs) == 1 else (self.proj(torch.cat(hs, -1)) if self.reduce == 'cat' else torch.stack(hs, 0).mean(0))
        if self.film is not None:
            h = self.film(h, temb)
        B, N, C = h.shape
        return self.dec(h.transpose(1, 2).reshape(B, C, *grid))


class ReferringModel(nn.Module):
    def __init__(self, variant, specs, device='cuda'):
        super().__init__()
        need = max(b for bl, _ in specs.values() for b in bl)
        base = TapSeg(variant, {f'b{need}': ((need,), 'cat')}, 2, frozen=True)
        self.vit, self.win, self.patch, self.grid = base.vit, base.win, base.patch, base.grid
        self.nreg, self.dim = base.nreg, base.dim
        self.tapped = sorted({b for bl, _ in specs.values() for b in bl})
        self.heads = nn.ModuleDict({k: TextHead(bl, self.dim, self.patch, 2, red, use_text=k.startswith('txt_'))
                                    for k, (bl, red) in specs.items()})

    @torch.no_grad()
    def encode(self, x):
        self.vit.eval()
        h = self.vit._prepare_tokens_with_masks(x)
        feats = {}
        if 0 in self.tapped:
            feats[0] = h[:, self.nreg:]
        for i, blk in enumerate(self.vit.blocks, 1):
            h = blk(h)
            if i in self.tapped:
                feats[i] = h[:, self.nreg:]
            if i >= max(self.tapped):
                break
        return {k: v.detach() for k, v in feats.items()}

    def forward(self, x, temb):
        feats = self.encode(x)
        return {k: h(feats, self.grid, temb) for k, h in self.heads.items()}


class ReferringCrops(Dataset):
    def __init__(self, root, stems, bank, length=10**7, seed=0, jitter=(4, 64, 64), p_para=0.0, p_hardneg=0.0):
        self.root, self.bank, self.length, self.seed, self.jitter = root, bank, length, seed, jitter
        self.roots = root if isinstance(root, dict) else {s: root for s in stems}   # stem -> cache dir
        self.p_para, self.p_hardneg = p_para, p_hardneg
        self.items = []   # (stem, inst_id, fid)
        self.meta = {}
        for s in stems:
            m = load_meta(self.roots[s], s); self.meta[s] = m
            for ins in m['instances']:
                fid = str(ins['id'] - 1)   # inst id = finding index + 1
                if bank.get(s, fid) is not None:
                    self.items.append((s, ins['id'], fid))
        log(f'referring items: {len(self.items)} findings from {len(stems)} volumes')

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1000003 + i + os.getpid() * 7919)
        s, iid, fid = self.items[rng.randrange(len(self.items))]
        m = self.meta[s]; shape = m['shape']; w = WIN
        ins = next(x for x in m['instances'] if x['id'] == iid)
        mode = rng.random()
        if mode < 0.8:   # centred (matched 0.6 / mismatched 0.2 share this geometry)
            c = rng.choice(ins['fg_samples'])
            c = [c[k] + rng.randint(-self.jitter[k], self.jitter[k]) for k in range(3)]
        else:            # random in-body
            b = m.get('body') or [[0, 0, 0], [d - 1 for d in shape]]
            c = [rng.randint(b[0][k], max(b[1][k], b[0][k])) for k in range(3)]
        lo = [min(max(c[k] - w[k] // 2, 0), max(shape[k] - w[k], 0)) for k in range(3)]
        ct, seg, inst = load_volume(self.roots[s], s)
        sl = tuple(slice(lo[k], lo[k] + w[k]) for k in range(3))
        x = np.array(ct[sl]); y = (np.array(inst[sl]) == iid).astype(np.int64)
        del ct, seg, inst
        pad = [(0, w[k] - x.shape[k]) for k in range(3)]
        if any(p[1] > 0 for p in pad):
            x = np.pad(x, pad, constant_values=-1024); y = np.pad(y, pad)
        if mode >= 0.6 and mode < 0.8:   # mismatched text -> empty target
            temb = self.bank.get_neg(s, fid, rng) if (self.p_hardneg > 0 and rng.random() < self.p_hardneg) else None
            if temb is None:                 # random other finding's text (any scan but this one)
                s2, i2, f2 = self.items[rng.randrange(len(self.items))]
                while s2 == s:
                    s2, i2, f2 = self.items[rng.randrange(len(self.items))]
                temb = self.bank.get(s2, f2)
            y = np.zeros_like(y)
        else:
            temb = self.bank.get_pos(s, fid, rng, self.p_para)
        x = normalise(x); y = torch.from_numpy(y)
        if rng.random() < 0.5:
            x = x.flip(-1); y = y.flip(-1)
        return x[None], y, temb


@torch.no_grad()
def eval_referring(model, root, stems, bank, dev, shuffle_seed=1234, max_findings=0, tag='', out=None):
    """Per-finding Dice within a dilated-bbox ROI, matched and shuffled text per txt head."""
    model.eval()
    g = gaussian_weight(WIN).to(dev)
    all_keys = list(bank.idx)
    rng = np.random.default_rng(shuffle_seed)
    recs = []
    t0 = time.time(); nf = 0
    for s in stems:
        m = load_meta(root, s)
        insts = [i for i in m['instances'] if bank.get(s, str(i['id'] - 1)) is not None]
        if not insts:
            continue
        ct, seg, inst = load_volume(root, s)
        ct = np.array(ct); inst = np.array(inst)
        x = normalise(ct).to(dev)
        for ins in insts:
            fid = str(ins['id'] - 1)
            lo = [max(0, ins['lo'][k] - (8, 48, 48)[k]) for k in range(3)]
            hi = [min(ct.shape[k], ins['hi'][k] + 1 + (8, 48, 48)[k]) for k in range(3)]
            roi = tuple(slice(lo[k], hi[k]) for k in range(3))
            xr = x[roi]
            pad = [max(0, WIN[k] - xr.shape[k]) for k in range(3)]
            if any(pad):
                xr = F.pad(xr, (0, pad[2], 0, pad[1], 0, pad[0]), value=float(x.min()))
            wins = sliding_windows(xr.shape, WIN, (6, 112, 112))
            t_match = bank.get(s, fid).to(dev)[None]
            k2 = all_keys[rng.integers(len(all_keys))]
            while k2.split('|')[0] == s:
                k2 = all_keys[rng.integers(len(all_keys))]
            t_shuf = bank.emb[bank.idx[k2]].to(dev)[None]
            accs = {}
            norm = torch.zeros(xr.shape, device=dev)
            for i0 in range(0, len(wins), 12):
                ws = wins[i0:i0 + 12]
                xb = torch.stack([xr[a:a + WIN[0], b:b + WIN[1], c:c + WIN[2]] for a, b, c in ws])[:, None]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    feats = model.encode(xb)
                    for hk, head in model.heads.items():
                        arms = (('m', t_match), ('s', t_shuf)) if head.use_text else (('m', t_match),)
                        for arm, tt in arms:
                            o = head(feats, model.grid, tt.expand(len(ws), -1))
                            p = torch.softmax(o.float(), 1)[:, 1] * g
                            key = f'{hk}|{arm}'
                            if key not in accs:
                                accs[key] = torch.zeros(xr.shape, device=dev)
                            for j, (a, b, c) in enumerate(ws):
                                accs[key][a:a + WIN[0], b:b + WIN[1], c:c + WIN[2]] += p[j]
                for a, b, c in ws:
                    norm[a:a + WIN[0], b:b + WIN[1], c:c + WIN[2]] += g
            tgt = torch.from_numpy((inst[roi] == ins['id']).astype(np.uint8)).to(dev)
            r = dict(stem=s, fid=fid, cls=ins['cls'], vol_mm3=ins['n_vox'] * VOX_MM3, dice={})
            crop = tuple(slice(0, tgt.shape[k]) for k in range(3))
            for key, acc in accs.items():
                pr = (acc / norm)[crop] > 0.5
                den = int(pr.sum()) + int(tgt.sum())
                r['dice'][key] = float(2 * (pr & tgt.bool()).sum()) / den if den else None
            recs.append(r); nf += 1
        if nf and nf % 50 < len(insts):
            log(f'  eval {tag} {nf} findings {(time.time()-t0)/max(nf,1):.1f}s/finding')
    if out:
        json.dump(recs, open(out, 'w'))
    return recs


def summarise(recs, heads):
    outs = {}
    for hk in heads:
        for arm in ('m', 's'):
            key = f'{hk}|{arm}'
            d = [r['dice'][key] for r in recs if key in r['dice'] and r['dice'][key] is not None]
            if d:
                outs[key] = round(float(np.mean(d)), 4)
    return outs


@torch.no_grad()
def eval_steer(model, root, stems, bank, dev, shuffle_seed=1234, tag='', out=None):
    """Per-finding Dice in ROI for the steered model, matched and shuffled text per head."""
    model.eval()
    g = gaussian_weight(WIN).to(dev)
    all_keys = list(bank.idx)
    rng = np.random.default_rng(shuffle_seed)
    recs = []
    t0 = time.time(); nf = 0
    for s in stems:
        m = load_meta(root, s)
        insts = [i for i in m['instances'] if bank.get(s, str(i['id'] - 1)) is not None]
        if not insts:
            continue
        ct, seg, inst = load_volume(root, s)
        ct = np.array(ct); inst = np.array(inst)
        x = normalise(ct).to(dev)
        for ins in insts:
            fid = str(ins['id'] - 1)
            lo = [max(0, ins['lo'][k] - (8, 48, 48)[k]) for k in range(3)]
            hi = [min(ct.shape[k], ins['hi'][k] + 1 + (8, 48, 48)[k]) for k in range(3)]
            roi = tuple(slice(lo[k], hi[k]) for k in range(3))
            xr = x[roi]
            pad = [max(0, WIN[k] - xr.shape[k]) for k in range(3)]
            if any(pad):
                xr = F.pad(xr, (0, pad[2], 0, pad[1], 0, pad[0]), value=float(x.min()))
            wins = sliding_windows(xr.shape, WIN, (6, 112, 112))
            tm, mm = bank.get(s, fid)
            k2 = all_keys[rng.integers(len(all_keys))]
            while k2.split('|')[0] == s:
                k2 = all_keys[rng.integers(len(all_keys))]
            ts_, ms_ = bank.emb[bank.idx[k2]].float(), bank.mask[bank.idx[k2]]
            arms = {'m': (tm.to(dev), mm.to(dev)), 's': (ts_.to(dev), ms_.to(dev))}
            accs = {}
            norm = torch.zeros(xr.shape, device=dev)
            for i0 in range(0, len(wins), 12):
                ws = wins[i0:i0 + 12]
                xb = torch.stack([xr[a:a + WIN[0], b:b + WIN[1], c:c + WIN[2]] for a, b, c in ws])[:, None]
                for arm, (tt, tmask) in arms.items():
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        outs = model(xb, tt[None].expand(len(ws), -1, -1), tmask[None].expand(len(ws), -1))
                    for hk, o in outs.items():
                        p = torch.softmax(o.float(), 1)[:, 1] * g
                        key = f'{hk}|{arm}'
                        if key not in accs:
                            accs[key] = torch.zeros(xr.shape, device=dev)
                        for j, (a, b, c) in enumerate(ws):
                            accs[key][a:a + WIN[0], b:b + WIN[1], c:c + WIN[2]] += p[j]
                for a, b, c in ws:
                    norm[a:a + WIN[0], b:b + WIN[1], c:c + WIN[2]] += g
            tgt = torch.from_numpy((inst[roi] == ins['id']).astype(np.uint8)).to(dev)
            crop = tuple(slice(0, tgt.shape[k]) for k in range(3))
            r = dict(stem=s, fid=fid, cls=ins['cls'], vol_mm3=ins['n_vox'] * VOX_MM3, dice={})
            for key, acc in accs.items():
                pr = (acc / norm)[crop] > 0.5
                den = int(pr.sum()) + int(tgt.sum())
                r['dice'][key] = float(2 * (pr & tgt.bool()).sum()) / den if den else None
            recs.append(r); nf += 1
        if nf and nf % 50 < len(insts):
            log(f'  eval {tag} {nf} findings {(time.time()-t0)/max(nf,1):.1f}s/finding')
    if out:
        json.dump(recs, open(out, 'w'))
    return recs


def main_steer(args):
    torch.manual_seed(args.seed)
    dev = 'cuda'
    bank = TokenTextBank()
    tr, va = split_stems(args.root, 'rexground')
    json.dump(dict(train=tr, val=va), open(os.path.join(args.out, 'split.json'), 'w'))
    specs = {tok: parse_heads(tok[3:] if tok.startswith('st_') else tok)[tok[3:] if tok.startswith('st_') else tok] for tok in args.heads.split(',')}
    model = SteerReferringModel(args.variant, specs).to(dev)
    n_ad = sum(p.numel() for p in model.adapters.parameters())
    log(f'steer heads {list(specs)}; adapters after blocks {list(model.adapters)}; adapter params {n_ad/1e6:.2f}M')
    from primus.train import poly_lr
    opts = {k: torch.optim.AdamW(h.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98), fused=True)
            for k, h in model.heads.items()}
    decay, no_decay = [], []
    for n, p in model.adapters.named_parameters():
        (no_decay if (p.ndim < 2 or 'alpha' in n) else decay).append(p)   # never decay the gates
    opts['_adapters'] = torch.optim.AdamW([dict(params=decay, weight_decay=args.wd),
                                           dict(params=no_decay, weight_decay=0.0)],
                                          lr=args.lr, betas=(0.9, 0.98), fused=True)
    loss_fn = make_loss('dc_ce')
    ds = ReferringCrops(args.root, tr, bank, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, pin_memory=True,
                    persistent_workers=True, prefetch_factor=4)
    start = 0
    ck = os.path.join(args.out, 'last.pt')
    if args.resume and os.path.exists(ck):
        sd = torch.load(ck, map_location='cpu', weights_only=False)
        model.heads.load_state_dict(sd['heads']); model.adapters.load_state_dict(sd['adapters'])
        for k, o in opts.items():
            o.load_state_dict(sd['opts'][k])
        start = sd['it'] + 1; log(f'resumed at it {start}')
    run = {k: 0.0 for k in specs}; n_run = 0; it = start
    for x, y, (txt, tmask) in dl:
        if it >= args.iters:
            break
        lr = poly_lr(args.lr, it, args.iters, args.warmup)
        for o in opts.values():
            for gp in o.param_groups:
                gp['lr'] = lr
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        txt, tmask = txt.to(dev, non_blocking=True).float(), tmask.to(dev, non_blocking=True)
        model.train(); model.vit.eval()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outs = model(x, txt, tmask)
        losses = {k: loss_fn(v, y) for k, v in outs.items()}
        for o in opts.values():
            o.zero_grad(set_to_none=True)
        sum(losses.values()).backward()
        for k in specs:
            torch.nn.utils.clip_grad_norm_(model.heads[k].parameters(), 1.0)
            opts[k].step()
        torch.nn.utils.clip_grad_norm_(model.adapters.parameters(), 1.0)
        opts['_adapters'].step()
        for k, l in losses.items():
            run[k] += float(l.detach())
        n_run += 1
        if (it + 1) % 100 == 0:
            log(f'it {it+1} lr {lr:.2e} | ' + ' '.join(f'{k}={v/n_run:.3f}' for k, v in run.items())
                + ' | gates ' + json.dumps(model.gates()))
            run = {k: 0.0 for k in specs}; n_run = 0
        if (it + 1) % args.val_every == 0 or (it + 1) == args.iters:
            torch.save(dict(it=it, heads=model.heads.state_dict(), adapters=model.adapters.state_dict(),
                            opts={k: o.state_dict() for k, o in opts.items()}, args=vars(args), specs=specs,
                            gates=model.gates()), ck)
            sub = va if (it + 1) == args.iters else va[:args.val_subset]
            recs = eval_steer(model, args.root, sub, bank, dev, tag=f'it{it+1}',
                              out=os.path.join(args.out, f'eval_it{it+1}.json') if (it + 1) == args.iters else None)
            log(f'VAL it {it+1} n_findings={len(recs)} Dice: {summarise(recs, specs)} gates {json.dumps(model.gates())}')
        it += 1
    log('done')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--variant', default='tap-ct-b-3d')
    ap.add_argument('--heads', default='txt_b4,txt_b6,txt_b9,txt_b12,txt_cat_6_9_12,notxt_b9,notxt_cat_6_9_12')
    ap.add_argument('--iters', type=int, default=25000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=5e-2)
    ap.add_argument('--warmup', type=int, default=250)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--val-every', type=int, default=5000)
    ap.add_argument('--val-subset', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--steer', action='store_true', help='SteerViT mode: adapters in the encoder, plain heads')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log('args', vars(args)); log('host', os.uname().nodename)
    if args.steer:
        return main_steer(args)
    torch.manual_seed(args.seed)
    dev = 'cuda'
    bank = TextBank()
    tr, va = split_stems(args.root, 'rexground')
    json.dump(dict(train=tr, val=va), open(os.path.join(args.out, 'split.json'), 'w'))
    specs = {}
    for tok in args.heads.split(','):
        base = tok[6:] if tok.startswith('notxt_') else (tok[4:] if tok.startswith('txt_') else tok)
        specs[tok] = parse_heads(base)[base]
    model = ReferringModel(args.variant, specs).to(dev)
    log(f'heads {list(specs)}; tapped {model.tapped}')
    from primus.train import poly_lr
    opts = {k: torch.optim.AdamW(h.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98), fused=True)
            for k, h in model.heads.items()}
    loss_fn = make_loss('dc_ce')
    ds = ReferringCrops(args.root, tr, bank, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    start = 0
    ck = os.path.join(args.out, 'last.pt')
    if args.resume and os.path.exists(ck):
        sd = torch.load(ck, map_location='cpu', weights_only=False)
        model.heads.load_state_dict(sd['heads'])
        for k, o in opts.items():
            o.load_state_dict(sd['opts'][k])
        start = sd['it'] + 1; log(f'resumed at it {start}')
    run = {k: 0.0 for k in specs}; n_run = 0; it = start
    for x, y, temb in dl:
        if it >= args.iters:
            break
        lr = poly_lr(args.lr, it, args.iters, args.warmup)
        for o in opts.values():
            for gp in o.param_groups:
                gp['lr'] = lr
        x, y, temb = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), temb.to(dev, non_blocking=True)
        model.train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outs = model(x, temb)
        losses = {k: loss_fn(v, y) for k, v in outs.items()}
        for o in opts.values():
            o.zero_grad(set_to_none=True)
        sum(losses.values()).backward()
        for k, o in opts.items():
            torch.nn.utils.clip_grad_norm_(model.heads[k].parameters(), 1.0)
            o.step()
        for k, l in losses.items():
            run[k] += float(l.detach())
        n_run += 1
        if (it + 1) % 100 == 0:
            log(f'it {it+1} lr {lr:.2e} | ' + ' '.join(f'{k}={v/n_run:.3f}' for k, v in run.items()))
            run = {k: 0.0 for k in specs}; n_run = 0
        if (it + 1) % args.val_every == 0 or (it + 1) == args.iters:
            torch.save(dict(it=it, heads=model.heads.state_dict(), opts={k: o.state_dict() for k, o in opts.items()},
                            args=vars(args), specs=specs), ck)
            sub = va if (it + 1) == args.iters else va[:args.val_subset]
            recs = eval_referring(model, args.root, sub, bank, dev, tag=f'it{it+1}',
                                  out=os.path.join(args.out, f'eval_it{it+1}.json') if (it + 1) == args.iters else None)
            log(f'VAL it {it+1} n_findings={len(recs)} mean per-finding Dice: {summarise(recs, specs)}')
        it += 1
    log('done')


if __name__ == '__main__':
    main()

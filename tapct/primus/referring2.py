"""Referring segmentation, second generation: fine-tunable encoder x {text routes} x {conv-stem skips}.

Motivated by the §I/§J finding: the conv stem DETECTS small objects (recall 0.50-0.55) but cannot DISCRIMINATE
(Dice 0.25), while the ViT carries semantics. In referring segmentation the missing discriminator is exactly the
text. So text can be injected at three sites and the sites can be ablated in one run:
  * `tok`   - FiLM on the ViT token stream (semantic path)          -> what the frozen §G arm did
  * `skip`  - FiLM on the conv-stem skip features (detail path)     -> text selects among detected blobs
  * `both`  - both sites
  * `steer` - gated cross-attention INSIDE the encoder (SteerViT), optionally with any of the above
Encoder can be frozen (heads only, all heads share one forward) or fine-tuned (`--finetune`, single head advised).

Head key grammar: `<route>_<blockspec>`, route in {txt, notxt, v3tok, v3skip, v3both, v3notxt}, blockspec as in
primus.model.parse_heads (b9, cat_6_9_12, mean_9_12). SteerViT is a per-run flag, not a route.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from primus.data import WIN, load_meta, load_volume, normalise, sliding_windows, gaussian_weight, split_stems
from primus.model import TapSeg, ConvStem, ConvStemV3, SkipDecode, parse_heads, make_loss, init_he
import random
from torch.utils.data import Dataset
from primus.referring import (GatedTextCrossAttn, FiLM, TokenTextBank, make_text_bank, ReferringCrops, log, VOX_MM3)

# tok: False | 'post' (one FiLM after fusing the tapped blocks) | 'per' (one FiLM per tapped block, BEFORE fusion)
# gate: text-conditioned softmax weights over the tapped blocks ("let the text choose the depth")
ROUTES = {
    'notxt':   dict(v3=False, tok=False,  skip=False, gate=False),
    'txt':     dict(v3=False, tok='post', skip=False, gate=False),
    'txtk':    dict(v3=False, tok='per',  skip=False, gate=False),
    'txtw':    dict(v3=False, tok='per',  skip=False, gate=True),
    'v3notxt': dict(v3=True,  tok=False,  skip=False, gate=False),
    'v3tok':   dict(v3=True,  tok='post', skip=False, gate=False),
    'v3skip':  dict(v3=True,  tok=False,  skip=True,  gate=False),
    'v3both':  dict(v3=True,  tok='post', skip=True,  gate=False),
    'v3bothk': dict(v3=True,  tok='per',  skip=True,  gate=False),
    'v3bothw': dict(v3=True,  tok='per',  skip=True,  gate=True),
}


def split_key(key):
    for r in sorted(ROUTES, key=len, reverse=True):
        if key.startswith(r + '_'):
            return r, key[len(r) + 1:]
    raise ValueError(f'unknown route in head key {key!r} (routes: {sorted(ROUTES)})')


class SkipFiLM(nn.Module):
    """Per-channel FiLM on each conv-stem skip level; zero-init -> identity at step 0."""

    def __init__(self, chans=(16, 32, 64), tdim=1024):
        super().__init__()
        self.mlps = nn.ModuleList()
        for c in chans:
            m = nn.Sequential(nn.Linear(tdim, 256), nn.GELU(), nn.Linear(256, 2 * c))
            nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)
            self.mlps.append(m)

    def forward(self, skips, t):
        out = []
        for s, m in zip(skips, self.mlps):
            g, b = m(t).chunk(2, -1)
            idx = (slice(None), slice(None), None, None, None)
            out.append(s * (1 + g[idx]) + b[idx])
        return out


class RouteHead(nn.Module):
    """One head = block taps -> [FiLM on tokens] -> decoder (plain PatchDecode, or SkipDecode fed by a conv stem
    whose skips may themselves be FiLM'd)."""

    def __init__(self, blocks, dim, patch, n_classes, reduce='cat', v3=False, tok=False, skip=False, gate=False,
                 aux=False, tdim=1024, stem='lite', ras_stem=False):
        super().__init__()
        self.blocks, self.v3, self.tok, self.gate = tuple(blocks), v3, tok, gate
        self.ras_stem = ras_stem                        # SPECTRE: tokens/decoder are RAS, so the stem sees RAS input
        k = len(self.blocks)
        self.reduce = 'weighted' if gate else reduce
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in self.blocks])
        self.proj = nn.Linear(dim * k, dim) if (k > 1 and self.reduce == 'cat') else None
        self.film = FiLM(dim, tdim) if tok == 'post' else None
        self.films = nn.ModuleList([FiLM(dim, tdim) for _ in self.blocks]) if tok == 'per' else None
        # zero-init depth logits -> uniform softmax at step 0, i.e. exactly mean fusion
        self.gate_mlp = nn.Linear(tdim, k) if gate else None
        if gate:
            nn.init.zeros_(self.gate_mlp.weight); nn.init.zeros_(self.gate_mlp.bias)
        # deep supervision: per-tapped-block patch-level detection head (is this token inside the finding?),
        # applied to the TEXT-INFUSED tokens of that block, so early blocks get a direct grounding signal.
        self.aux = nn.ModuleList([nn.Linear(dim, 1) for _ in self.blocks]) if aux else None
        self.last_w = None
        self.last_aux = None
        if v3:
            from primus.model import PatchDecode as _PD, ConvStemN, SkipDecodeN
            n_stages = len(_PD(patch, dim, n_classes).strides)
            if n_stages == 3:
                self.stem = ConvStemV3() if stem == 'v3' else ConvStem()
                self.dec = SkipDecode(patch, dim, n_classes, stem_ch=self.stem.channels)
            else:                                       # e.g. SPECTRE (16,16,8): four decoder stages
                assert stem == 'lite', 'wide stem only defined for the 3-stage decoder'
                self.stem = ConvStemN(_PD(patch, dim, n_classes).strides)
                self.dec = SkipDecodeN(patch, dim, n_classes, stem_ch=self.stem.channels)
            self.skip_film = SkipFiLM(self.stem.channels, tdim) if skip else None
            init_he(self.stem); init_he(self.dec)
        else:
            assert not skip, 'skip conditioning requires a v3 route'
            from primus.model import PatchDecode
            self.dec = PatchDecode(patch, dim, n_classes); init_he(self.dec)
            self.stem = self.skip_film = None

    def forward(self, feats, grid, x=None, temb=None):
        hs = [n(feats[b]) for n, b in zip(self.norms, self.blocks)]
        if self.films is not None:                      # per-block FiLM, BEFORE fusion
            hs = [f(h_, temb) for f, h_ in zip(self.films, hs)]
        self.last_aux = [a(h_).squeeze(-1) for a, h_ in zip(self.aux, hs)] if self.aux is not None else None
        if len(hs) == 1:
            h = hs[0]
            self.last_w = None
        elif self.reduce == 'weighted':                 # text picks the depth
            w = torch.softmax(self.gate_mlp(temb), -1)  # (B, k)
            self.last_w = w.detach()
            h = (torch.stack(hs, 1) * w[:, :, None, None]).sum(1)
        elif self.reduce == 'cat':
            h = self.proj(torch.cat(hs, -1))
        else:
            h = torch.stack(hs, 0).mean(0)
        if self.film is not None:
            h = self.film(h, temb)
        B, N, C = h.shape
        h = h.transpose(1, 2).reshape(B, C, *grid)
        if self.stem is None:
            return self.dec(h)
        if self.ras_stem:
            from primus.model import TapSeg
            x = TapSeg.lps_to_ras(x)
        sk = self.stem(x)
        if self.skip_film is not None:
            sk = self.skip_film(sk, temb)
        return self.dec(h, sk)


class RefNet(nn.Module):
    def __init__(self, variant, specs, frozen=True, steer=False, aux=False, tdim=1024, stem='lite'):
        super().__init__()
        need = max(b for bl, _ in specs.values() for b in bl)
        base = TapSeg(variant, {f'b{need}': ((need,), 'cat')}, 2, frozen=frozen)
        self.vit, self.patch = base.vit, base.patch
        self.is_spectre = base.is_spectre
        from primus.data import WIN as _WIN                      # actual window may be deeper than the variant's
        self.win = _WIN
        self.grid = tuple(w // p for w, p in zip(self.win, self.patch))          # (z, y, x)
        self.grid_tok = self.grid[::-1] if self.is_spectre else self.grid       # token layout (SPECTRE: x, y, z)
        self.patch_tok = base.patch_tok
        self.nreg, self.dim, self.frozen = base.nreg, base.dim, frozen
        if self.is_spectre:
            assert not steer and stem == 'lite', 'SPECTRE referring arm: no SteerViT, lite stem only'
        self.tapped = sorted({b for bl, _ in specs.values() for b in bl})
        self.adapters = nn.ModuleDict({str(i): GatedTextCrossAttn(self.dim, tdim)
                                       for i in range(2, need + 1, 2)}) if steer else nn.ModuleDict()
        self.heads = nn.ModuleDict()
        for k, (bl, red) in specs.items():
            r, _ = split_key(k)
            self.heads[k] = RouteHead(bl, self.dim, self.patch_tok, 2, red, aux=aux, tdim=tdim, stem=stem,
                                      ras_stem=self.is_spectre, **ROUTES[r])

    @staticmethod
    def pool(txt, mask):
        """Qwen3-Embedding last-token pooling from right-padded token states."""
        last = mask.sum(1).clamp(min=1) - 1
        return F.normalize(txt[torch.arange(txt.shape[0], device=txt.device), last], dim=-1)

    def _encode(self, x, txt, mask):
        rope = None
        if self.is_spectre:
            h = self.vit.patch_embed(TapSeg.lps_to_ras(x))
            h, rope = self.vit._pos_embed(h)
            h = self.vit.norm_pre(h)
        else:
            h = self.vit._prepare_tokens_with_masks(x)
        feats = {}
        if 0 in self.tapped:
            feats[0] = h[:, self.nreg:]
        for i, blk in enumerate(self.vit.blocks, 1):
            h = blk(h, rope=rope) if self.is_spectre else blk(h)
            if str(i) in self.adapters:
                pt = self.adapters[str(i)](h[:, self.nreg:], txt, mask)
                h = torch.cat([h[:, :self.nreg], pt], 1)
            if i in self.tapped:
                feats[i] = h[:, self.nreg:]
            if i >= max(self.tapped):
                break
        return feats

    def encode(self, x, txt, mask):
        if self.frozen and not self.adapters:          # nothing upstream needs grads
            with torch.no_grad():
                self.vit.eval()
                return {k: v.detach() for k, v in self._encode(x, txt, mask).items()}
        if self.frozen:
            self.vit.eval()                            # frozen weights, but adapters need the graph
        return self._encode(x, txt, mask)

    def forward(self, x, txt, mask, head_names=None):
        feats = self.encode(x, txt, mask)
        t = self.pool(txt, mask)
        return {k: self.run_head(k, feats, x, t) for k in (head_names or list(self.heads))}

    def run_head(self, k, feats, x, t):
        """One head on already-encoded features, returned in the cache's (z,y,x) LPS layout. The evaluators must
        use this rather than calling heads[k] directly: for SPECTRE the tokens (and the decoder output) are RAS."""
        o = self.heads[k](feats, self.grid_tok, x, t)
        return TapSeg.ras_to_lps(o) if self.is_spectre else o

    def gates(self):
        return {k: round(a.gate(), 4) for k, a in self.adapters.items()}


class ReferringCropsNN(Dataset):
    """nnU-Net-style probabilistic sampling for referring crops.

    nnU-Net (`nnUNetDataLoaderBase.get_bbox`) forces `oversample_foreground_percent` (default 0.33) of each batch
    onto a random foreground voxel and draws the REST UNIFORMLY over the whole volume — so ~2/3 of crops are
    ordinary, mostly-background field of view. Our legacy sampler was nearly the inverse (60 % finding-centred,
    20 % in-body random), which is the diagnosed cause of scan-level over-prediction (§O): the model is almost
    never shown distant background paired with the CORRECT text and an empty target.

    Draw per item (i.i.d. rather than nnU-Net's per-batch quota — same in expectation):
        u < mismatch_pct                    : foreground-centred crop, WRONG text, empty target  (text selectivity)
        u < mismatch_pct + fg_pct           : foreground-centred crop, matched text              (nnU-Net force_fg)
        else                                : uniform random crop anywhere in the volume, matched text
                                              (target = whatever finding voxels fall inside — usually none)
    `body_pct` optionally routes part of the uniform share through the body bbox so crops are not all air.
    """

    def __init__(self, root, stems, bank, length=10 ** 7, seed=0, jitter=(4, 64, 64),
                 fg_pct=0.33, mismatch_pct=0.2, body_pct=0.5):
        self.root, self.bank, self.length, self.seed, self.jitter = root, bank, length, seed, jitter
        self.fg_pct, self.mismatch_pct, self.body_pct = fg_pct, mismatch_pct, body_pct
        self.items, self.meta = [], {}
        for s in stems:
            m = load_meta(root, s); self.meta[s] = m
            for ins in m['instances']:
                fid = str(ins['id'] - 1)
                if bank.get(s, fid) is not None:
                    self.items.append((s, ins['id'], fid))
        log(f'nnU-Net-style sampler: {len(self.items)} findings / {len(stems)} volumes; '
            f'fg {fg_pct:.2f} mismatch {mismatch_pct:.2f} uniform {1 - fg_pct - mismatch_pct:.2f} '
            f'(of which {body_pct:.0%} inside the body bbox)')

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1000003 + i + os.getpid() * 7919)
        s, iid, fid = self.items[rng.randrange(len(self.items))]
        m = self.meta[s]; shape = m['shape']; w = WIN
        ins = next(x for x in m['instances'] if x['id'] == iid)
        u = rng.random()
        force_fg = u < self.mismatch_pct + self.fg_pct
        if force_fg:
            c = rng.choice(ins['fg_samples'])
            c = [c[k] + rng.randint(-self.jitter[k], self.jitter[k]) for k in range(3)]
            lo = [min(max(c[k] - w[k] // 2, 0), max(shape[k] - w[k], 0)) for k in range(3)]
        else:                                        # uniform over the volume (nnU-Net's non-forced branch)
            if rng.random() < self.body_pct and m.get('body'):
                b = m['body']
                lo = [min(max(rng.randint(b[0][k], max(b[1][k], b[0][k])) - w[k] // 2, 0),
                          max(shape[k] - w[k], 0)) for k in range(3)]
            else:
                lo = [rng.randint(0, max(shape[k] - w[k], 0)) for k in range(3)]
        ct, seg, inst = load_volume(self.root, s)
        sl = tuple(slice(lo[k], lo[k] + w[k]) for k in range(3))
        x = np.array(ct[sl]); y = (np.array(inst[sl]) == iid).astype(np.int64)
        del ct, seg, inst
        pad = [(0, w[k] - x.shape[k]) for k in range(3)]
        if any(p[1] > 0 for p in pad):
            x = np.pad(x, pad, constant_values=-1024); y = np.pad(y, pad)
        if u < self.mismatch_pct:                    # wrong text -> nothing to segment
            s2, i2, f2 = self.items[rng.randrange(len(self.items))]
            while s2 == s:
                s2, i2, f2 = self.items[rng.randrange(len(self.items))]
            temb = self.bank.get(s2, f2); y = np.zeros_like(y)
        else:
            temb = self.bank.get(s, fid)
        x = normalise(x); y = torch.from_numpy(y)
        if rng.random() < 0.5:
            x = x.flip(-1); y = y.flip(-1)
        return x[None], y, temb


@torch.no_grad()
def evaluate(model, root, stems, bank, dev, shuffle_seed=1234, tag='', out=None):
    """Per-finding Dice in a dilated ROI, matched vs shuffled text, for every head."""
    model.eval()
    g = gaussian_weight(WIN).to(dev)
    keys = list(bank.idx)
    rng = np.random.default_rng(shuffle_seed)
    recs = []
    t0 = time.time(); nf = 0
    for s in stems:
        meta = load_meta(root, s)
        insts = [i for i in meta['instances'] if bank.get(s, str(i['id'] - 1)) is not None]
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
            wins = sliding_windows(xr.shape, WIN, (max(1, WIN[0] // 2), 112, 112))
            tm, mm = bank.get(s, fid)
            k2 = keys[rng.integers(len(keys))]
            while k2.split('|')[0] == s:
                k2 = keys[rng.integers(len(keys))]
            arms = {'m': (tm.to(dev), mm.to(dev)),
                    's': (bank.emb[bank.idx[k2]].float().to(dev), bank.mask[bank.idx[k2]].to(dev))}
            accs = {}; norm = torch.zeros(xr.shape, device=dev)
            ebatch = max(1, (8 * 12) // WIN[0])      # keep eval memory ~constant as the window deepens
            for i0 in range(0, len(wins), ebatch):
                ws = wins[i0:i0 + ebatch]
                xb = torch.stack([xr[a:a + WIN[0], b:b + WIN[1], c:c + WIN[2]] for a, b, c in ws])[:, None]
                for arm, (tt, tmk) in arms.items():
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        outs = model(xb, tt[None].expand(len(ws), -1, -1), tmk[None].expand(len(ws), -1))
                    for hk, o in outs.items():
                        p = torch.softmax(o.float(), 1)[:, 1] * g
                        key = f'{hk}|{arm}'
                        accs.setdefault(key, torch.zeros(xr.shape, device=dev))
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


def patch_targets(y, patch):
    """(B,D,H,W) int64 mask -> (B,N) float: 1 where the token's patch contains any positive voxel."""
    return F.max_pool3d(y[:, None].float(), kernel_size=patch, stride=patch).flatten(1)


def aux_loss(head, tgt):
    """Mean BCE over tapped blocks; pos_weight from the batch (patches are ~99 % negative)."""
    pos = tgt.sum().clamp(min=1.0)
    pw = ((tgt.numel() - pos) / pos).clamp(1.0, 50.0)
    ls = [F.binary_cross_entropy_with_logits(l.float(), tgt, pos_weight=pw) for l in head.last_aux]
    return sum(ls) / len(ls), [float(l.detach()) for l in ls]


def summarise(recs, heads):
    o = {}
    for h in heads:
        for arm in ('m', 's'):
            k = f'{h}|{arm}'
            d = [r['dice'][k] for r in recs if k in r['dice'] and r['dice'][k] is not None]
            if d:
                o[k] = round(float(np.mean(d)), 4)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--variant', default='tap-ct-b-3d')
    ap.add_argument('--text-bank', default='tokens', help="'tokens' (Qwen3 token states) or a pooled npz path, e.g. SPECTRE's")
    ap.add_argument('--heads', required=True, help='e.g. v3tok_b6,v3skip_b6,v3both_b6,v3notxt_b6,txt_b6')
    ap.add_argument('--steer', action='store_true')
    ap.add_argument('--finetune', action='store_true')
    ap.add_argument('--init-heads', default=None)
    ap.add_argument('--init-from', default=None, help='checkpoint (last.pt) whose heads/adapters/encoder weights start this run; fresh optimisers and schedule (post-training)')
    ap.add_argument('--finetune-blocks', type=int, default=0, help='with --finetune: unfreeze only the top-k encoder blocks (0 = all)')
    ap.add_argument('--iters', type=int, default=25000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--accum', type=int, default=1,
                    help='micro-batches per optimiser step; effective batch = batch * accum. '
                         'Needed for deep windows, where batch 8 does not fit (see reports section 8).')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--enc-lr', type=float, default=2e-5)
    ap.add_argument('--wd', type=float, default=5e-2)
    ap.add_argument('--warmup', type=int, default=250)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--val-every', type=int, default=5000)
    ap.add_argument('--val-subset', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--stem', default='lite', choices=['lite', 'v3'],
                    help='conv-stem capacity for v3* routes: lite 0.9M, v3 ~5M (PrimusV3-scale)')
    ap.add_argument('--loss', default='dc_ce', choices=['dc_ce', 'batchdc_ce', 'dc_ce_fg', 'batchdc_ce_fg'],
                    help="batch-dice matters with --sampler nnunet: per-sample Dice is inert on empty crops "
                         "(measured grad 3e-18 vs 1e-10), though CE still supervises them")
    ap.add_argument('--sampler', default='legacy', choices=['legacy', 'nnunet'])
    ap.add_argument('--fg-pct', type=float, default=0.33, help='nnunet sampler: forced-foreground share')
    ap.add_argument('--mismatch-pct', type=float, default=0.2)
    ap.add_argument('--body-pct', type=float, default=0.5)
    ap.add_argument('--aux-w', type=float, default=0.0, help='weight of the per-block patch-detection deep supervision')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--train-on-val', action='store_true', help='fold the validation split into training (final run)')
    ap.add_argument('--p-para', type=float, default=0.0, help='prob. of a paraphrase/template variant for matched text')
    ap.add_argument('--p-hardneg', type=float, default=0.0, help='prob. of a swapped-location hard negative in the mismatch arm')
    ap.add_argument('--ddp', action='store_true', help='data-parallel over the GPUs of one node (launch with torchrun); effective batch = batch * accum * world')
    args = ap.parse_args()
    global log
    rank, world = 0, 1
    if args.ddp:
        import torch.distributed as dist
        dist.init_process_group('nccl')
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
        if rank != 0:
            log = lambda *a, **k: None                       # rank 0 speaks for the job
    os.makedirs(args.out, exist_ok=True)
    log('args', vars(args)); log('host', os.uname().nodename, f'| ddp world {world}, effective batch {args.batch * args.accum * world}')
    for _r in args.root.split(','):
        if not os.path.isdir(_r) or not any(n.endswith('_meta.json') for n in os.listdir(_r)[:50]):
            raise SystemExit(f'cache not visible on {os.uname().nodename}: {_r} — node-local /processing copies '
                             f'exist on herakles/euctemon/eudoxus only; pin with --nodelist or rsync the cache first.')
    torch.manual_seed(args.seed + rank)
    dev = f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}' if args.ddp else 'cuda'
    bank = make_text_bank(args.text_bank); log('text bank', args.text_bank)
    roots = args.root.split(',')
    tr, va, root_of = [], [], {}
    for r in roots:
        t_, v_ = split_stems(r, 'rexground')
        tr += t_; va += v_ if r == roots[0] else []          # validation stems come from the first root only
        for s_ in t_ + v_:
            root_of[s_] = r
    if args.train_on_val:
        tr = tr + va                                          # final-run mode: the held-out split joins training
        log(f'TRAIN-ON-VAL: {len(va)} validation volumes folded into training (quick val is now in-sample)')
    log(f'roots {roots}: train {len(tr)} / val {len(va)} volumes')
    json.dump(dict(train=tr, val=va, roots=roots, train_on_val=bool(args.train_on_val)),
              open(os.path.join(args.out, 'split.json'), 'w'))
    args.root = roots[0]
    specs = {}
    for k in args.heads.split(','):
        _, bs = split_key(k)
        specs[k] = parse_heads(bs)[bs]
    if args.finetune and len(specs) > 1:
        log('WARNING: fine-tuning with >1 head — heads share (and jointly shape) the encoder, so per-head '
            'comparisons are NOT clean; only the best head is interpretable.')
    model = RefNet(args.variant, specs, frozen=not args.finetune, steer=args.steer, aux=args.aux_w > 0,
                   stem=args.stem).to(dev)
    if args.finetune or os.environ.get('PRIMUS_SDPA') == '1':
        from primus.model import enable_sdpa
        n_p, d_p = enable_sdpa(model.vit)
        log(f'SDPA attention: patched {n_p} modules, max |naive - sdpa| on a probe input {d_p:.2e}')
        if n_p == 0 or d_p > 1e-3:
            raise SystemExit('SDPA patch did not apply cleanly; refusing to train on a different model')
    if args.finetune and args.finetune_blocks > 0:
        # Partial unfreezing: train only the top-k transformer blocks (+ nothing else in the encoder). The full
        # fine-tune overfits (4 Sep: quick-val decays after 5k steps); the early blocks carry generic CT features.
        import re as _re
        ids = sorted({int(m.group(1)) for n, _ in model.vit.named_parameters() for m in [_re.search(r'blocks\.(\d+)\.', n)] if m})
        keep = set(ids[-args.finetune_blocks:])
        n_before = sum(p.numel() for p in model.vit.parameters() if p.requires_grad)
        for n, p in model.vit.named_parameters():
            m = _re.search(r'blocks\.(\d+)\.', n)
            if not (m and int(m.group(1)) in keep):
                p.requires_grad_(False)
        n_after = sum(p.numel() for p in model.vit.parameters() if p.requires_grad)
        log(f'finetune-blocks {args.finetune_blocks}: blocks {sorted(keep)} of {len(ids)} trainable; encoder params {n_before/1e6:.1f}M -> {n_after/1e6:.1f}M')
    n_h = sum(p.numel() for p in model.heads.parameters())
    n_a = sum(p.numel() for p in model.adapters.parameters())
    n_e = sum(p.numel() for p in model.vit.parameters() if p.requires_grad)
    log(f'text bank: {len(bank.idx)} base keys, {getattr(bank, "n_pos", 0)} positive variants, {getattr(bank, "n_neg", 0)} hard negatives; p_para={args.p_para} p_hardneg={args.p_hardneg}')
    log(f'heads {list(specs)} | steer={args.steer} finetune={args.finetune} | params heads {n_h/1e6:.2f}M '
        f'adapters {n_a/1e6:.2f}M encoder(trainable) {n_e/1e6:.2f}M | adapters at {list(model.adapters)}')
    if args.init_heads:
        sd = torch.load(args.init_heads, map_location='cpu', weights_only=False)['heads']
        for k in specs:
            sub = {kk.split('.', 1)[1]: v for kk, v in sd.items() if kk.split('.', 1)[0] == k}
            if sub:
                miss, unexp = model.heads[k].load_state_dict(sub, strict=False)
                log(f'init head {k} from {args.init_heads}: {len(sub)} tensors, missing {len(miss)}')
            else:
                log(f'init head {k}: NO matching tensors in {args.init_heads} (fresh init)')
    if args.init_from:
        sd = torch.load(args.init_from, map_location='cpu', weights_only=False)
        model.heads.load_state_dict(sd['heads'])
        if len(model.adapters):
            model.adapters.load_state_dict(sd['adapters'])
        if 'vit' in sd:
            model.vit.load_state_dict(sd['vit'])
        log(f"init model from {args.init_from} (it {sd['it']}, encoder {'loaded' if 'vit' in sd else 'NOT in checkpoint'}); fresh optimisers")
    from primus.train import poly_lr
    opts = {k: torch.optim.AdamW(h.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98), fused=True)
            for k, h in model.heads.items()}
    if len(model.adapters):
        dec = [p for n, p in model.adapters.named_parameters() if p.ndim >= 2 and 'alpha' not in n]
        nod = [p for n, p in model.adapters.named_parameters() if p.ndim < 2 or 'alpha' in n]
        opts['_adapters'] = torch.optim.AdamW([dict(params=dec, weight_decay=args.wd),
                                               dict(params=nod, weight_decay=0.0)], lr=args.lr, betas=(0.9, 0.98), fused=True)
    enc_opt = None
    if args.finetune:
        enc_opt = torch.optim.AdamW([p for p in model.vit.parameters() if p.requires_grad], lr=args.enc_lr,
                                    weight_decay=args.wd, betas=(0.9, 0.98), fused=True)
    loss_fn = make_loss(args.loss)
    net = model
    if args.ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        net = DDP(model, device_ids=[int(os.environ.get('LOCAL_RANK', 0))], find_unused_parameters=True)   # the encoder's final norm / mask token are not on our tap path
    from contextlib import nullcontext
    ds = (ReferringCrops(root_of, tr, bank, seed=args.seed, p_para=args.p_para, p_hardneg=args.p_hardneg) if args.sampler == 'legacy' else
          ReferringCropsNN(args.root, tr, bank, seed=args.seed, fg_pct=args.fg_pct,
                           mismatch_pct=args.mismatch_pct, body_pct=args.body_pct))
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, pin_memory=True,
                    persistent_workers=True, prefetch_factor=4)
    start = 0
    ck = os.path.join(args.out, 'last.pt')
    if args.resume and os.path.exists(ck):
        sd = torch.load(ck, map_location='cpu', weights_only=False)
        model.heads.load_state_dict(sd['heads'])
        if len(model.adapters):
            model.adapters.load_state_dict(sd['adapters'])
        if 'vit' in sd:
            model.vit.load_state_dict(sd['vit'])
        for k, o in opts.items():
            o.load_state_dict(sd['opts'][k])
        if enc_opt is not None and 'enc_opt' in sd:
            enc_opt.load_state_dict(sd['enc_opt'])
        start = sd['it'] + 1; log(f'resumed at it {start}')
    def grouped(loader, n):
        """Yield lists of n micro-batches; one list = one optimiser step (gradient accumulation)."""
        buf = []
        for b in loader:
            buf.append(b)
            if len(buf) == n:
                yield buf
                buf = []

    run = {k: 0.0 for k in specs}; aux_log = {}; n_run = 0; it = start
    for group in grouped(dl, args.accum):
        if it >= args.iters:
            break
        lr = poly_lr(args.lr, it, args.iters, args.warmup)
        for o in opts.values():
            for gp in o.param_groups:
                gp['lr'] = lr
        if enc_opt is not None:
            for gp in enc_opt.param_groups:
                gp['lr'] = poly_lr(args.enc_lr, it, args.iters, args.warmup)
        for o in opts.values():
            o.zero_grad(set_to_none=True)
        if enc_opt is not None:
            enc_opt.zero_grad(set_to_none=True)
        losses = {k: 0.0 for k in specs}
        for j, (x, y, (txt, tmask)) in enumerate(group):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            txt, tmask = txt.to(dev, non_blocking=True).float(), tmask.to(dev, non_blocking=True)
            model.train()
            if not args.finetune:
                model.vit.eval()
            sync = nullcontext() if (not args.ddp or j == len(group) - 1) else net.no_sync()   # all-reduce once per step
            with sync, torch.autocast('cuda', dtype=torch.bfloat16):
                outs = net(x, txt, tmask)
            mb = {k: loss_fn(v, y) for k, v in outs.items()}
            if args.aux_w > 0:
                tgt = patch_targets(y, model.patch)
                for k, hd in model.heads.items():
                    if hd.last_aux:
                        a, per = aux_loss(hd, tgt)
                        mb[k] = mb[k] + args.aux_w * a
                        aux_log.setdefault(k, np.zeros(len(per)))
                        aux_log[k] += np.array(per) / len(group)
            with (nullcontext() if (not args.ddp or j == len(group) - 1) else net.no_sync()):
                (sum(mb.values()) / len(group)).backward()
            for k, l in mb.items():
                losses[k] = losses[k] + float(l.detach()) / len(group)
        for k in specs:
            torch.nn.utils.clip_grad_norm_(model.heads[k].parameters(), 1.0)
            opts[k].step()
        if '_adapters' in opts:
            torch.nn.utils.clip_grad_norm_(model.adapters.parameters(), 1.0)
            opts['_adapters'].step()
        if enc_opt is not None:
            torch.nn.utils.clip_grad_norm_([p for p in model.vit.parameters() if p.requires_grad], 1.0)
            enc_opt.step()
        for k, l in losses.items():
            run[k] += l
        n_run += 1
        if (it + 1) % 100 == 0:
            msg = ' '.join(f'{k}={v/n_run:.3f}' for k, v in run.items())
            extra = ''
            if aux_log:
                extra += ' | aux/blk ' + json.dumps({k: [round(x / n_run, 3) for x in v] for k, v in aux_log.items()})
            for k, hd in model.heads.items():
                if hd.last_w is not None:
                    extra += f' | depth_w[{k}] ' + json.dumps([round(x, 3) for x in hd.last_w.float().mean(0).tolist()])
            log(f'it {it+1} lr {lr:.2e} | {msg}' + (f' | gates {json.dumps(model.gates())}' if len(model.adapters) else '') + extra)
            run = {k: 0.0 for k in specs}; aux_log = {}; n_run = 0
        if (it + 1) % args.val_every == 0 or (it + 1) == args.iters:
          if rank == 0:
            torch.save(dict(it=it, heads=model.heads.state_dict(),
                            adapters=model.adapters.state_dict() if len(model.adapters) else {},
                            opts={k: o.state_dict() for k, o in opts.items()},
                            **({'vit': model.vit.state_dict(), 'enc_opt': enc_opt.state_dict()} if enc_opt else {}),
                            args=vars(args), specs=specs), ck)
            sub = va if (it + 1) == args.iters else va[:args.val_subset]
            recs = evaluate(model, args.root, sub, bank, dev, tag=f'it{it+1}',
                            out=os.path.join(args.out, f'eval_it{it+1}.json') if (it + 1) == args.iters else None)
            log(f'VAL it {it+1} n={len(recs)} {summarise(recs, specs)}'
                + (f' gates {json.dumps(model.gates())}' if len(model.adapters) else ''))
          if args.ddp:
            torch.distributed.barrier()                       # the other ranks wait for rank 0's checkpoint + validation
        it += 1
    log('done')
    if args.ddp:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()

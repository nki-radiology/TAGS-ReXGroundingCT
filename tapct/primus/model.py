"""TAP-CT encoder + Primus-style lightweight decoders attached at chosen blocks.

Decoder = Primus `PatchDecode` (Wald et al., TMLR 2026; vendored from dynamic_network_architectures
0.4: back-to-back ConvTranspose-LayerNorm-GELU stages, one per log2(patch) level, channels shrinking
geometrically to the class count). Heads: one per tapped block (input LayerNorm + PatchDecode); multi-block
heads concatenate the tapped blocks and project back to the model width with a linear layer first.
"""
import math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lattice.models import load_model, load_spectre, load_ctvit, load_ctfm, geometry, SPECTRE_BACKBONE, SPECTRE_BACKBONE_NOVLA   # noqa: E402


def enable_sdpa(vit):
    """Route TAP-CT's vendored DINOv2 attention through torch SDPA. Without xformers the vendored module
    materialises the (B, heads, N, N) attention matrix per block, which is what stores 4 GiB per block at
    z=24 during fine-tuning. SDPA computes the identical softmax(qk^T * scale) v (scale = head_dim^-0.5 is
    SDPA's default) without materialising it. Returns (n_patched, max_abs_diff) where the difference is
    measured on a random input through one patched module, so a silent semantic mismatch cannot pass."""
    patched, diff = 0, 0.0

    def sdpa_forward(self, x, attn_bias=None):
        if attn_bias is not None:
            raise AssertionError('attn_bias is not supported by the SDPA path')
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2],
                                           dropout_p=self.attn_drop.p if self.training else 0.0)
        return self.proj_drop(self.proj(y.transpose(1, 2).reshape(B, N, C)))

    for m in vit.modules():
        if type(m).__name__ in ('Attention', 'MemEffAttention') and hasattr(m, 'qkv') and hasattr(m, 'scale'):
            if patched == 0:
                with torch.no_grad():
                    p = next(m.parameters())
                    x = torch.randn(2, 64, m.qkv.in_features, device=p.device, dtype=p.dtype)
                    ref = m(x)
                    new = sdpa_forward(m, x)
                    diff = float((ref - new).abs().max())
            m.forward = sdpa_forward.__get__(m, type(m))
            patched += 1
    return patched, diff


class LayerNormNd(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels)); self.bias = nn.Parameter(torch.zeros(num_channels)); self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True); s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        idx = (None, slice(None), *([None] * (x.ndim - 2)))
        return self.weight[idx] * x + self.bias[idx]


class PatchDecode(nn.Module):
    """Verbatim logic of dynamic_network_architectures.building_blocks.patch_encode_decode.PatchDecode."""

    def __init__(self, patch_size, embed_dim, out_channels, norm=LayerNormNd, activation=nn.GELU):
        super().__init__()
        if max(patch_size) == 1:                       # full-resolution features (CNN stem / stride-1 stage): 1x1 head
            self.decode = nn.Conv3d(embed_dim, out_channels, kernel_size=1)
            self.strides, self.channels = [], [embed_dim, out_channels]
            return

        def _round_to_8(inp):
            return int(max(8, np.round((inp + 1e-6) / 8) * 8))
        if any(p & (p - 1) for p in patch_size):        # non-power-of-two patch (CT-ViT's 10): odd factor, then the 2s
            stages, chans, cur = [], [embed_dim], list(patch_size)
            odd = [max(1, p // (p & -p)) for p in cur]              # largest odd divisor per axis
            n2 = [int(np.log2(c // o)) for c, o in zip(cur, odd)]  # remaining factors of 2 per axis
            plan = [tuple(odd)] + [tuple(2 if k < n else 1 for n in n2) for k in range(max(n2))]
            plan = [st for st in plan if max(st) > 1]
            mid = _round_to_8(embed_dim / 4)
            chans = [embed_dim] + [mid] * (len(plan) - 1) + [out_channels]
            for k, st in enumerate(plan):
                if k < len(plan) - 1:
                    stages.append(nn.Sequential(nn.ConvTranspose3d(chans[k], chans[k + 1], kernel_size=st, stride=st),
                                                norm(chans[k + 1]), activation()))
                else:
                    stages.append(nn.ConvTranspose3d(chans[k], chans[k + 1], kernel_size=st, stride=st))
            self.decode = nn.Sequential(*stages); self.strides, self.channels = plan, chans
            total = [int(np.prod([st[a] for st in plan])) for a in range(3)]
            assert total == list(patch_size), (total, patch_size)
            return
        num_stages = int(np.log(max(patch_size)) / np.log(2))
        strides = [[2 if (p / 2 ** n) % 2 == 0 else 1 for p in patch_size] for n in range(num_stages)][::-1]
        dim_red = (embed_dim / (2 * out_channels)) ** (1 / num_stages)
        channels = [embed_dim] + [_round_to_8(embed_dim / dim_red ** (x + 1)) for x in range(num_stages)]
        channels[-1] = out_channels
        stages = []
        for s in range(num_stages - 1):
            stages.append(nn.Sequential(nn.ConvTranspose3d(channels[s], channels[s + 1], kernel_size=strides[s], stride=strides[s]),
                                        norm(channels[s + 1]), activation()))
        stages.append(nn.ConvTranspose3d(channels[-2], channels[-1], kernel_size=strides[-1], stride=strides[-1]))
        self.decode = nn.Sequential(*stages)
        self.strides = strides; self.channels = channels

    def forward(self, x):
        return self.decode(x)


def init_he(m, neg_slope=1e-2):
    for mod in m.modules():
        if isinstance(mod, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(mod.weight, a=neg_slope, nonlinearity='leaky_relu')
            if mod.bias is not None:
                nn.init.constant_(mod.bias, 0)


class Head(nn.Module):
    """blocks: tuple of block indices (0 = patch-embed+pos). Tokens of each block are LayerNormed; multi-block
    heads concatenate and linearly project to `dim`; then PatchDecode to full resolution."""

    def __init__(self, blocks, dim, patch, n_classes, reduce='cat'):
        super().__init__()
        self.blocks, self.reduce = tuple(blocks), reduce
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in self.blocks])
        self.proj = nn.Linear(dim * len(self.blocks), dim) if (len(self.blocks) > 1 and reduce == 'cat') else None
        self.dec = PatchDecode(patch, dim, n_classes)
        init_he(self.dec)

    def forward(self, feats, grid):
        """feats: dict block -> (B, N, C) tokens (patch tokens only, (gz,gy,gx) row-major)."""
        hs = [n(feats[b]) for n, b in zip(self.norms, self.blocks)]
        if len(hs) == 1:
            h = hs[0]
        elif self.reduce == 'cat':
            h = self.proj(torch.cat(hs, -1))
        else:
            h = torch.stack(hs, 0).mean(0)
        B, N, C = h.shape
        h = h.transpose(1, 2).reshape(B, C, *grid)
        return self.dec(h)


class DropPath(nn.Module):
    def __init__(self, p):
        super().__init__(); self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = 1 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class TapSeg(nn.Module):
    """variant: TAP-CT variant name; heads: dict name -> (blocks tuple, reduce). depth: number of encoder blocks
    kept (truncation for fine-tuning). frozen: encoder under no_grad."""

    def __init__(self, variant, heads, n_classes, control='trained', depth=None, frozen=True, drop_path=0.0,
                 ckpt=None, device='cuda'):
        super().__init__()
        self.variant = variant
        self.win, self.patch, self.grid = geometry(variant)          # cache convention (z, y, x)
        self.is_spectre = variant.startswith('spectre')
        self.is_ctvit = variant.startswith('ctvit')
        self.is_cnn = variant.startswith('ctfm')
        self.is_pillar = variant.startswith('pillar0')
        self.block_geom = {}                                          # CNN/hierarchical encoders: block -> (dim, patch, grid)
        if self.is_pillar:
            # Pillar-0 Atlas: taps 0 (patch embed) and 1..3 (level outputs at 1x / 4x / 16x the patch); see primus/pillar0.py
            assert ckpt is None and drop_path == 0 and frozen, 'Pillar-0 arm: frozen level sweep only'
            from primus.data import WIN as _WIN, NORM as _NORM
            assert tuple(_WIN) == tuple(self.win) and _NORM == 'hu', \
                f'Pillar-0 arm needs PRIMUS_WIN_Z/YX = {self.win} and PRIMUS_NORM=hu; got WIN={_WIN} NORM={_NORM}'
            from primus.pillar0 import load_pillar0, level_geometry
            self.vit = load_pillar0(variant, control)
            self.vit.blocks = nn.ModuleList(list(self.vit.atlas_models))   # 3 levels, for depth accounting only
            for b, (pt, gr) in level_geometry(variant).items():
                self.block_geom[b] = (self.vit.embed_dim, pt, gr)
            assert all(len(bl) == 1 for bl, _ in heads.values()), 'Pillar-0: single-level heads only (levels differ in grid)'
            self.patch_tok, self.grid_tok = self.patch, self.grid
        elif self.is_cnn:
            # CT-FM SegResNet: b0 = stem (stride 1, 32 ch), b1..b5 = stage outputs at stride 1/2/4/8/16 with
            # 32/64/128/256/512 channels. Stages live on different grids, so heads are single-stage only.
            assert ckpt is None and drop_path == 0 and frozen, 'CT-FM arm: frozen stage sweep only'
            from primus.data import WIN as _WIN, NORM as _NORM
            assert tuple(_WIN) == tuple(self.win) and _NORM == 'ctfm', \
                f'CT-FM arm needs PRIMUS_WIN_Z=64 PRIMUS_WIN_YX=128 PRIMUS_NORM=ctfm; got WIN={_WIN} NORM={_NORM}'
            self.vit = load_ctfm(control)
            self.vit.blocks = nn.ModuleList(list(self.vit.layers))   # 5 stages, for depth accounting only
            chans = [32, 32, 64, 128, 256, 512]
            for b in range(6):
                st = 2 ** max(0, b - 1)
                self.block_geom[b] = (chans[b], (st, st, st), tuple(w // st for w in self.win))
            assert all(len(bl) == 1 for bl, _ in heads.values()), 'CT-FM: single-stage heads only (stages differ in grid)'
            self.patch_tok, self.grid_tok = self.patch, self.grid
        elif self.is_ctvit:
            # CT-CLIP's CT-ViT: (b, c, f, h, w) input at 0.75 mm in-plane / 1.5 mm z. Fed the 1.5 mm cache with the
            # in-plane crop upsampled x2 (so the 20-px patch keeps its trained 15 mm extent); frame 0 has its own
            # token and is excluded from the dense head (its slice is filled from slice 1).
            assert ckpt is None and drop_path == 0 and frozen, 'CT-ViT arm: frozen block sweep only'
            from primus.data import WIN as _WIN, NORM as _NORM
            assert tuple(_WIN) == tuple(self.win) and _NORM == 'ctclip', \
                f'CT-ViT arm needs PRIMUS_WIN_Z=41 PRIMUS_WIN_YX=240 PRIMUS_NORM=ctclip; got WIN={_WIN} NORM={_NORM}'
            self.vit = load_ctvit(control)
            self.vit.blocks = nn.ModuleList(list(self.vit.enc_spatial_transformer.layers) + list(self.vit.enc_temporal_transformer.layers))
            self.patch_tok, self.grid_tok = self.patch, (4, 24, 24)   # rest-frame tokens only (t, h, w)
        elif self.is_spectre:
            # SPECTRE-L: RAS (x, y, z) input, patch (16,16,8), 24 blocks, RoPE. Fed the same 1.5 mm LPS cache as
            # TAP-CT (its own convention is native spacing) so the two encoders see identical voxels; the window is
            # converted LPS (z,y,x) -> RAS (x,y,z) in encode() and the prediction converted back in forward().
            assert ckpt is None and drop_path == 0 and frozen, 'SPECTRE arm: frozen block sweep only'
            bb = SPECTRE_BACKBONE_NOVLA if 'novla' in variant else SPECTRE_BACKBONE
            from primus.data import WIN as _WIN, NORM as _NORM          # fail fast on a mismatched job environment
            assert tuple(_WIN) == tuple(self.win) and _NORM == 'spectre', \
                f'SPECTRE arm needs PRIMUS_WIN_Z=64 PRIMUS_WIN_YX=128 PRIMUS_NORM=spectre; got WIN={_WIN} NORM={_NORM}'
            _, self.vit = load_spectre(control, 'cpu', backbone=bb)
            self.patch_tok = tuple(self.patch[::-1])                  # (16,16,8) as the tokens are laid out
            self.grid_tok = tuple(self.grid[::-1])                    # (8,8,8) here, but keep it general
        else:
            self.vit = load_model(variant, control, 'cpu', ckpt=ckpt)
            self.patch_tok, self.grid_tok = self.patch, self.grid
        self.dim = self.vit.embed_dim
        need = max(max(b) for b, _ in heads.values())
        self.depth = depth if depth is not None else need
        assert self.depth >= need, (self.depth, need)
        self.vit.blocks = nn.ModuleList(list(self.vit.blocks)[:self.depth])   # truncate
        self.frozen = frozen
        if frozen:
            for p in self.vit.parameters():
                p.requires_grad_(False)
        if drop_path > 0:
            for blk in self.vit.blocks:
                blk.drop_path1 = DropPath(drop_path); blk.drop_path2 = DropPath(drop_path); blk.sample_drop_ratio = 0.0
        self.tapped = sorted({b for bl, _ in heads.values() for b in bl})
        if self.is_spectre or self.is_ctvit or self.is_cnn or self.is_pillar:
            assert not any(k.startswith('v3_') for k in heads), 'conv-stem heads are laid out for the TAP-CT patch; plain heads only'
        def _geom(bl):
            if self.block_geom:
                d, pt, _ = self.block_geom[bl[0]]
                return d, pt
            return self.dim, self.patch_tok
        self.heads = nn.ModuleDict({k: (HeadV3 if k.startswith('v3_') else Head)(bl, *_geom(bl), n_classes, red)
                                    for k, (bl, red) in heads.items()})
        self.nreg = 0 if (self.is_ctvit or self.is_cnn or self.is_pillar) else (self.vit.num_prefix_tokens if self.is_spectre else 1 + self.vit.num_register_tokens)

    @staticmethod
    def lps_to_ras(x):
        """(B,C,z,y,x) LPS -> (B,C,x,y,z) RAS: flip x and y, then transpose (same as lattice.run_probe.native_to_ras)."""
        return x.flip(3).flip(4).permute(0, 1, 4, 3, 2).contiguous()

    @staticmethod
    def ras_to_lps(x):
        return x.permute(0, 1, 4, 3, 2).flip(3).flip(4).contiguous()

    def encode_ctvit(self, x):
        """Per-layer taps through CT-ViT's factorised encoder (verbatim CTViT.encode, unrolled). Returns tokens of the
        four rest-frame groups only, (B, 4*24*24, 512), row-major over (t, h, w)."""
        from einops import rearrange
        v = self.vit
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)   # (B,1,41,480,480)
        first, rest = x[:, :, :1], x[:, :, 1:]
        tok = torch.cat((v.to_patch_emb_first_frame(first), v.to_patch_emb(rest)), dim=1)      # (B, 5, 24, 24, 512)
        b, t, h, w, _ = tok.shape
        video_shape = tuple(tok.shape[:-1])
        feats = {}
        if 0 in self.tapped:
            feats[0] = tok[:, 1:].reshape(b, -1, tok.shape[-1])
        z = rearrange(tok, 'b t h w d -> (b t) (h w) d')
        attn_bias = v.spatial_rel_pos_bias(h, w, device=z.device)
        i = 0
        for peg, self_attn, cross_attn, ff in v.enc_spatial_transformer.layers:
            if peg is not None:
                z = peg(z, shape=video_shape) + z
            z = self_attn(z, attn_bias=attn_bias) + z
            z = ff(z) + z
            i += 1
            if i in self.tapped:
                feats[i] = rearrange(z, '(b t) (h w) d -> b t h w d', b=b, h=h, w=w)[:, 1:].reshape(b, -1, z.shape[-1])
            if i >= max(self.tapped):
                return feats
        z = v.enc_spatial_transformer.norm_out(z)
        z = rearrange(z, '(b t) (h w) d -> b t h w d', b=b, h=h, w=w)
        z = rearrange(z, 'b t h w d -> (b h w) t d')
        for peg, self_attn, cross_attn, ff in v.enc_temporal_transformer.layers:
            if peg is not None:
                z = peg(z, shape=video_shape) + z
            z = self_attn(z) + z
            z = ff(z) + z
            i += 1
            if i in self.tapped:
                feats[i] = rearrange(z, '(b h w) t d -> b t h w d', b=b, h=h, w=w)[:, 1:].reshape(b, -1, z.shape[-1])
            if i >= max(self.tapped):
                return feats
        return feats

    def encode_cnn(self, x):
        """CT-FM: stage maps flattened to (B, N, C) row-major over (z, y, x) of each stage's own grid."""
        v = self.vit
        feats = {}
        h = v.conv_init(x)
        if 0 in self.tapped:
            feats[0] = h.flatten(2).transpose(1, 2)
        for i, level in enumerate(v.layers, 1):
            h = level['blocks'](h)
            if i in self.tapped:
                feats[i] = h.flatten(2).transpose(1, 2)
            if i >= max(self.tapped):
                break
            h = level['downsample'](h)
        return feats

    def encode(self, x):
        """x (B,1,*win) -> dict block -> (B, N, C)."""
        if self.is_pillar:
            from primus.pillar0 import level_taps
            return level_taps(self.vit, x, self.tapped)
        if self.is_cnn:
            return self.encode_cnn(x)
        if self.is_ctvit:
            return self.encode_ctvit(x)
        rope = None
        if self.is_spectre:
            h = self.vit.patch_embed(self.lps_to_ras(x))
            h, rope = self.vit._pos_embed(h)
            h = self.vit.norm_pre(h)
        else:
            h = self.vit._prepare_tokens_with_masks(x)
        feats = {}
        if 0 in self.tapped:
            feats[0] = h[:, self.nreg:]
        for i, blk in enumerate(self.vit.blocks, 1):
            h = blk(h, rope=rope) if self.is_spectre else blk(h)
            if i in self.tapped:
                feats[i] = h[:, self.nreg:]
            if i >= max(self.tapped):
                break
        return feats

    def forward(self, x, head_names=None):
        if self.frozen:
            with torch.no_grad():
                self.vit.eval()
                feats = self.encode(x)
            feats = {k: v.detach() for k, v in feats.items()}
        else:
            feats = self.encode(x)
        names = head_names or list(self.heads)
        def _grid(k):
            return self.block_geom[self.heads[k].blocks[0]][2] if self.block_geom else self.grid_tok
        out = {k: (self.heads[k](feats, _grid(k), x) if isinstance(self.heads[k], HeadV3) else self.heads[k](feats, _grid(k)))
               for k in names}
        if self.is_spectre:                                  # decoder output is RAS (x,y,z); labels are LPS (z,y,x)
            out = {k: self.ras_to_lps(v) for k, v in out.items()}
        if self.is_ctvit:                                    # head covers slices 1..40; repeat slice 1 for slice 0
            out = {k: torch.cat([v[:, :, :1], v], 2) for k, v in out.items()}
        return out


def default_heads(n_blocks=12, combos=True):
    heads = {f'b{i}': ((i,), 'cat') for i in range(n_blocks + 1)}
    if combos:
        heads['cat_4_8_12'] = ((4, 8, 12), 'cat')
        heads['cat_6_9_12'] = ((6, 9, 12), 'cat')
        heads['mean_9_12'] = ((9, 10, 11, 12), 'mean')
    return heads


def parse_heads(spec):
    """'b6,v3_b9,cat_4_8_12,mean_9_12,v3_cat_6_9_12' -> heads dict ('v3_' prefix = conv-stem skip head)."""
    heads = {}
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        key = tok
        if tok.startswith('v3_'):
            tok = tok[3:]
        if tok.startswith('b') and tok[1:].isdigit():
            heads[key] = ((int(tok[1:]),), 'cat')
        elif tok.startswith('cat_'):
            heads[key] = (tuple(int(x) for x in tok[4:].split('_')), 'cat')
        elif tok.startswith('mean_'):
            a, b = (int(x) for x in tok[5:].split('_'))
            heads[key] = (tuple(range(a, b + 1)), 'mean')
        else:
            raise ValueError(tok)
    return heads


# --------------------------------------------------------------------- PrimusV3-lite: conv stem + skip decoder
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, 3, stride=stride, padding=1)
        self.norm = LayerNormNd(cout); self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvStem(nn.Module):
    """Three resolution levels matched to the PatchDecode stages for patch (4,8,8) on a 12-slice window:
    S0 full res (16 ch), S1 /2 iso (32 ch), S2 z/4 xy/4 (64 ch). PrimusV3 idea at ~1/40 of its parameter count."""

    def __init__(self, cin=1, c=(16, 32, 64)):
        super().__init__()
        self.b0 = nn.Sequential(ConvBlock(cin, c[0]), ConvBlock(c[0], c[0]))
        self.b1 = nn.Sequential(ConvBlock(c[0], c[1], stride=2), ConvBlock(c[1], c[1]))
        self.b2 = nn.Sequential(ConvBlock(c[1], c[2], stride=2), ConvBlock(c[2], c[2]))
        self.channels = c

    def forward(self, x):
        s0 = self.b0(x); s1 = self.b1(s0); s2 = self.b2(s1)
        return s0, s1, s2


class ConvStemV3(nn.Module):
    """Wider stem in the spirit of PrimusV3's iterative patch embedding (32 -> 64 -> 256 with skips at every
    resolution). ~5 M params vs ConvStem's 0.9 M — the detail path was the largest measured win (S I/J/L), so
    this is the one component with empirical warrant for more capacity."""

    def __init__(self, cin=1, c=(32, 64, 256)):
        super().__init__()
        self.b0 = nn.Sequential(ConvBlock(cin, c[0]), ConvBlock(c[0], c[0]))
        self.b1 = nn.Sequential(ConvBlock(c[0], c[1], stride=2), ConvBlock(c[1], c[1]))
        self.b2 = nn.Sequential(ConvBlock(c[1], c[2], stride=2), ConvBlock(c[2], c[2]))
        self.channels = c

    def forward(self, x):
        s0 = self.b0(x); s1 = self.b1(s0); s2 = self.b2(s1)
        return s0, s1, s2


class SkipDecode(nn.Module):
    """PatchDecode stages with the stem skip concatenated after each upsampling (PrimusV3-style)."""

    def __init__(self, patch, embed_dim, out_channels, stem_ch=(16, 32, 64)):
        super().__init__()
        ref = PatchDecode(patch, embed_dim, out_channels)          # reuse stride/channel schedule
        st, ch = ref.strides, ref.channels                          # e.g. [(1,2,2),(2,2,2),(2,2,2)], [768,136,24,C]
        assert len(st) == 3, st
        self.up1 = nn.ConvTranspose3d(ch[0], ch[1], st[0], st[0]); self.n1 = LayerNormNd(ch[1])
        self.f1 = ConvBlock(ch[1] + stem_ch[2], ch[1])
        self.up2 = nn.ConvTranspose3d(ch[1], ch[2], st[1], st[1]); self.n2 = LayerNormNd(ch[2])
        self.f2 = ConvBlock(ch[2] + stem_ch[1], ch[2])
        self.up3 = nn.ConvTranspose3d(ch[2], ch[2], st[2], st[2]); self.n3 = LayerNormNd(ch[2])
        self.out = nn.Conv3d(ch[2] + stem_ch[0], out_channels, 1)
        self.act = nn.GELU()

    def forward(self, h, skips):
        s0, s1, s2 = skips
        h = self.act(self.n1(self.up1(h))); h = self.f1(torch.cat([h, s2], 1))
        h = self.act(self.n2(self.up2(h))); h = self.f2(torch.cat([h, s1], 1))
        h = self.act(self.n3(self.up3(h))); return self.out(torch.cat([h, s0], 1))


class ConvStemN(nn.Module):
    """Conv stem with one skip level per SkipDecodeN stage: s0 at full resolution, s_k = s_{k-1} downsampled by the
    decoder's stride of stage (n_stages - k). For patch (4,8,8) this reproduces ConvStem exactly (two iso /2 levels);
    for SPECTRE's (16,16,8) it yields three levels (/2, /4, /8)."""

    def __init__(self, strides, cin=1, c0=16):
        super().__init__()
        n = len(strides)
        chans = [c0 * 2 ** k for k in range(n)]                     # 16, 32, 64, (128)
        blocks = [nn.Sequential(ConvBlock(cin, chans[0]), ConvBlock(chans[0], chans[0]))]
        for k in range(1, n):
            st = tuple(strides[n - k])
            blocks.append(nn.Sequential(ConvBlock(chans[k - 1], chans[k], stride=st), ConvBlock(chans[k], chans[k])))
        self.blocks = nn.ModuleList(blocks)
        self.channels = tuple(chans)

    def forward(self, x):
        out = []
        for b in self.blocks:
            x = b(x); out.append(x)
        return tuple(out)


class SkipDecodeN(nn.Module):
    """SkipDecode for any number of PatchDecode stages (PrimusV3-style skip concat after every upsampling)."""

    def __init__(self, patch, embed_dim, out_channels, stem_ch):
        super().__init__()
        ref = PatchDecode(patch, embed_dim, out_channels)
        st, ch = ref.strides, ref.channels
        n = len(st)
        assert len(stem_ch) == n, (stem_ch, st)
        self.ups = nn.ModuleList(); self.norms = nn.ModuleList(); self.fuse = nn.ModuleList()
        for k in range(n):
            cin, cout = ch[k], ch[min(k + 1, n - 1)] if k < n - 1 else ch[n - 1]
            if k == n - 1:
                cin, cout = ch[n - 1], ch[n - 1]
            self.ups.append(nn.ConvTranspose3d(cin, cout, st[k], st[k])); self.norms.append(LayerNormNd(cout))
            skip_c = stem_ch[n - 1 - k]
            self.fuse.append(ConvBlock(cout + skip_c, cout) if k < n - 1 else None)
        self.out = nn.Conv3d(ch[n - 1] + stem_ch[0], out_channels, 1)
        self.act = nn.GELU()

    def forward(self, h, skips):
        n = len(self.ups)
        for k in range(n):
            h = self.act(self.norms[k](self.ups[k](h)))
            s = skips[n - 1 - k]
            h = self.fuse[k](torch.cat([h, s], 1)) if k < n - 1 else torch.cat([h, s], 1)
        return self.out(h)


class HeadV3(nn.Module):
    """Skip-decoder head: own conv stem (kept per-head so heads stay independent), block taps as in Head."""

    def __init__(self, blocks, dim, patch, n_classes, reduce='cat'):
        super().__init__()
        self.blocks, self.reduce = tuple(blocks), reduce
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in self.blocks])
        self.proj = nn.Linear(dim * len(self.blocks), dim) if (len(self.blocks) > 1 and reduce == 'cat') else None
        self.stem = ConvStem()
        self.dec = SkipDecode(patch, dim, n_classes)
        init_he(self.dec); init_he(self.stem)

    def forward(self, feats, grid, x=None):
        hs = [n(feats[b]) for n, b in zip(self.norms, self.blocks)]
        h = hs[0] if len(hs) == 1 else (self.proj(torch.cat(hs, -1)) if self.reduce == 'cat' else torch.stack(hs, 0).mean(0))
        B, N, C = h.shape
        h = h.transpose(1, 2).reshape(B, C, *grid)
        return self.dec(h, self.stem(x))


# ------------------------------------------------------------------------------------------ loss (nnU-Net style)
def soft_dice_loss(logits, target, smooth=1e-5, batch_dice=False):
    """Soft Dice over foreground classes (nnU-Net MemoryEfficientSoftDiceLoss, do_bg=False). batch_dice=False:
    per sample per class, then mean; True: sums pooled over the batch (the setting of the ReXGrounding 502 plan)."""
    p = torch.softmax(logits, 1)[:, 1:]
    t = F.one_hot(target, logits.shape[1])[..., 1:].permute(0, 4, 1, 2, 3).to(p.dtype)
    dims = (0, 2, 3, 4) if batch_dice else (2, 3, 4)
    inter = (p * t).sum(dims); sp = p.sum(dims); st = t.sum(dims)
    dc = (2 * inter + smooth) / (sp + st + smooth).clamp_min(1e-8)
    return 1 - dc.mean()


def fg_dice_loss(logits, target, smooth=1e-5):
    """Class-agnostic foreground Dice (any lesion vs background), per sample."""
    p = 1 - torch.softmax(logits, 1)[:, 0]
    t = (target > 0).to(p.dtype)
    dims = (1, 2, 3)
    inter = (p * t).sum(dims); sp = p.sum(dims); st = t.sum(dims)
    return 1 - ((2 * inter + smooth) / (sp + st + smooth)).mean()


LOSSES = {}


def make_loss(name):
    """dc_ce: nnU-Net default (per-sample dice); batchdc_ce: batch dice (502 plan); dc_ce_fg: + class-agnostic fg dice."""
    def f(logits, target):
        lg = logits.float(); ce = F.cross_entropy(lg, target)
        if name == 'dc_ce':
            return soft_dice_loss(lg, target) + ce
        if name == 'batchdc_ce':
            return soft_dice_loss(lg, target, batch_dice=True) + ce
        if name == 'dc_ce_fg':
            return soft_dice_loss(lg, target) + ce + fg_dice_loss(lg, target)
        if name == 'batchdc_ce_fg':
            return soft_dice_loss(lg, target, batch_dice=True) + ce + fg_dice_loss(lg, target)
        raise ValueError(name)
    return f


dc_ce_loss = make_loss('dc_ce')

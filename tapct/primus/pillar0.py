"""Pillar-0 (YalaLab, arXiv 2511.17803) as a frozen hierarchical encoder for the block-depth study.

Atlas = multi-scale attention over a token pyramid: level 0 (2 blocks) sees all scales and emits the finest
(patch-resolution) tokens, which are then dropped; level 1 (2 blocks) emits the 4x-coarser scale; level 2
(8 blocks) the 16x-coarser one. The number of scales is derived from the INPUT size, so a window large enough
for three scales is required for the deep level to run at all. Taps here = the three level outputs (stage-wise,
like a CNN), each returned in spatial row-major (z, y, x) order after undoing Atlas's window-major layout.

Input: 11 channels = the 10 anatomical HU windows + min-max from RAVE (vendored in primus/pillar_windowing.py),
computed from raw HU. Pillar-0 trains at 1.25 mm; here it is fed the shared 1.5 mm cache like the other encoders.
Releases: ChestCT (patch 8x8x4 in (H,W,D)) and AbdomenCT (patch 6^3); the model modality key is per release.
"""
import glob
import math
import os

import torch
import torch.nn as nn
from einops import rearrange

from primus.pillar_windowing import apply_windowing

RELEASES = {  # variant -> (HF repo, modality key, window (z,y,x), token patch (z,y,x))
    # patch_size in the release config is (D, H, W) = our (z, y, x); token grids must be divisible by 4 twice
    # (merge_ratio) for all three levels to run: chest 128^3 -> (16, 16, 32) tokens, abdomen 192^3 -> 32^3.
    'pillar0-chest-seg': ('YalaLab/Pillar0-ChestCT', 'chest_ct', (128, 128, 128), (8, 8, 4)),
    'pillar0-abd-seg': ('YalaLab/Pillar0-AbdomenCT', 'abdomen_ct', (192, 192, 192), (6, 6, 6)),
}


def snapshot(repo):
    d = glob.glob(os.path.expanduser(f'~/.cache/huggingface/hub/models--{repo.replace("/", "--")}/snapshots/*/'))
    if not d:
        raise FileNotFoundError(f'{repo} not in the HF cache')
    return d[0]


def load_pillar0(variant, control='trained'):
    """Build only the vision tower (MultiModalAtlas) from the release's own code — the full CLIP class also
    instantiates the Qwen3-8B text tower through transformers' remote-code loader, which cannot resolve the repo's
    relative imports from a local snapshot offline — and load the `model.visual.*` tensors from model.safetensors."""
    import importlib, json, sys, types
    from safetensors.torch import load_file
    repo, mod, win, patch = RELEASES[variant]
    p = snapshot(repo)
    pkg = 'pillar0_' + repo.split('-')[-1].lower()                      # ext/pillar0_chestct -> snapshot symlink
    sys.path.insert(0, os.environ.get('PILLAR0_EXT', 'ext'))
    atlas_mod = importlib.import_module(f'{pkg}.multimodal_atlas')
    cfg = json.load(open(os.path.join(p, 'config.json')))
    vcfg = cfg['vision_cfg']
    args = types.SimpleNamespace(**vcfg)                                 # what CLIPVisionCfg would have carried
    visual = atlas_mod.MultiModalAtlas(args=args, embed_dim=cfg['embed_dim'], num_classes=2, model_config=vcfg['model_config'])
    if control != 'randinit':
        sd = load_file(os.path.join(p, 'model.safetensors'))
        vs = {k[len('model.visual.'):]: v for k, v in sd.items() if k.startswith('model.visual.')}
        missing, unexpected = visual.load_state_dict(vs, strict=False)
        assert not missing, f'visual tower tensors missing: {missing[:5]}'
        assert not unexpected, f'unexpected tensors: {unexpected[:5]}'
    visual.eval()
    # the multiscale layout is computed from the CONFIG image size; point it at our window ((H, W, D) order)
    mc = visual.model_config['modalities'][mod]
    mc['image_size'] = list(win)                                       # (D, H, W) = our (z, y, x)
    visual.embed_dim = visual.model_config['model']['atlas_args']['dim']
    visual.pillar_modality = mod
    visual.pillar_win, visual.pillar_patch = win, patch
    return visual


def eleven_windows(hu):
    """(B,1,D,H,W) raw HU -> (B,11,D,H,W) in Pillar-0's channel order (10 anatomical windows + minmax).
    RAVE returns the windows stacked on a leading axis for a 5-D input; fold them into the channel dim."""
    out = apply_windowing(hu, windows='all', modality='CT')          # (11, B, 1, D, H, W)
    assert out.ndim == 6 and out.shape[0] == 11 and out.shape[2] == 1, out.shape
    return out[:, :, 0].permute(1, 0, 2, 3, 4).contiguous()          # (B, 11, D, H, W)


# Empirically mapped per-tap token layouts (primus/pillar0_map.py, 2026-08-30), CHEST release:
#   tap 0 (patch embed):        window-major (z,y,x), no shift
#   tap 1 (level-0 out, scale 0): window-major (z,y,x), tokens shifted +1 in z -> roll -1 after un-windowing
#   tap 2 (level-1 out, scale 1): row-major (z,y,x)
#   tap 3 (level-2 out, scale 2): <= 2 tokens, global — not usable for dense read-out
# The abdomen release's tap-1 order does not resolve to any candidate; it is dropped (user decision).
TAP_LAYOUT = {'pillar0-chest-seg': {0: ('wm', (0, 0, 0)), 1: ('wm', (-1, 0, 0)), 2: ('rm', (0, 0, 0)), 3: ('rm', (0, 0, 0))}}


def unwindow(tokens, layout, merge_ratio, bsz, tap=None, variant=None):
    """Atlas scale tokens back to (B, N, C) in row-major (z, y, x) order, per the empirically mapped layout."""
    mode, shift = TAP_LAYOUT[variant][tap]
    if math.prod(layout['grid_size']) == 1 or mode == 'rm':
        t = tokens if tokens.shape[0] == bsz else rearrange(tokens, '(b nw) k c -> b (nw k) c', b=bsz)
        return t
    d, h, w = layout['grid_size']                            # window grid, (z, y, x) order per the mapping
    m0, m1, m2 = merge_ratio
    t = rearrange(tokens, '(b d h w) (m0 m1 m2) c -> b (d m0) (h m1) (w m2) c', b=bsz, d=d, h=h, w=w, m0=m0, m1=m1, m2=m2)
    if any(shift):                                           # measured offset is (decoded - true); roll it away
        t = torch.roll(t, shifts=shift, dims=(1, 2, 3))
    return t.reshape(bsz, -1, t.shape[-1])


@torch.no_grad()
def level_taps(visual, hu, tapped, variant='pillar0-chest-seg'):
    """Run the Atlas levels, returning {level+1: (B, N, C) row-major tokens} for the tapped levels (1-based;
    tap 0 = the patch-embedded, un-attended scale-0 tokens)."""
    mod = visual.pillar_modality
    x11 = eleven_windows(hu)
    x, _ = visual.build_scales({mod: x11}, None)
    mc = visual.model_config['modalities'][mod]
    layout = visual.prepare_multiscale_layout(img_size=mc['image_size'], merge_ratio=mc['merge_ratio'],
                                              local2global=mc['local2global'], patch_size=mc['patch_size'])
    grid_sizes = [l['grid_size'] for l in layout]
    bsz = hu.shape[0]
    feats = {}
    if 0 in tapped:
        feats[0] = unwindow(x[0], layout[0], mc['merge_ratio'], bsz, tap=0, variant=variant)
    for level, atlas_model in enumerate(visual.atlas_models):
        if len(x) == 0:
            break
        is_last = len(x) == 1 or level == len(visual.atlas_models) - 1
        x = atlas_model(x, grid_sizes=grid_sizes[level:], multiscale_layout=layout[level:],
                        merge_ratio=mc['merge_ratio'], local2global=mc['local2global'], modality=mod)
        if (level + 1) in tapped:
            feats[level + 1] = unwindow(x[0], layout[level], mc['merge_ratio'], bsz, tap=level + 1, variant=variant)
        if (level + 1) >= max(tapped):
            break
        if not is_last:
            x = x[1:]
    return feats


def level_geometry(variant, n_levels=3):
    """block -> (patch (z,y,x), grid) for taps 0..3 on this variant's window."""
    _, _, win, patch = RELEASES[variant]
    geom = {}
    for b in range(n_levels + 1):
        f = 4 ** max(0, b - 1)
        pt = tuple(p * f for p in patch)
        geom[b] = (pt, tuple(w // p for w, p in zip(win, pt)))
    return geom

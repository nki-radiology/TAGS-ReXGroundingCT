"""TAP-CT variants and loader: `geometry` and `load_model` from Kevin Groot Lipman's `lattice` package
(lattice/models.py, shared 2026-10-19), unchanged except that the model revision on Hugging Face can be pinned.

Only the TAP-CT entries are kept. model.py also imports loaders for encoders the challenge never used (SPECTRE,
CT-CLIP's CT-ViT, CT-FM); they exist here as stubs so that import works.
"""
import os
import torch

VARIANTS = {
    'tap-ct-s-3d': dict(win=(12, 224, 224), patch=(4, 8, 8)),
    'tap-ct-b-3d': dict(win=(12, 224, 224), patch=(4, 8, 8)),   # window depth is overridden by PRIMUS_WIN_Z (24 here)
    'tap-ct-s-2-5d': dict(win=(6, 224, 224), patch=(1, 16, 16)),
    'tap-ct-b-2-5d': dict(win=(6, 224, 224), patch=(1, 16, 16)),
    'tap-ct-s-2d': dict(win=(1, 224, 224), patch=(1, 16, 16)),
    'tap-ct-b-2d': dict(win=(1, 224, 224), patch=(1, 16, 16)),
}
# Hugging Face revision of fomofo/<variant>. Our fine-tuned encoders are consistent with this revision of tap-ct-b-3d.
HF_REVISION = os.environ.get('TAPCT_HF_REVISION', '7694ac2563707791f8fbb098695ec6571cfdbf4a')

SPECTRE_BACKBONE = SPECTRE_BACKBONE_NOVLA = None


def _not_included(*_a, **_k):
    raise NotImplementedError('only the TAP-CT loader is included in this release')


load_spectre = load_ctvit = load_ctfm = _not_included


def geometry(name):
    v = VARIANTS[name]
    g = tuple(w // p for w, p in zip(v['win'], v['patch']))
    return v['win'], v['patch'], g


def load_model(name, control, device, seed=0, ckpt=None):
    """control in {'trained', 'randinit', 'shuffled', 'poszero'}; returns the inner DinoVisionTransformer.
    ckpt: optional path to a DINOv2-style teacher_checkpoint.pth (intermediate training checkpoint)."""
    from transformers import AutoModel, AutoConfig
    hf = f'fomofo/{name}'
    rev = HF_REVISION if name == 'tap-ct-b-3d' else None
    if control == 'randinit':
        cfg = AutoConfig.from_pretrained(hf, trust_remote_code=True, revision=rev)
        torch.manual_seed(seed)
        m = AutoModel.from_config(cfg, trust_remote_code=True)
    else:
        m = AutoModel.from_pretrained(hf, trust_remote_code=True, revision=rev)
    vit = m.model
    if ckpt is not None:
        sd = torch.load(ckpt, map_location='cpu', weights_only=False)
        sd = sd.get('teacher', sd)
        bb = {k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')}
        if not bb:
            raise RuntimeError(f'no backbone.* keys in {ckpt}; keys: {list(sd)[:5]}')
        missing, unexpected = vit.load_state_dict(bb, strict=False)
        missing = [k for k in missing if 'mask_token' not in k]
        if missing or unexpected:
            raise RuntimeError(f'ckpt key mismatch: missing={missing[:5]} unexpected={unexpected[:5]}')
        print(f'loaded {len(bb)} backbone tensors from {ckpt}', flush=True)
    if control == 'shuffled':
        # permute the elements of every parameter inside the transformer blocks: identical scale
        # statistics per tensor, no function. Patch embedding, pos-embed and norms are left trained.
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for n, p in vit.blocks.named_parameters():
                if p.ndim >= 2 or n.endswith('gamma'):
                    flat = p.data.view(-1)
                    p.data.copy_(flat[torch.randperm(flat.numel(), generator=g)].view_as(p.data))
    elif control == 'poszero':
        with torch.no_grad():
            vit.pos_embed.zero_()
    assert not vit.chunked_blocks
    return vit.to(device).eval()

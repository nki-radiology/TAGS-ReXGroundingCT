"""VoxTellTrainer_T1_full -- Track-1 model using free text + category + TotalSegmentator.

Everything the main track permits, in one model:
  * TEXT (required): the exact free-text finding -> Qwen3-Embedding-4B (2560-d), consumed
    directly by the model. This is what keeps the method Track-1 legal; it is never parsed
    into structured fields before reaching the network.
  * CATEGORY (auxiliary): a zero-initialised learned embedding ADDED to the text vector, so
    at init the model behaves exactly as pretrained and learns how much category to inject.
    The category is ALWAYS present: 7687/7687 training findings and 582/582 test findings
    carry one. So there is no "unknown" state and no train/test shift, and cat_dropout
    defaults to 0 -- dropping it would train the model in a regime that never occurs at
    inference. The real risk (leaning on category and regressing to whole-category masks,
    which cannot separate two findings of the SAME category in different locations) is
    MEASURED by the textsens diagnostic rather than guarded by a mismatch; --cat-dropout
    remains available if that diagnostic shows over-reliance.
  * ANATOMY (input channels): the 5 TotalSegmentator lung lobes as binary channels, which is
    exactly what the text names ("right upper lobe"). VoxTell ships with input_channels=1, so
    the first conv is EXPANDED -- pretrained CT weights copied into ch0, anatomy channels
    zero-initialised -- rather than reinitialised, preserving pretrained behaviour at init.
  * RESTRICTION (post-processing, inference only): handled in eval, not here.

Recipe notes carried over from train_t1_base: 192^3 patches (VoxTell's pos_embed is hard-tied
to 12^3=1728 tokens, so this is not negotiable), deep supervision at 5 scales, NO left-right
mirroring (it would teach the model that laterality is noise), SGD 1e-4 poly.

Prompts are ONLY real annotated findings -- no fabricated negatives. See train_t1_base for why
(absence of an annotation is not absence of pathology in a report-derived dataset).

    python train_t1_full.py --epochs 250 --batch 2
"""
import argparse, os
import contextlib
import json
import pydoc
import sys
from pathlib import Path

import os

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from voxtell.model.voxtell_model import VoxTellModel  # noqa: E402

from train_t1_base import DS_WEIGHTS, deep_sup_loss, zscore  # noqa: E402,F401

from paths import VOXTELL_DIR as MODEL_DIR, TEXT_BANK_DIR as BANK_DIR, CTRATE_ROOT  # noqa: E402
# Pre-decompressed z-scored fp16 siblings (stage_npy.py). Node-local is preferred, but
# /processing on herakles is 98% full and the full cache is 513 GB, so anything that did not
# fit is mmap'd straight off rhea instead: measured 0.50 s for the 8 crops an item actually
# touches, vs ~2 s to gunzip+zscore the .nii.gz. Set T1_NPY_FALLBACK="" to disable.
NPY_SRC_ROOT = os.environ.get("T1_NPY_SRCROOT", "")
NPY_FALLBACK = os.environ.get("T1_NPY_FALLBACK", "")
TS_ROOT = CTRATE_ROOT / "ts_seg" / "ts_total"
LOBES = [10, 11, 12, 13, 14]                    # upper_l, lower_l, upper_r, middle_r, lower_r
CODES14 = ["1a", "1b", "1c", "1d", "1e", "1f", "2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h"]
CAT2IDX = {c: i for i, c in enumerate(CODES14)}   # 14 real categories; no "unknown"
TEXT_DIM = 2560
FIRST_CONV_KEYS = [                             # encoder and decoder share the stem
    "encoder.stem.convs.0.conv.weight",
    "encoder.stem.convs.0.all_modules.0.weight",
    "decoder.encoder.stem.convs.0.conv.weight",
    "decoder.encoder.stem.convs.0.all_modules.0.weight",
]


TS_FAST = Path(os.environ.get("TAGS_TS_FAST", str(TS_ROOT)))   # optional staged copy on fast disk


def ts_path(case: str) -> Path:
    sub = "valid_fixed" if case.startswith("valid") else "train_fixed"
    pid = "_".join(case.split("_")[:2])
    acc = "_".join(case.split("_")[:3])
    rel = Path(sub) / pid / acc / f"{case}.nii.gz"
    fast = TS_FAST / rel                     # prefer /processing copy if present (long training)
    return fast if fast.exists() else TS_ROOT / rel


def tn_penalty(out, tn, valid):
    """True-negative restriction: mean predicted probability inside DISALLOWED voxels (`tn`==1),
    averaged over valid prompts. `out` is the deep-sup list (or tensor); use the finest scale,
    which matches the full-res `tn` mask. Pushes prob->0 where the finding cannot be."""
    import torch as _t
    finest = out if _t.is_tensor(out) else max(out, key=lambda o: o.shape[-1])
    p = _t.sigmoid(finest.float())                      # (mb, n_pos, D, H, W)
    tn = tn.to(p.dtype)
    num = (p * tn).flatten(2).sum(-1)                    # (mb, n_pos)
    den = tn.flatten(2).sum(-1).clamp_min(1.0)
    pen = num / den
    v = valid.to(p.dtype)
    return (pen * v).sum() / v.sum().clamp_min(1.0)


def build_model(device, in_channels=6, deep_supervision=True):
    """Load v1.1 and widen the stem from 1 -> in_channels (pretrained CT in ch0, zeros after)."""
    plans = json.load(open(MODEL_DIR / "plans.json"))
    cfg = plans["configurations"]["3d_fullres"]
    ak = dict(cfg["architecture"]["arch_kwargs"])
    for k in cfg["architecture"]["_kw_requires_import"]:
        if ak[k] is not None:
            ak[k] = pydoc.locate(ak[k])
    net = VoxTellModel(input_channels=in_channels, **ak, decoder_layer=4,
                       text_embedding_dim=TEXT_DIM, num_maskformer_stages=5, num_heads=32,
                       query_dim=2048, project_to_decoder_hidden_dim=2048,
                       deep_supervision=deep_supervision)
    sd = dict(torch.load(MODEL_DIR / "fold_0" / "checkpoint_final.pth", map_location="cpu",
                         weights_only=False)["network_weights"])
    for k in FIRST_CONV_KEYS:
        if k not in sd:
            continue
        w = sd[k]                                            # (out, 1, 3,3,3)
        wide = torch.zeros(w.shape[0], in_channels, *w.shape[2:], dtype=w.dtype)
        wide[:, :1] = w                                      # CT keeps pretrained weights
        sd[k] = wide                                         # anatomy channels start at zero
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    if missing:
        print(f"[model] reinitialised {len(missing)} tensors: {missing[:3]}", flush=True)
    return net.to(device), [int(x) for x in cfg["patch_size"]]



class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with a trainable low-rank update: y = W0 x + (B A) x * a/r.

    B is zero-initialised, so at start the module == the pretrained Linear (no behaviour change).
    Used to ADAPT VoxTell's text pathway without overwriting it -- full fine-tuning demonstrably
    destroyed the pretrained text-reading (Dice(A,B) 1.000); freezing + LoRA is meant to preserve
    it while still letting the finding text shape the query.
    """
    def __init__(self, base, r=16, alpha=16):
        super().__init__()
        self.base = base
        for pm in self.base.parameters():
            pm.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * (r ** -0.5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t()) @ self.B.t() * self.scale


def apply_lora(net, targets, r=16, alpha=16):
    """Replace every nn.Linear whose qualified name contains any `targets` substring with LoRA."""
    n = 0
    for name, mod in list(net.named_modules()):
        if isinstance(mod, nn.Linear) and any(t in name for t in targets):
            parent, *path, last = [net] + name.split(".")
            for step in path:
                parent = getattr(parent, step)
            setattr(parent, last, LoRALinear(mod, r, alpha))
            n += 1
    return n


def attention_exclusion(attn, target, valid, gt_overlap_max=0.05):
    """Make different findings on one scan compete for the SAME image tokens.

    MEASURED MOTIVATION. Ground-truth masks for two findings on a scan are disjoint 99% of the
    time on val and 92% on train, yet our predictions overlap on 46% of pairs (0.228 mean on
    same-category pairs against a target of 0.014). 23% of predicted volume is claimed by more
    than one finding, and assigning that contested volume correctly is worth +0.046 Dice.

    WHY AT THE ATTENTION, not the output mask. The mask comes from an einsum of the mask
    embedding against image features, so overlap there is downstream of the decision. The
    cross-attention map is where the model decides which regions a sentence refers to. Penalising
    the dot product of two findings' attention distributions makes them compete at that point.

    THE GUARD MATTERS. 7.3% of same-category training pairs have near-identical ground truth --
    the annotation genuinely assigns them the same region. Penalising those fights the labels, so
    pairs whose targets already overlap above `gt_overlap_max` are skipped.

    attn:   (B, N, T) attention over T bottleneck tokens, per finding
    target: (B, N, ...) the ground-truth masks, used ONLY to skip legitimately-overlapping pairs
    valid:  (B, N) 1 where the prompt is real (crops are padded to a fixed N)
    """
    if attn is None or attn.dim() != 3:
        return attn.new_zeros(()) if attn is not None else 0.0
    B, N, T = attn.shape
    if N < 2:
        return attn.new_zeros(())
    a = attn / (attn.sum(-1, keepdim=True) + 1e-8)          # rows are distributions already
    flat = target.flatten(2).float()                         # (B,N,V)
    inter = torch.einsum("bnv,bmv->bnm", flat, flat)
    size = flat.sum(-1)
    dice_gt = 2 * inter / (size[:, :, None] + size[:, None, :] + 1e-6)
    ok = (dice_gt <= gt_overlap_max).float()                 # skip genuinely-overlapping pairs
    ok = ok * valid[:, :, None] * valid[:, None, :]
    eye = torch.eye(N, device=attn.device)[None]
    ok = ok * (1 - eye)                                      # never penalise a finding vs itself
    ov = torch.einsum("bnt,bmt->bnm", a, a)                  # attention overlap per pair
    denom = ok.sum().clamp(min=1.0)
    return (ov * ok).sum() / denom


class T1Model(nn.Module):
    """VoxTell + the category embedding as ONE module.

    DDP only synchronises gradients of parameters inside the module it wraps, so the category
    embedding must live here rather than beside the net -- otherwise each rank would silently
    learn its own copy.
    """

    def __init__(self, net, n_cats, text_dim, anat_pos=0, attn_excl=0.0, film_stages=(),
                 gate_stages=(), region_aux=False, detect_aux=False):
        super().__init__()
        self.net = net
        self.cat_embed = nn.Embedding(n_cats, text_dim)
        nn.init.zeros_(self.cat_embed.weight)          # start as pure text
        self.anat_pos, self.attn_excl = anat_pos, attn_excl
        self._attn = None
        self.film_stages = tuple(film_stages)
        self.gate_stages = tuple(gate_stages)
        self.region_aux = bool(region_aux)
        self.detect_aux = bool(detect_aux)
        if self.film_stages or self.gate_stages:
            self._install_film(text_dim)
        if self.region_aux:
            self._install_region(text_dim)
        if self.detect_aux:
            self._install_detect()
        if attn_excl:
            # VoxTell's prompt decoder RETURNS the cross-attention maps and throws them away
            # (`mask_embedding, _ = self.transformer_decoder(...)`). They are (B, n_findings,
            # 1728) -- one distribution per finding over the bottleneck tokens -- and they are
            # exactly where the model decides WHICH image regions a sentence refers to.
            # Stash them on the forward pass so the loss can make findings compete for tokens.
            td = self.net.transformer_decoder
            _orig_a = td.forward

            def _fwd_a(tgt, memory, pos=None, _o=_orig_a, _self=self, **kw):
                out = _o(tgt, memory,
                         pos=_self._pos if getattr(_self, "_pos", None) is not None else pos,
                         **kw)
                _self._attn = out[1] if isinstance(out, (tuple, list)) else None
                return out

            self._pos = getattr(self, "_pos", None)
            td.forward = _fwd_a
        if anat_pos and not attn_excl:
            # VoxTell adds `self.pos_embed` -- a FIXED sinusoidal grid, identical for every patch
            # and every scan -- to the keys of the prompt decoder. So at the point where text
            # attends to image, the model has NO signal for where in the PATIENT it is; token
            # (0,0,0) is the same whether the window sits at the apex or the base.
            #
            # Wrapping the decoder lets us substitute a per-sample, anatomy-relative encoding
            # without forking the installed package. A plain attribute, NOT a buffer: DDP
            # broadcasts buffers every forward, which would clobber per-rank values.
            # Patch the BOUND METHOD, not the module. Replacing the submodule renames every
            # parameter (transformer_decoder.* -> transformer_decoder.td.*) and the checkpoint
            # then refuses to load -- which is exactly how the first attempt died.
            td = self.net.transformer_decoder
            _orig = td.forward

            def _fwd(tgt, memory, pos=None, _o=_orig, _self=self, **kw):
                return _o(tgt, memory,
                          pos=_self._pos if _self._pos is not None else pos, **kw)

            self._pos = None
            td.forward = _fwd

    def _install_film(self, text_dim):
        """Make the ENCODER SKIPS prompt-dependent.

        VoxTell computes `skips = encoder(img)` once and hands the SAME tensors to every prompt;
        text only ever reaches the 12^3 bottleneck through the transformer decoder. So the
        full-resolution features -- the ones that decide a mask's fine shape -- are computed
        without any knowledge of what is being looked for. That is consistent with our failure
        mode: half the ground truth is never predicted at all (recall 0.505, invariant under every
        selection scheme), which no amount of re-selecting text-blind features can fix.

        FiLM (per-prompt channel-wise scale/shift) on each skip makes them prompt-aware. The
        decoder already loops over prompts, so this costs an affine per channel, NOT a second
        convolutional pass.

        The heads are ZERO-INITIALISED: gamma = 1 + 0, beta = 0, so at init the model is
        numerically identical to the un-filmed one and can be warm-started from any existing
        checkpoint. Verified by an exact-equality smoke test before launch.
        """
        import torch as _t
        enc = self.net.encoder
        # Read the channel widths from the encoder's own config. A forward probe would need a
        # patch large enough that the deepest stage keeps >1 spatial element (InstanceNorm
        # rejects 1^3 in train mode), and there is no reason to run convolutions for this.
        chans = getattr(enc, "output_channels", None)
        if chans is None:
            chans = [st[-1].conv2.all_modules[0].out_channels for st in enc.stages]
        chans = [int(c) for c in chans]
        self._film_chans = chans
        # SPATIAL GATE (--attn-gate). Every high-resolution conditioning path in this
        # architecture is spatially UNIFORM -- the decoder's mask embeddings and FiLM alike apply
        # one scalar per channel, so they can say "emphasise nodule-like features" but never
        # "emphasise them HERE and not THERE" (see EXPERIMENTS.md 4o-1). The ONLY spatially
        # selective signal is the transformer's 12^3 cross-attention, and it never leaves the
        # bottleneck. This upsamples that map and uses it as a per-VOXEL gate on the skips:
        #     skip <- skip * (1 + g_c * A_up)
        # g_c is a per-channel gain, zero-initialised, so the model is the identity at init.
        self.gate_gain = nn.ParameterDict()
        for i in range(len(chans)):
            if i in self.gate_stages:
                self.gate_gain[str(i)] = nn.Parameter(torch.zeros(chans[i]))
        self.film_trunk = nn.Sequential(nn.Linear(text_dim, 256), nn.GELU())
        self.film_heads = nn.ModuleList()
        for i in range(len(chans)):
            if i in self.film_stages:
                h = nn.Linear(256, 2 * chans[i])
                nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)   # identity at init
                self.film_heads.append(h)
            else:
                self.film_heads.append(nn.Identity())

        net = self.net
        _self = self

        def _forward(img, text_embedding=None, _n=net, _s=_self):
            from einops import rearrange, repeat
            skips = _n.encoder(img)
            sel = skips[_n.selected_decoder_layer]
            be = rearrange(sel, "b c d h w -> b h w d c")
            be = _n.project_bottleneck_embed(be)
            be = rearrange(be, "b h w d c -> (h w d) b c")
            te = text_embedding.squeeze(2)                      # (B, N, D)
            tx = repeat(te, "b n dim -> n b dim")
            tx = _n.project_text_embed(tx)
            mask_embedding, attn = _n.transformer_decoder(
                tgt=tx, memory=be,
                pos=_s._pos if getattr(_s, "_pos", None) is not None else _n.pos_embed,
                memory_key_padding_mask=None)
            if _s.attn_excl:
                _s._attn = attn
            amap = None
            if _s.gate_stages:
                import torch.nn.functional as _F
                a = attn[-1] if isinstance(attn, (list, tuple)) else attn   # last layer, (B,N,T)
                Bv, Nv, T = a.shape
                g = int(round(T ** (1 / 3)))
                # tokens were flattened as (h w d) from 'b c d h w -> b h w d c', so undo in
                # that order and put d first again -- getting this wrong silently transposes
                # the gate and would look like the method simply does not work.
                amap = a.reshape(Bv, Nv, g, g, g).permute(0, 1, 4, 2, 3)     # (B,N,d,h,w)
                # attention sums to 1 over ~1728 tokens, so raw values are ~6e-4. Peak-normalise
                # per finding, else the gate is numerically dead whatever the gain learns.
                amap = amap / amap.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-8)
            mask_embedding = repeat(mask_embedding, "n b dim -> b n dim")
            mask_embeddings = [pr(mask_embedding) for pr in _n.project_to_decoder_channels]

            cond = _s.film_trunk(te)                            # (B, N, 256)
            outs = []
            for pi in range(te.shape[1]):
                pe = [m[:, pi:pi + 1] for m in mask_embeddings]
                sk = []
                for i, sfeat in enumerate(skips):
                    f = sfeat
                    if i in _s.film_stages:
                        gb = _s.film_heads[i](cond[:, pi])       # (B, 2C)
                        c = gb.shape[1] // 2
                        gamma = 1.0 + gb[:, :c].view(-1, c, 1, 1, 1)
                        beta = gb[:, c:].view(-1, c, 1, 1, 1)
                        f = f * gamma + beta
                    if i in _s.gate_stages and amap is not None:
                        import torch.nn.functional as _F
                        au = _F.interpolate(amap[:, pi:pi + 1], size=f.shape[2:],
                                            mode="trilinear", align_corners=False)
                        gg = _s.gate_gain[str(i)].view(1, -1, 1, 1, 1)
                        f = f * (1.0 + gg * au)
                    sk.append(f)
                outs.append(_n.decoder(sk, pe))
            outs = [torch.cat(so, dim=1) for so in zip(*outs)]
            return outs if _n.deep_supervision else outs[0]

        self._pos = getattr(self, "_pos", None)
        net.forward = _forward

    def _install_region(self, text_dim):
        """Auxiliary REGION head: predict, per prompt, the soft anatomical region its text names.

        Targets come from region_prior/lung_regions.py -- parsed from the finding TEXT, which is
        complete even where train masks are capped at 3 instances, so this is full-coverage
        supervision for WHERE, decoupled from the capped WHAT. The head reads the final
        full-resolution decoder features through a channel projection of the prompt embedding;
        gradients flow back through the shared trunk, which is the point. A forward hook on the
        decoder's last stage captures those features -- the decoder is already called once per
        prompt, so a counter indexes which prompt is in flight. Logits are computed inside the
        hook and features are NOT retained (a (mb,32,192^3) tensor per prompt would be ~1 GB).
        """
        # The decoder REVERSES the projection list, so the final full-resolution einsum uses
        # projection [0] (out=32), not [-1] (an intermediate multi-head one, out=10240).
        _first = self.net.project_to_decoder_channels[0]
        _lins = [m_ for m_ in _first.modules() if isinstance(m_, nn.Linear)]
        C = _lins[-1].out_features if _lins else _first.out_features
        self.region_proj = nn.Sequential(nn.Linear(text_dim, 256), nn.GELU(),
                                         nn.Linear(256, C))
        self._rcond, self._rk, self._rlogits = None, 0, []
        self.region_out = None

        def _rhook(mod, inp, out, _self=self):
            if _self._rcond is None:
                return
            k = _self._rk; _self._rk += 1
            c = _self._rcond[:, k].to(out.dtype)
            if c.shape[1] != out.shape[1]:
                raise RuntimeError(f"region head built for {c.shape[1]} channels, "
                                   f"decoder final stage has {out.shape[1]}")
            _self._rlogits.append(torch.einsum("bcxyz,bc->bxyz", out, c))

        self.net.decoder.stages[-1].register_forward_hook(_rhook)

    def _install_detect(self):
        """Auxiliary DETECTION head: one prompt-INDEPENDENT channel on the final decoder
        stage predicting the union of all annotated findings in the patch ("is anything
        here"). Asymmetric BCE at the loss: full weight on annotated voxels, low weight on
        background, because unREPORTED pathology is unannotated -- background is uncertain,
        foreground is not."""
        _first = self.net.project_to_decoder_channels[0]
        _lins = [m_ for m_ in _first.modules() if isinstance(m_, nn.Linear)]
        C = _lins[-1].out_features if _lins else _first.out_features
        self.detect_head = nn.Conv3d(C, 1, 1)
        self._dk, self.detect_out = None, None

        def _dhook(mod, inp, out, _self=self):
            if _self._dk is None:
                return
            if _self._dk == 0:
                _self.detect_out = _self.detect_head(out).squeeze(1)
            _self._dk += 1

        self.net.decoder.stages[-1].register_forward_hook(_dhook)

    def set_pos(self, pos):
        """pos: (B, n_tokens, dim) -> (n_tokens, B, dim); None falls back to the fixed grid."""
        if self.anat_pos:
            self._pos = None if pos is None else pos.permute(1, 0, 2).contiguous()

    def forward(self, img, txt, cat, use_cat):
        # dropped samples add exactly zero -- no fabricated "unknown" category
        cond = txt + self.cat_embed(cat)[:, :, None, :] * use_cat[:, :, None, None]
        if self.region_aux:
            self._rcond = self.region_proj(cond.squeeze(2))     # (B, N, C_final)
            self._rk, self._rlogits = 0, []
        if self.detect_aux:
            self._dk, self.detect_out = 0, None
        out = self.net(img, cond)
        if self.detect_aux:
            self._dk = None
        if self.region_aux:
            self.region_out = torch.stack(self._rlogits, 1) if self._rlogits else None
            self._rcond = None
        return out


def _npy_candidates(img):
    """Where this scan's .norm.npy might live, node-local first.

    np.load is tried directly rather than os.path.exists-then-load: one syscall instead of
    two, and on a partially-rsynced tree the exists/load race is real.
    """
    out = [img + ".norm.npy"]
    if NPY_FALLBACK and img.startswith(NPY_SRC_ROOT):
        out.append(img.replace(NPY_SRC_ROOT, NPY_FALLBACK, 1) + ".norm.npy")
    return out


class PromptPatchesFull(Dataset):
    """One item = (CT+anatomy patch, N target masks, N text embeddings, N category ids, valid)."""

    def __init__(self, rows, text2vec, patch, n_items, fg_prob=0.85, n_pos=2, cat_dropout=0.0,
                 n_crops=8, pair_prob=0.7, require_same_cat=False, use_anat=True,
                 lobe2vecs=None, lobe_prob=0.0, fg_jitter=0.0, prefer_same_cat=False,
                 cat_balance=0.0, cat_boost=None,
                 anat_pos=0, tn=False, region_aux=False,
                 aug_map=None, aug_prob=0.0, paste=None, aug_exclusive=False,
                 lat_map=None, lat_prob=0.0,
                 pos_dim=2048, pos_grid=12, pos_stride=4):
        self.by_case = {}
        for r in rows:
            self.by_case.setdefault(r["case"], []).append(r)
        self.cases = sorted(self.by_case)
        # category-balanced CASE sampling (stage-1 'presence oversampling' revived, §11f):
        # weight a case by the mean rarity of its findings' categories, rarity = freq^-alpha.
        self.case_p = None
        if cat_balance > 0 or cat_boost:
            from collections import Counter
            cfreq = Counter(r["category"] for r in rows)
            n_all = sum(cfreq.values())
            if cat_balance > 0:
                w = np.array([np.mean([(cfreq[r["category"]] / n_all) ** -cat_balance
                                       for r in self.by_case[c]]) for c in self.cases])
            else:
                w = np.ones(len(self.cases), np.float64)
            # cat_boost {"2d": 1.5}: multiply the weight of any case CONTAINING that
            # category. Targeted oversampling, not rarity-based: the catbal probe showed
            # rarity-weighting (freq^-alpha) helps only 2d -- which it DOWN-weights, being
            # the most common class -- while dragging the rare tail (1b -0.047, 2e -0.077).
            # This tests the surviving hypothesis directly: more nodule exposure, nothing else.
            if cat_boost:
                for i_, c_ in enumerate(self.cases):
                    have = {r["category"] for r in self.by_case[c_]}
                    for k_, v_ in cat_boost.items():
                        if k_ in have:
                            w[i_] *= v_
            self.case_p = (w / w.sum()).astype(np.float64)
            self.case_p /= self.case_p.sum()
        self.text2vec, self.patch = text2vec, patch
        self.fg_prob, self.n_pos, self.cat_dropout = fg_prob, n_pos, cat_dropout
        self.region_aux = region_aux
        # text augmentation: aug_map[text] -> list of dicts(text, w, caution); swap probability
        # aug_prob, halved-to-0.2-capped for caution groups (near-synonyms), per the spec's rules.
        self.aug_map, self.aug_prob = (aug_map or {}), aug_prob
        self.aug_exclusive = aug_exclusive
        self._crop_pasted = False
        self.lat_map, self.lat_prob = (lat_map or {}), lat_prob
        # lesion copy-paste: paste=(p_add, p_dis) or None. Bank loaded lazily PER WORKER.
        self.paste = paste
        self._paste_bank = None
        # Payload dtype. The loader shipped 679 MB of float32 per item (8x1x192^3 image +
        # 8x2x192^3 masks) through the worker->main shm queue -- 170 GB per epoch -- and the
        # H100 then sat idle 55% of the time waiting for it. The CT is fp16 in memory anyway
        # and the masks are bool, so ship them as-is (170 MB) and promote on the GPU, where it
        # is free. copy-paste blends soft values into both, so that arm keeps float32.
        self._payload_f32 = bool(paste)
        self._fg_idx = {}                # per-case foreground-voxel cache, see _fg_coords
        # Amortise the volume read over n_crops patches. (The '~13 s decompression' this
        # comment used to cite was WRONG -- measured reads are ~1.1 s -- but amortising
        # still matters: the 1-GPU configs sit at 33% GPU utilisation, loader-bound.)
        self.n_crops = n_crops
        self.pair_prob = pair_prob   # fraction of crops that must contain TWO findings
        self.require_same_cat = require_same_cat   # ALSO resamples the case (biases data)
        self.prefer_same_cat = prefer_same_cat     # pair selection only, all cases kept
        self.use_anat = use_anat                   # False -> 1-channel, like pretrained
        self.anat_pos, self.pos_dim = anat_pos, pos_dim
        self.pos_grid, self.pos_stride = pos_grid, pos_stride
        self._fields, self._out_pos = None, []
        self.n_items = n_items
        # LOCATION SUPERVISION from dense TotalSeg lobes (see _lobe_crop). lobe2vecs maps a
        # lobe id -> list of embedded phrasings; lobe_prob is the fraction of crops that become
        # lobe-aux crops; fg_jitter offsets finding-centred crops so other lobes stay in view.
        self.lobe2vecs = {int(k): v for k, v in (lobe2vecs or {}).items()}
        self.lobe_prob = lobe_prob
        self.fg_jitter = fg_jitter
        # TRUE-NEGATIVE restriction supervision (rules-legal spatial constraint, GT-derived).
        # For a FOCAL finding (GT sits in 1-2 TS lobes), penalise predicted probability in every
        # OTHER lung lobe -- the finding cannot be there. Purely from GT + TotalSeg, no text parse,
        # so nothing structured reaches the model. Diffuse (>=3 lobes) / extra-pulmonary findings
        # are exempt (self._tn_allowed[fidx] absent) so the term never fights the Dice loss.
        self.tn = tn
        self._tn_allowed = {}

    def _precompute_tn(self, masks, ts_full):
        """Per finding: the set of TS lobes its GT occupies, if the finding is FOCAL and pulmonary.
        Absent -> that finding contributes no TN penalty (diffuse / outside lung / empty)."""
        self._tn_allowed = {}
        if not self.tn or ts_full is None:
            return
        for fidx, g in masks.items():
            tot = int(g.sum())
            if tot == 0:
                continue
            labs = ts_full[g]                                 # TS label at each GT voxel
            if int(np.isin(labs, LOBES).sum()) < 0.5 * tot:   # mostly outside lung -> exempt
                continue
            allowed = [L for L in LOBES if int((labs == L).sum()) >= 0.05 * tot]
            if not allowed or len(allowed) >= 3:              # empty or diffuse -> exempt
                continue
            self._tn_allowed[fidx] = set(allowed)

    def _tn_mask(self, fidx, ts_full, sl, shape):
        """Disallowed (must-be-background) voxels for one pick, in the crop frame: lung lobes the
        finding's GT does NOT occupy. Zeros if the finding is exempt or TS is unavailable."""
        al = self._tn_allowed.get(fidx)
        if al is None or ts_full is None:
            return np.zeros(shape, np.float32)
        tsc = ts_full[sl]
        return (np.isin(tsc, LOBES) & ~np.isin(tsc, list(al))).astype(np.float32)

    def __len__(self):
        return self.n_items

    @staticmethod
    def _mask(seg_path, fidx, union=None):
        """One finding's mask, or the UNION of several.

        `union` is a list of seg channels to OR together. It exists for the text-prompted
        CATEGORY stage (build_stage1_index.py), where one prompt -- "pulmonary nodules or masses"
        -- names every finding of that category in the scan, not one of them. Reading channels
        individually and OR-ing is cheaper than materialising the whole 4-D array, which for a
        13-finding case is several GB per worker.
        """
        obj = nib.load(seg_path).dataobj
        if union:
            out = None
            for k in union:
                a = np.asanyarray(obj[k]) > 0 if len(obj.shape) == 4 else np.asanyarray(obj) > 0
                out = a if out is None else (out | a)
            return out.astype(np.bool_)
        arr = obj[fidx] if len(obj.shape) == 4 else np.asanyarray(obj)
        return (np.asanyarray(arr) > 0).astype(np.bool_)      # bool: 8x smaller than float32

    def __getitem__(self, _):
        """Load ONE case, then emit n_crops independent patches from it.

        Memory is the binding constraint here (an overnight run was OOM-killed): a case can
        have 13 findings, and holding every mask channel as float32 was ~14 GB in a single
        worker, x18 workers. So masks are kept as bool, the CT as float16, and TotalSeg as
        uint8 -- only the tiny 192^3 crops are promoted to float32.
        """
        if getattr(self, "case_p", None) is not None:
            rows = self.by_case[self.cases[np.random.choice(len(self.cases), p=self.case_p)]]
        else:
            rows = self.by_case[self.cases[np.random.randint(len(self.cases))]]
        # `prefer_same_cat` asks the PAIR SELECTOR for same-category pairs but does NOT resample
        # the case. `require_same_cat` also resamples, which restricts training to the 35% of
        # cases that contain such a pair -- 57% of findings -- and would confound any comparison
        # against a control trained on all of them.
        if self.require_same_cat:        # resample until the case actually offers such a pair
            for _try in range(20):
                cnt = {}
                for r in rows:
                    cnt[r["category"]] = cnt.get(r["category"], 0) + 1
                if any(v > 1 for v in cnt.values()):
                    break
                rows = self.by_case[self.cases[np.random.randint(len(self.cases))]]
        # mmap, not a full read: an item only ever touches n_crops x 192^3, so this pulls
        # ~113 MB of pages instead of materialising the whole ~180 MB volume -- and it keeps
        # the decompressed working set out of the cgroup, which at --mem=64G was evicting the
        # page cache (workingset_refault_file ~5.2 M pages).
        vol = None
        for _cand in _npy_candidates(rows[0]["image"]):
            try:
                vol = np.load(_cand, mmap_mode="r"); break
            except OSError:
                continue                  # not staged (or a torn write): fall through
        if vol is None:
            vol = zscore(np.asanyarray(nib.load(rows[0]["image"]).dataobj).astype(np.float32)).astype(np.float16)
        ph, pw, pd = self.patch
        pad = [(0, max(0, ph - vol.shape[0])), (0, max(0, pw - vol.shape[1])),
               (0, max(0, pd - vol.shape[2]))]
        # np.pad COPIES even when every pad width is zero, which is the normal case here
        # (512x512xD with D >= 192). Skip it, or the mmap above is immediately materialised.
        _needs_pad = any(b for _a, b in pad)
        if _needs_pad:
            vol = np.pad(vol, pad)

        # Only READ TotalSeg if something actually consumes it. With --no-anat (every run we
        # have done) the anat channels are built and then discarded by `if self.use_anat`, so a
        # full ~120 MB volume was being pulled over NFS per sample for nothing -- which is the
        # loader starvation measured at 33% GPU utilisation.
        need_ts = bool(self.use_anat or self.lobe_prob > 0 or self.anat_pos or self.tn
                       or self.paste)
        p = ts_path(rows[0]["case"])
        ts_full = (np.pad(np.asanyarray(nib.load(str(p)).dataobj).astype(np.uint8), pad)
                   if (need_ts and p.exists()) else None)
        masks = {r["fidx"]: (np.pad(m_, pad) if _needs_pad else m_)
                 for r in rows
                 for m_ in (self._mask(r["seg"], r["fidx"], r.get("union")),)}

        # Region-prior aux targets (region_prior/lung_regions.py): soft [0,1] volumes parsed
        # from the finding TEXT, which is complete even where train masks are capped at <=3
        # instances -- full-coverage aux supervision. The generator returns (nk,nj,ni); our
        # volumes are nibabel (ni,nj,nk), so transpose(2,1,0) -- verified Dice 1.0000 against
        # the nibabel-loaded TS lung mask before any training used it.
        reg = None
        if self.region_aux:
            try:
                from lung_regions import RegionGenerator
                p_ = ts_path(rows[0]["case"])
                if p_.exists():
                    g_ = RegionGenerator(rows[0]["image"], str(p_))
                    reg = {r["text"]: np.pad(
                        g_.target_for_text(r["text"]).transpose(2, 1, 0).astype(np.float16), pad)
                        for r in rows}
            except Exception:
                reg = None                      # loss is masked out via rval, never faked
        self._precompute_tn(masks, ts_full)       # focal-finding lobe sets for TN restriction
        cents = self._centroids(rows, masks)      # once per case, not once per crop
        self._fg_idx = {}                         # ditto for the foreground voxel lists
        # Anatomy-relative coordinate fields: three distance transforms, so ONCE per case at
        # stride `pos_stride`, then indexed per crop. Per-crop would be seconds each and the
        # loader already starves the GPU (measured 33% utilisation on the 1-GPU configs).
        self._fields = None
        if self.anat_pos and ts_full is not None:
            try:
                from anat_pos import anat_fields
                self._fields, _ = anat_fields(ts_full, nib.load(rows[0]["image"]).affine,
                                              stride=self.pos_stride)
            except Exception:
                self._fields = None
        self._out_pos = []
        out_img, out_tgt, out_txt, out_cat, out_use, out_val = [], [], [], [], [], []
        out_tn, out_cap = [], []
        out_reg, out_rval = [], []
        for _crop in range(self.n_crops):
            self._one_crop(rows, vol, ts_full, masks, cents, ph, pw, pd,
                           out_img, out_tgt, out_txt, out_cat, out_use, out_val, out_tn, out_cap,
                           reg, out_reg, out_rval)
        _img_t = torch.from_numpy(np.stack(out_img))      # fp16 unless copy-paste is on
        _tgt_t = torch.from_numpy(np.stack(out_tgt))      # bool  unless copy-paste is on
        if self._payload_f32:
            _img_t, _tgt_t = _img_t.float(), _tgt_t.float()
        return (_img_t,
                _tgt_t,
                torch.from_numpy(np.stack(out_txt)).float(),
                torch.tensor(np.stack(out_cat), dtype=torch.long),
                torch.tensor(np.stack(out_use), dtype=torch.float32),
                torch.from_numpy(np.stack(out_val)),
                (torch.from_numpy(np.stack(self._out_pos)).float() if self._out_pos
                 else torch.zeros(len(out_img), 1, 1)),
                torch.from_numpy(np.stack(out_tn)).float(),
                torch.from_numpy(np.stack(out_cap)).float(),
                (torch.from_numpy(np.stack(out_reg)).float() if self.region_aux
                 else torch.zeros(len(out_img), self.n_pos, 1, 1, 1)),
                (torch.from_numpy(np.stack(out_rval)).float() if self.region_aux
                 else torch.zeros(len(out_img), self.n_pos)))

    def _append_pos(self, y, x, z, ph):
        """Positional encoding for this crop's 12^3 bottleneck tokens, in PATIENT coordinates.

        Falls back to zeros when the fields are unavailable (no TotalSeg, no lung) -- the model
        then sees an uninformative but correctly-shaped signal rather than crashing.
        """
        g, d = self.pos_grid, self.pos_dim
        if self._fields is None:
            self._out_pos.append(np.zeros((g ** 3, d), np.float32)); return
        from anat_pos import token_pos
        self._out_pos.append(token_pos(self._fields, self.pos_stride, (y, x, z), ph, g, d))

    @staticmethod
    def _centroid(m):
        """Centroid without materialising the index array: mean position per axis from
        1-D marginal sums, which is ~an order of magnitude cheaper than np.argwhere."""
        n = int(m.sum())
        if n == 0:
            return None
        out = []
        for ax in range(3):
            marg = m.sum(axis=tuple(a for a in range(3) if a != ax))
            out.append(float(np.dot(marg, np.arange(m.shape[ax]))) / n)
        return np.array(out)

    def _centroids(self, rows, masks):
        """One centroid per finding for this case, computed once (see _pair_origin)."""
        return {r["fidx"]: self._centroid(masks[r["fidx"]]) for r in rows}

    def _fg_coords(self, fidx, m):
        """Foreground voxel coordinates for one finding, computed ONCE per case.

        This is the same bug _pair_origin fixed for the centroids, still live on the
        fg-anchor path: `np.argwhere(m > 0)` scans the whole 80-130 M-voxel mask AND
        allocates a second full-size bool temp for `m > 0`. Measured 0.70 s per call, and
        with `fg_prob x n_crops` it fired ~2.6 times per item -- ~1.8 s of a ~5 s loader
        budget, on a pipeline that already leaves the GPU idle. flatnonzero is one pass
        returning one small int64 array (the masks hold 200-50k voxels), and the result is
        reused across every crop of the case. None means the finding is empty.
        """
        if fidx not in self._fg_idx:
            flat = np.flatnonzero(m)
            self._fg_idx[fidx] = (np.stack(np.unravel_index(flat, m.shape), 1)
                                  if flat.size else None)
        return self._fg_idx[fidx]

    def _pair_origin(self, rows, cents, ph, pw, pd, shape, require_same=False):
        """Origin of a crop containing TWO findings, or None if no pair fits.

        This is the whole point of the sampler. Centring every crop on its target teaches the
        model "segment the abnormal thing in the middle", which is solvable from the image
        alone -- the prompt is then redundant, and the model provably ignored it (swapping the
        lobe in the text left the mask pixel-identical, textsens Dice(A,B)=0.999, both with and
        without category dropout). If instead TWO findings sit inside the same crop, "the middle
        thing" is ambiguous and the ONLY way to be right is to read which one the text names.
        """
        H, W, D = shape
        # Centroids are computed ONCE per case and cached. `np.argwhere` scans the whole
        # ~50M-voxel mask, so calling it inside the pair loop meant O(n^2) full-volume scans
        # per crop, x n_crops -- which made the sampler slower than the network.
        cand = [r for r in rows if cents.get(r["fidx"]) is not None]
        if len(cand) < 2:
            return None
        # Enumerate candidate pairs, SAME-CATEGORY FIRST. Measurement showed the old sampler
        # already put two findings in 44% of patches, yet the model still ignored the text --
        # because those pairs almost always had DIFFERENT categories, and the category input
        # alone then identifies which finding is meant. Only when both findings share a
        # category is the location in the text the sole discriminator. That is also exactly
        # the ~23% of test findings the metric rewards.
        order = np.random.permutation(len(cand))
        pairs = [(cand[order[i]], cand[order[j]])
                 for i in range(len(order)) for j in range(i + 1, len(order))]
        if require_same:      # ONLY same-category pairs: location is the sole discriminator
            pairs = [ab for ab in pairs if ab[0]["category"] == ab[1]["category"]]
        else:
            pairs.sort(key=lambda ab: ab[0]["category"] != ab[1]["category"])   # same-cat first
        for a, b in pairs:
                ca, cb = cents[a["fidx"]], cents[b["fidx"]]
                if (abs(ca - cb) < np.array([ph, pw, pd]) * 0.8).all():   # both fit in one patch
                    mid = (ca + cb) / 2
                    y = int(np.clip(mid[0] - ph // 2, 0, max(0, H - ph)))
                    x = int(np.clip(mid[1] - pw // 2, 0, max(0, W - pw)))
                    z = int(np.clip(mid[2] - pd // 2, 0, max(0, D - pd)))
                    return (y, x, z), [a, b]
        return None

    def _one_crop(self, rows, vol, ts_full, masks, cents, ph, pw, pd,
                  out_img, out_tgt, out_txt, out_cat, out_use, out_val, out_tn, out_cap,
                  reg=None, out_reg=None, out_rval=None):
        H, W, D = vol.shape
        if (self.lobe_prob > 0 and ts_full is not None and self.lobe2vecs
                and np.random.rand() < self.lobe_prob):
            if self._lobe_crop(rows, vol, ts_full, cents, ph, pw, pd,
                               out_img, out_tgt, out_txt, out_cat, out_use, out_val, out_tn):
                out_cap.append(np.zeros(self.n_pos, np.float32))   # synthetic lobe crops: not capped
                if self.region_aux:                # keep every parallel list the same length
                    out_reg.append(np.zeros((self.n_pos, ph, pw, pd), np.float16))
                    out_rval.append(np.zeros(self.n_pos, np.float32))
                return
            # crop offered no lobe to teach from -> fall through to a normal finding crop
        self._crop_pasted = False
        forced = None
        if np.random.rand() < self.pair_prob:            # try to get 2 findings in one crop
            forced = self._pair_origin(rows, cents, ph, pw, pd, vol.shape,
                                       require_same=self.require_same_cat
                                       or self.prefer_same_cat)
            if forced is None and self.prefer_same_cat:
                forced = self._pair_origin(rows, cents, ph, pw, pd, vol.shape,
                                           require_same=False)   # fall back

        if forced is not None:
            (y, x, z), pair = forced
            anchor = pair[0]
        else:
            anchor = rows[np.random.randint(len(rows))]
            pair = None
            m_anchor = masks[anchor["fidx"]]
            # NB: the rand() draw stays outside so the RNG stream is unchanged vs the
            # pre-cache version -- resumed runs see the same crop sequence.
            fg = (self._fg_coords(anchor["fidx"], m_anchor)
                  if np.random.rand() < self.fg_prob else None)
            if fg is not None:
                c = fg[np.random.randint(len(fg))].astype(np.float64)
                if self.fg_jitter:            # push the finding off-centre so other lobes show
                    c = c + (np.random.rand(3) - 0.5) * 2 * self.fg_jitter * np.array([ph, pw, pd])
                y = int(np.clip(c[0] - ph // 2, 0, H - ph)); x = int(np.clip(c[1] - pw // 2, 0, W - pw))
                z = int(np.clip(c[2] - pd // 2, 0, D - pd))
            else:
                y, x, z = (np.random.randint(0, H - ph + 1), np.random.randint(0, W - pw + 1),
                           np.random.randint(0, D - pd + 1))
        m_anchor = masks[anchor["fidx"]]
        sl = (slice(y, y + ph), slice(x, x + pw), slice(z, z + pd))

        # anatomy channels: 5 lung lobes, zeros if TotalSeg is unavailable for this case.
        # Built ONLY when they are used -- previously they were stacked and then discarded.
        if self.use_anat:
            if ts_full is not None:
                anat = np.stack([(ts_full[sl] == l).astype(np.float32) for l in LOBES])
            else:
                anat = np.zeros((len(LOBES), ph, pw, pd), np.float32)
            img = np.concatenate([vol[sl][None], anat], 0)      # (6,...)
        else:
            img = vol[sl][None].astype(np.float32) if self._payload_f32 else vol[sl][None]

        if pair is not None:                     # both of these are inside this crop
            picks = list(pair)[: self.n_pos]
        else:
            others = [r for r in rows if r is not anchor]
            np.random.shuffle(others)
            picks = [anchor] + others[: self.n_pos - 1]
        targets = [m_anchor[sl]] + [masks[r["fidx"]][sl] for r in picks[1:]]
        # counterfactual laterality: flip the word, empty the answer. Applied BEFORE paste so a
        # pasted ADD can never land on a flipped prompt (its named region changed sides).
        self._lat_flipped = [False] * len(picks)
        if self.lat_prob > 0:
            for j_, r_ in enumerate(picks):
                if (r_["text"] in self.lat_map and np.random.rand() < self.lat_prob):
                    self._lat_flipped[j_] = True
                    targets[j_] = np.zeros_like(targets[j_])
        if self.paste and ts_full is not None:
            # img/targets are VIEWS into the case-level caches; pasting in place would corrupt
            # every later crop of this case. Copy first, always.
            img = img.copy()
            targets = [t.copy() for t in targets]      # _payload_f32 is True whenever paste is
            from copy_paste import PasteBank, apply_copy_paste
            if self._paste_bank is None:
                try:
                    self._paste_bank = PasteBank()
                except Exception:
                    self._paste_bank = False           # bank missing: disable quietly, never crash
            if self._paste_bank:
                rng = np.random.default_rng(np.random.randint(2**31))
                _pk = [r_ for j_, r_ in enumerate(picks) if not self._lat_flipped[j_]]
                na_, nd_ = apply_copy_paste(img, targets, _pk, len(_pk), ts_full[sl],
                                            self._paste_bank, rng, self.paste[0], self.paste[1])
                # exclusive mode: a crop that received a paste gets NO text swap. The combo arm
                # showed joint corruption of both ends of the text<->image mapping is destructive
                # (0.3112 vs control 0.3162); this removes exactly the doubly-corrupted samples.
                self._crop_pasted = bool(na_ or nd_)
        def _pick_text(r, _flip=False):
            if _flip:
                return self.text2vec[self.lat_map[r["text"]]]
            t = r["text"]
            if (self.aug_exclusive and getattr(self, "_crop_pasted", False)):
                return self.text2vec[t]
            if self.aug_prob > 0 and t in self.aug_map and np.random.rand() < self.aug_prob:
                vs = [v for v in self.aug_map[t] if v["text"] in self.text2vec]
                vs = [v for v in vs if (not v["caution"]) or np.random.rand() < 0.2 / max(self.aug_prob, 0.2)]
                if vs:
                    w = np.array([v["w"] for v in vs], np.float64); w /= w.sum()
                    return self.text2vec[vs[np.random.choice(len(vs), p=w)]["text"]]
            return self.text2vec[t]
        texts = [_pick_text(r, _flip=self._lat_flipped[j_]) for j_, r in enumerate(picks)]
        cats, use_cat = [], []
        for r in picks:
            cats.append(CAT2IDX[r["category"]])   # every finding has one; KeyError is a real bug
            use_cat.append(0.0 if np.random.rand() < self.cat_dropout else 1.0)

        valid = np.zeros(self.n_pos, np.float32)
        valid[: len(picks)] = 1.0
        while len(targets) < self.n_pos:
            targets.append(np.zeros_like(targets[0]))
            texts.append(texts[0]); cats.append(0); use_cat.append(0.0)

        out_img.append(img); out_tgt.append(np.stack(targets))
        if self.anat_pos:
            self._append_pos(y, x, z, ph)
        out_txt.append(np.stack(texts)[:, None]); out_cat.append(np.array(cats))
        out_use.append(np.array(use_cat, np.float32)); out_val.append(valid)
        cap = np.zeros(self.n_pos, np.float32)
        for i_, r_ in enumerate(picks[: self.n_pos]):
            cap[i_] = 1.0 if (r_.get("entity_count") or 1) == 3 else 0.0
        out_cap.append(cap)
        if self.region_aux:
            rt, rv = [], []
            for r in picks[: self.n_pos]:
                t_ = reg.get(r["text"]) if reg else None
                if t_ is None:
                    rt.append(np.zeros((ph, pw, pd), np.float16)); rv.append(0.0)
                else:
                    rt.append(t_[sl]); rv.append(1.0)
            while len(rt) < self.n_pos:
                rt.append(np.zeros((ph, pw, pd), np.float16)); rv.append(0.0)
            out_reg.append(np.stack(rt)); out_rval.append(np.array(rv, np.float32))
        if self.tn:                                # disallowed lung-lobe mask per pick
            tn_masks = [self._tn_mask(r["fidx"], ts_full, sl, (ph, pw, pd)) for r in picks]
            while len(tn_masks) < self.n_pos:
                tn_masks.append(np.zeros((ph, pw, pd), np.float32))
            out_tn.append(np.stack(tn_masks))
        else:
            out_tn.append(np.zeros((self.n_pos, 1, 1, 1), np.float32))

    def _lobe_crop(self, rows, vol, ts_full, cents, ph, pw, pd,
                   out_img, out_tgt, out_txt, out_cat, out_use, out_val, out_tn):
        """Location-supervision patch from DENSE TotalSeg lobes. SAFE on a sparsely annotated
        dataset because TotalSeg labels every lobe voxel, so a lobe absent from the crop is
        genuinely absent (unlike an un-annotated pathology, which is why fabricated pathology
        negatives are banned -- see the module docstring). Emits one PRESENT lobe (prompt ->
        its mask) and one ABSENT lobe (prompt -> empty): same image, different lobe words,
        different required output -- exactly what textsens measures, taught directly.

        1-channel only: with the anatomy channels on, the target lobe IS an input channel and
        the model would trivially copy it (main asserts --no-anat when lobe_prob>0). Returns
        False (skip, fall back to a finding crop) if the sampled crop holds no positive lobe.
        """
        H, W, D = vol.shape
        valid_cents = [c for c in cents.values() if c is not None]
        if valid_cents and np.random.rand() < self.fg_prob:       # centre on lung, with jitter
            c = valid_cents[np.random.randint(len(valid_cents))].astype(np.float64)
            c = c + (np.random.rand(3) - 0.5) * 2 * max(self.fg_jitter, 0.2) * np.array([ph, pw, pd])
            y = int(np.clip(c[0] - ph // 2, 0, H - ph)); x = int(np.clip(c[1] - pw // 2, 0, W - pw))
            z = int(np.clip(c[2] - pd // 2, 0, D - pd))
        else:
            y, x, z = (np.random.randint(0, H - ph + 1), np.random.randint(0, W - pw + 1),
                       np.random.randint(0, D - pd + 1))
        sl = (slice(y, y + ph), slice(x, x + pw), slice(z, z + pd))
        ts = ts_full[sl]
        counts = {l: int((ts == l).sum()) for l in LOBES}
        present = [l for l in LOBES if counts[l] >= 200]      # enough voxels to be a real target
        absent = [l for l in LOBES if counts[l] == 0]
        if not present:                                       # no positive lobe here -> skip
            return False
        picks = [present[np.random.randint(len(present))]]
        if absent:
            picks.append(absent[np.random.randint(len(absent))])  # the hard negative
        while len(picks) < self.n_pos:
            picks.append(present[np.random.randint(len(present))])
        picks = picks[: self.n_pos]

        img = (vol[sl][None].astype(np.float32) if self._payload_f32
               else vol[sl][None])                            # 1-channel (guarded to --no-anat)
        targets, texts, cats, use_cat = [], [], [], []
        for l in picks:
            targets.append((ts == l).astype(np.bool_))        # empty for an absent lobe
            vecs = self.lobe2vecs[l]
            texts.append(vecs[np.random.randint(len(vecs))])
            cats.append(0); use_cat.append(0.0)               # anatomical prompt: no category
        valid = np.zeros(self.n_pos, np.float32); valid[: len(picks)] = 1.0
        out_img.append(img); out_tgt.append(np.stack(targets))
        if self.anat_pos:
            self._append_pos(y, x, z, ph)
        out_txt.append(np.stack(texts)[:, None]); out_cat.append(np.array(cats))
        out_use.append(np.array(use_cat, np.float32)); out_val.append(valid)
        out_tn.append(np.zeros((self.n_pos, ph, pw, pd) if self.tn else (self.n_pos, 1, 1, 1),
                               np.float32))            # no TN on synthetic lobe crops
        return True


class _MixDS(Dataset):
    """Draw from the real index with prob (1-p) and the CT-RATE aux pool with prob p.

    Both members emit the same tuple shape, so the collate and training loop are unchanged.
    An aux failure (all 12 resamples empty) falls back to a real item rather than killing the
    epoch -- pseudo-label rows are individually unreliable by nature.
    """

    def __init__(self, real, aux, p, n_items):
        self.real, self.aux, self.p, self.n_items = real, aux, float(p), n_items

    def __len__(self):
        return self.n_items

    def __getitem__(self, i):
        if np.random.rand() < self.p:
            try:
                return self.aux[i]
            except Exception:
                pass
        return self.real[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--iters", type=int, default=250, help="volume READS per epoch per rank (x n_crops patches)")
    ap.add_argument("--batch", type=int, default=1, help="cases (volume reads) per step per rank")
    ap.add_argument("--micro", type=int, default=2, help="patches per forward (memory bound)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--optim", choices=["sgd", "adamw"], default="sgd")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-crops", type=int, default=8, help="patches per volume read")
    ap.add_argument("--resume", default="", help="checkpoint to continue from")
    ap.add_argument("--text-film", default="",
                    help="comma-separated encoder stages whose SKIPS get per-prompt FiLM "
                         "conditioning, e.g. '0,1,2,3'. Text currently reaches only the 12^3 "
                         "bottleneck, so full-resolution features are prompt-blind. Zero-init, "
                         "so the model starts numerically identical and can be warm-started.")
    ap.add_argument("--lat-swap", type=float, default=0.0,
                    help="counterfactual laterality: with this probability, a one-sided prompt's "
                         "text is swapped left<->right and its TARGET SET EMPTY -- 'the flipped "
                         "word means NOT here'. The text-side mirror of copypaste's distractor.")
    ap.add_argument("--boundary-weight", type=float, default=0.0,
                    help="extra BCE weight on the GT boundary band at the finest scale. Targets "
                         "the 31%% of residual FP mass within 5mm of GT (fp census) that no "
                         "region or augmentation lever reaches.")
    ap.add_argument("--aug-exclusive", action="store_true",
                    help="never apply text-aug and copy-paste to the SAME crop (removes the "
                         "doubly-corrupted 25%% that made the joint combo destructive)")
    ap.add_argument("--copy-paste", default="",
                    help="'p_add,p_dis' -- lesion copy-paste augmentation (copy_paste.py): p_add "
                         "= chance of adding same-category instances to a plural-safe prompt's "
                         "target (counters the 3-instance annotation cap); p_dis = chance of a "
                         "text-consistency-safe distractor lesion no target includes (attacks "
                         "precision). Needs paste_bank/ (build_lesion_bank.py).")
    ap.add_argument("--text-aug", type=float, default=0.0,
                    help="probability of swapping a finding text for an attested in-distribution "
                         "variant (region_prior/augment_synonyms.json). Variants embedded in "
                         "augment_bank/; caution groups capped at 0.2 regardless.")
    ap.add_argument("--detect-aux", type=float, default=0.0,
                    help="weight of prompt-independent 'anything here' voxel BCE on the union "
                         "of annotated findings; 0 disables")
    ap.add_argument("--detect-neg", type=float, default=0.1,
                    help="relative BCE weight on union-BACKGROUND voxels (unannotated pathology "
                         "makes background uncertain; keep low)")
    ap.add_argument("--region-aux", type=float, default=0.0,
                    help="weight of the auxiliary region-prediction loss (0 = off). Targets are "
                         "soft anatomical regions parsed from the finding text "
                         "(region_prior/lung_regions.py); BCE at the finest scale. Replaces the "
                         "hard lobe-aux gating -- a hard lobe clip keeps only 77% of finding "
                         "mass on complete val.")
    ap.add_argument("--attn-gate", default="",
                    help="comma-separated encoder stages whose skips get a per-VOXEL spatial gate "
                         "driven by the transformer's 12^3 cross-attention, upsampled. Unlike "
                         "--text-film (and the decoder's own mask embeddings) this is spatially "
                         "SELECTIVE, which is the capability EXPERIMENTS.md 4o-1 identifies as "
                         "missing. Zero-init gain, so identity at init.")
    ap.add_argument("--film-lr-mult", type=float, default=10.0,
                    help="LR multiplier for the FiLM heads: they start at exactly zero, so at the "
                         "base 1e-4 SGD they would barely move in a 60-epoch fine-tune and the "
                         "run would report a false null.")
    ap.add_argument("--fusion-stage", type=int, default=-1,
                    help="encoder stage for text-image fusion (-1 = leave at the pretrained 4). "
                         "3 gives a 24^3 grid instead of 12^3, halving token size to ~5.5mm so "
                         "the text can point at a lesion rather than a neighbourhood.")
    ap.add_argument("--fusion-lr-mult", type=float, default=10.0,
                    help="LR multiplier for the parameters --fusion-stage creates. The reseeded "
                         "projection starts from a scale-preserving guess, not a correct mapping; "
                         "at the base 1e-4 SGD it may not move in 60 epochs and a null result "
                         "would then mean 'did not move', not 'did not help'.")
    ap.add_argument("--save-every", type=int, default=1,
                    help="write the checkpoint every N epochs (still atomic; last epoch always saved)")
    ap.add_argument("--cat-balance", type=float, default=0.0,
                    help="alpha for category-rarity case oversampling (0=uniform; 0.5=sqrt inverse freq)")
    ap.add_argument("--cat-boost", default="",
                    help="targeted case oversampling, 'CAT:MULT' e.g. '2d:1.5' (space-separate "
                         "pairs for several). NB sbatch --export eats commas; pass on the "
                         "command line, never via env.")
    ap.add_argument("--fg-prob", type=float, default=0.85,
                    help="fraction of patches centred on a finding (nnU-Net standard ~0.33)")
    ap.add_argument("--pair-prob", type=float, default=0.7,
                    help="fraction of crops forced to contain TWO findings")
    ap.add_argument("--index", default="t1_index_train.jsonl", help="training index file")
    ap.add_argument("--bank", default="",
                    help="text bank directory (texts.json + embeddings.npy). Empty = the "
                         "ReXGroundingCT finding bank. The text-prompted CATEGORY stage uses its "
                         "own 14-prompt bank, embedded with the SAME encoder so both stages live "
                         "in one space.")
    ap.add_argument("--lora-rank", type=int, default=0,
                    help="LoRA rank on the text pathway; 0 = full fine-tune (off)")
    ap.add_argument("--enc-from", default="",
                    help="checkpoint whose ENCODER to graft in (e.g. the 0.229 category model)")
    ap.add_argument("--excl-weight", type=float, default=0.0,
                    help="weight of the cross-prompt exclusion penalty (0 = off)")
    ap.add_argument("--no-anat", action="store_true",
                    help="train WITHOUT the 5 TotalSeg lobe channels (1-ch, like pretrained)")
    ap.add_argument("--require-same-cat", action="store_true",
                    help="every pair crop must hold TWO findings of the SAME category")
    ap.add_argument("--cat-dropout", type=float, default=0.0)
    ap.add_argument("--tversky", default="",
                    help="'alpha,beta' -> asymmetric overlap loss; alpha weights FALSE NEGATIVES, "
                         "beta FALSE POSITIVES. alpha>beta forgives predicting outside the "
                         "annotation, which matters because the labels are incomplete (20%% of "
                         "bilateral findings are one-sided, predictions span 2.1x the annotated "
                         "slices). PRIOR IS NEGATIVE: beta=0.9 ran away into over-segmentation "
                         "twice (our 501 encoder, and dHT 'still paints pf 0.979'). Use mild "
                         "values and watch the predicted volume ratio.")
    ap.add_argument("--prefer-same-cat", action="store_true",
                    help="ask for SAME-category pairs in a crop when the case offers one, but do "
                         "NOT resample the case. --require-same-cat resamples, which restricts "
                         "training to the 35%% of cases that contain such a pair (57%% of "
                         "findings) and confounds any comparison with a control trained on all.")
    ap.add_argument("--attn-excl", type=float, default=0.0,
                    help="weight on the ATTENTION-disjointness penalty: two findings on one scan "
                         "must not attend to the same bottleneck tokens. GT pairs are disjoint "
                         "99%% (val) / 92%% (train) of the time while our predictions overlap on "
                         "46%% of pairs; 23%% of predicted volume is contested and assigning it "
                         "correctly is worth +0.046 Dice. Pairs whose GT genuinely overlaps are "
                         "skipped (7.3%% of same-category train pairs).")
    ap.add_argument("--anat-pos", type=int, default=0,
                    help="replace VoxTell's FIXED patch-relative positional grid with a "
                         "per-scan ANATOMY-relative one (distance/elevation/azimuth from the "
                         "nearer hilum, plus distance to pleura). The fixed grid is identical "
                         "for every patch and every scan, so at the point where text attends to "
                         "image the model has no signal for where in the PATIENT it is -- "
                         "'apical' is not representable there. Requires TotalSeg.")
    ap.add_argument("--lobe-prob", type=float, default=0.0,
                    help="fraction of crops that become dense-TotalSeg lobe-localisation crops "
                         "(present lobe -> its mask, absent lobe -> empty); requires --no-anat")
    ap.add_argument("--cap-forgive", type=float, default=0.0,
                    help="beta for prompts whose finding has entity_count==3, i.e. the ones that "
                         "hit the annotators' 3-instance cap and are therefore under-labelled "
                         "(30%% of train findings vs 10%% of val; train has 2.7%% at 4+ entities "
                         "against val's 21%%). Those prompts get alpha=1-beta so extra detections "
                         "are forgiven; every other prompt keeps 0.5/0.5, which is plain Dice. "
                         "0 = off.")
    ap.add_argument("--freeze-enc", action="store_true",
                    help="freeze the (grafted) encoder; train only the decoders. Ablation for "
                         "--enc-from: are the source encoder's features good enough unadapted?")
    ap.add_argument("--tn-weight", type=float, default=0.0,
                    help="TRUE-NEGATIVE restriction loss: penalise predicted probability in lung "
                         "lobes the finding's GT does NOT occupy (focal findings only; diffuse / "
                         "extra-pulmonary exempt). Rules-legal spatial constraint from GT+TotalSeg, "
                         "no text parse reaches the model. 0 = off. Requires TotalSeg.")
    ap.add_argument("--fg-jitter", type=float, default=0.0,
                    help="offset finding-centred crops by up to this fraction of the patch so "
                         "other lobes stay in view (implicit location supervision from GT zeros)")
    ap.add_argument("--aux-index", default="",
                    help="CT-RATE + TotalSegmentator pseudo-label index (aux_bank/index_pilot.jsonl). "
                         "Adds 2d/2e findings from real reports; PSEUDO-labels, so MIX with the "
                         "real data (--aux-prob), never train on them alone.")
    ap.add_argument("--aux-bank", default=str(HERE / "aux_bank"),
                    help="dir with the aux findings' Qwen embeddings (prep_aux_texts.py)")
    ap.add_argument("--aux-dilate", type=int, default=0,
                    help="in-plane dilation of aux masks, to match how annotations are drawn "
                         "(calibrate_convention.py: 2 matches size, 3 matches shape; 0 = off)")
    ap.add_argument("--aux-prob", type=float, default=0.4,
                    help="fraction of items drawn from the aux pool")
    ap.add_argument("--out", default=str(HERE / "VoxTellTrainer_T1_full.pt"))
    a = ap.parse_args()

    ddp = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    dev = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    main_proc = rank == 0

    def log(msg):
        if main_proc:
            print(msg, flush=True)

    rows = [json.loads(l) for l in open(HERE / "index" / a.index)]
    _bank_dir = Path(a.bank) if a.bank else BANK_DIR
    log(f"[bank] {_bank_dir}")
    bank = json.load(open(_bank_dir / "texts.json"))
    emb = np.load(_bank_dir / "embeddings.npy")
    text2vec = {t: emb[i] for i, t in enumerate(bank)}

    lat_map = {}
    if a.lat_swap > 0:
        lat_map = json.load(open(HERE / "region_prior" / "latswap_map.json"))
        lb = HERE / "latswap_bank"
        lt = json.load(open(lb / "texts.json")); le = np.load(lb / "embeddings.npy")
        for t_, v_ in zip(lt, le):
            text2vec.setdefault(t_, v_)
        log(f"[lat-swap] p={a.lat_swap}: {len(lat_map)} one-sided texts flippable")
    aug_map = {}
    if a.text_aug > 0:
        aug_map = json.load(open(HERE / "region_prior" / "augment_variants.json"))
        ab = HERE / "augment_bank"
        atexts = json.load(open(ab / "texts.json")); aemb = np.load(ab / "embeddings.npy")
        for t_, v_ in zip(atexts, aemb):
            text2vec.setdefault(t_, v_)
        n_cov = sum(1 for t_ in aug_map if t_ in {r['text'] for r in rows})
        log(f"[text-aug] p={a.text_aug}: {len(aug_map)} originals have variants "
            f"({n_cov} present in this index), bank +{len(atexts)} embeddings")
    lobe2vecs = {}
    if a.lobe_prob > 0:
        assert a.no_anat, "--lobe-prob requires --no-anat (else the lobe target is an input channel)"
        lb_emb = np.load(HERE / "lobe_bank" / "embeddings.npy")
        lb_ids = json.load(open(HERE / "lobe_bank" / "lobe_ids.json"))
        for lid, vec in zip(lb_ids, lb_emb):
            lobe2vecs.setdefault(int(lid), []).append(vec)
        log(f"[lobe-aux] {sum(len(v) for v in lobe2vecs.values())} phrases across "
            f"{len(lobe2vecs)} lobes; lobe_prob={a.lobe_prob} fg_jitter={a.fg_jitter}")

    n_in = 1 if a.no_anat else 1 + len(LOBES)
    net, patch = build_model(dev, in_channels=n_in, deep_supervision=True)
    if a.enc_from:
        # Graft the ENCODER of an already domain-adapted model (our 0.229 category model, a
        # plain ResEnc-L whose encoder was itself VoxTell-initialised then fine-tuned on THESE
        # findings). Its 448 encoder tensors match VoxTell's promptable encoder exactly, so we
        # get a CT encoder tuned to this data while keeping VoxTell's working text pathway
        # (transformer prompt decoder + mask decoder) intact.
        src = torch.load(a.enc_from, map_location="cpu", weights_only=False)
        src = src.get("network_weights", src)
        enc = {k: v for k, v in src.items() if k.startswith("encoder.")}
        cur = net.state_dict()
        ok = {k: v for k, v in enc.items() if k in cur and cur[k].shape == v.shape}
        net.load_state_dict({**cur, **ok}, strict=False)
        log(f"[enc-from] grafted {len(ok)}/{len(enc)} encoder tensors from {Path(a.enc_from).name}")

    film_stages = tuple(int(x) for x in a.text_film.split(",") if x.strip() != "")
    gate_stages = tuple(int(x) for x in a.attn_gate.split(",") if x.strip() != "")
    if a.region_aux > 0:
        sys.path.insert(0, str(HERE / "region_prior"))       # lung_regions, for the workers too
    model = T1Model(net, len(CODES14), TEXT_DIM, anat_pos=a.anat_pos,
                    attn_excl=a.attn_excl, film_stages=film_stages,
                    gate_stages=gate_stages, region_aux=a.region_aux > 0,
                    detect_aux=a.detect_aux > 0).to(dev)
    film_ids = set()
    if a.detect_aux > 0:
        film_ids |= {id(pm) for n_, pm in model.named_parameters() if n_.startswith("detect_head")}
        log(f"[detect-aux] lambda={a.detect_aux} neg={a.detect_neg} -- fresh 1x1 conv head")
    if a.region_aux > 0:
        film_ids |= {id(pm) for n_, pm in model.named_parameters() if n_.startswith("region_proj")}
        log(f"[region-aux] lambda={a.region_aux} -- "
            f"{sum(pm.numel() for n_, pm in model.named_parameters() if n_.startswith('region_proj'))} "
            f"fresh params at {a.film_lr_mult}x LR")
    if gate_stages:
        film_ids |= {id(pm) for n_, pm in model.named_parameters() if n_.startswith("gate_gain")}
        log(f"[attn-gate] stages {gate_stages} -- per-voxel spatial gate from the 12^3 "
            f"cross-attention, {sum(pm.numel() for n_, pm in model.named_parameters() if n_.startswith('gate_gain'))} "
            f"fresh params at {a.film_lr_mult}x LR")
    if film_stages:
        # zero-init, so these must be flagged fresh or they never leave zero at 1e-4 SGD.
        # NB: merged into fresh_ids AFTER that set is created below, not here.
        film_ids |= {id(pm) for n_, pm in model.named_parameters()
                     if n_.startswith(("film_trunk", "film_heads"))}
        log(f"[text-film] stages {film_stages} channels {model._film_chans} "
            f"-- {sum(pm.numel() for n_, pm in model.named_parameters() if n_.startswith('film'))} "
            f"fresh params at {a.film_lr_mult}x LR")
    if a.freeze_enc:
        # Frozen-encoder ablation: keep the grafted CT encoder fixed, train only the text/mask
        # decoders. Tests whether the source encoder's features ALONE are good enough (vs the
        # warm-start-then-finetune run, which is the fair primary comparison to nograft).
        nf = 0
        for pn, pm in model.net.named_parameters():
            if pn.startswith("encoder.") or ".encoder." in pn:
                pm.requires_grad_(False); nf += 1
        log(f"[freeze-enc] froze {nf} encoder params (decoders train)")
    if a.lora_rank > 0:
        # Freeze the whole pretrained network; train ONLY LoRA adapters on the text pathway
        # (project_text_embed + transformer decoder) plus the small category embedding.
        for pm in model.net.parameters():
            pm.requires_grad_(False)
        n_lora = apply_lora(model.net, ["project_text_embed", ".linear1", ".linear2"],
                            r=a.lora_rank, alpha=a.lora_rank)
        model.to(dev)
        tot = sum(pm.numel() for pm in model.parameters())
        tr = sum(pm.numel() for pm in model.parameters() if pm.requires_grad)
        log(f"[lora] rank={a.lora_rank} adapted {n_lora} linears; trainable {tr/1e6:.2f}M / {tot/1e6:.0f}M")
    start_ep = 0
    if a.resume and os.path.exists(a.resume):          # continue a run that was cut short
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        # A checkpoint that was itself trained at a finer fusion grid carries a bigger pos_embed
        # than the freshly-built net, and a strict load fails on it. This is the same helper
        # eval_sweep / cache_val_probs / predict_test all need; it is a no-op for a checkpoint at
        # the pretrained stage, so it is safe to call unconditionally. Without it, RESUMING a
        # fusion-stage run (requeue, timeout) dies where the first launch succeeded.
        from fusion_res import align_to_checkpoint
        align_to_checkpoint(model.net, ck["network_weights"], patch=int(patch[0]))
        model.to(dev)
        model.net.load_state_dict(ck["network_weights"])
        model.cat_embed.load_state_dict(ck["cat_embed"])
        start_ep = int(ck.get("epoch", -1)) + 1
        log(f"[resume] {a.resume} -> starting at epoch {start_ep}")
    # AFTER the checkpoint load, never before: the checkpoint carries the pretrained 12^3 shapes
    # and a strict load against a retargeted model would fail.
    fresh_ids = set()
    if a.fusion_stage >= 0:
        from fusion_res import retarget_fusion
        _fresh = retarget_fusion(model.net, a.fusion_stage, patch=int(patch[0]))
        fresh_ids = {id(pm) for pm in _fresh}
        model.to(dev)
        _g = int(round(int(model.net.pos_embed.shape[0]) ** (1 / 3)))
        log(f"[fusion] stage {a.fusion_stage}: {_g}^3 = {model.net.pos_embed.shape[0]} tokens "
            f"({int(patch[0])//_g} voxels/token), {sum(pm.numel() for pm in _fresh)/1e6:.2f}M "
            f"fresh params at {a.fusion_lr_mult}x LR")
    fresh_ids = set(fresh_ids) | film_ids    # FiLM heads are zero-init: always in the hot group
    if ddp:
        # VoxTell returns a list of deep-supervision heads; not every parameter contributes to
        # every head, so unused-parameter detection must stay on.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    log(f"[model] in_ch={n_in} patch={patch} world_size={world} "
        f"index={a.index} same_cat={a.require_same_cat} optim={a.optim} lr={a.lr} excl={a.excl_weight} "
        f"effective_batch={a.batch*a.n_crops*world} patches/step")

    # each rank samples independently from the full case list (the dataset is random-access,
    # so no DistributedSampler is needed -- ranks simply draw different random crops)
    _cat_boost = {k: float(v) for k, v in (p.split(':') for p in a.cat_boost.split())} if a.cat_boost else None
    if _cat_boost:
        log(f"[sampler] cat_boost={_cat_boost}")
    ds = PromptPatchesFull(rows, text2vec, patch, n_items=a.iters * a.batch,
                           cat_dropout=a.cat_dropout, n_crops=a.n_crops, fg_prob=a.fg_prob, cat_balance=a.cat_balance, cat_boost=_cat_boost,
                           pair_prob=a.pair_prob, require_same_cat=a.require_same_cat,
                           use_anat=not a.no_anat,
                           lobe2vecs=lobe2vecs, lobe_prob=a.lobe_prob, fg_jitter=a.fg_jitter, anat_pos=a.anat_pos,
                           tn=a.tn_weight > 0,
                           prefer_same_cat=a.prefer_same_cat,
                           region_aux=a.region_aux > 0,
                           aug_map=aug_map, aug_prob=a.text_aug,
                           paste=(tuple(float(x) for x in a.copy_paste.split(","))
                                  if a.copy_paste else None),
                           aug_exclusive=a.aug_exclusive,
                           lat_map=lat_map, lat_prob=a.lat_swap)

    if a.aux_index:
        from aux_dataset import AuxPatches, load_aux_bank
        aux_t2v = load_aux_bank(a.aux_bank)
        aux_ds = AuxPatches(a.aux_index, aux_t2v, patch, n_items=ds.n_items,
                            n_crops=a.n_crops, use_anat=not a.no_anat, dilate=a.aux_dilate)
        ds = _MixDS(ds, aux_ds, a.aux_prob, ds.n_items)
        log(f"[aux] mixing {len(aux_ds.rows)} CT-RATE pseudo-label findings at p={a.aux_prob} "
            f"({len(aux_t2v)} embedded texts)")

    if main_proc:                      # self-check: are we actually building two-finding patches?
        both = same = tot = 0
        for _ in range(3):
            _t = ds[0]                       # aux-mix datasets return shorter tuples
            tg, ct_, vl = _t[1], _t[3], _t[5]
            for k in range(tg.shape[0]):
                nz = [j for j in range(tg.shape[1]) if vl[k][j] > 0 and float(tg[k][j].sum()) > 0]
                tot += 1
                if len(nz) >= 2:
                    both += 1
                    if int(ct_[k][nz[0]]) == int(ct_[k][nz[1]]):
                        same += 1
        log(f"[sampler] {tot} probe patches: {both} ({100*both/max(tot,1):.0f}%) show TWO findings, "
            f"{same} of those SAME-category (the case where text location is the only signal)")
    # persistent_workers=False: workers are torn down between epochs so they cannot accumulate
    # decompressed volumes across the run (the overnight OOM). Respawn cost is trivial vs epochs.
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True,
                    drop_last=True, persistent_workers=False, prefetch_factor=2 if a.workers else None)
    trainable = [pm for pm in model.parameters() if pm.requires_grad]
    if fresh_ids:
        base = [pm for pm in trainable if id(pm) not in fresh_ids]
        new = [pm for pm in trainable if id(pm) in fresh_ids]
        mult = (a.film_lr_mult if (getattr(a, "text_film", "") or getattr(a, "attn_gate", "")
                                   or a.region_aux > 0)
                else a.fusion_lr_mult)
        groups = [dict(params=base, lr_mult=1.0), dict(params=new, lr_mult=mult)]
    else:
        groups = [dict(params=trainable, lr_mult=1.0)]
    opt = (torch.optim.SGD(groups, lr=a.lr, momentum=0.99, nesterov=True)
           if a.optim == "sgd" else
           torch.optim.AdamW(groups, lr=a.lr, weight_decay=1e-5))
    _tv = None
    if a.tversky:
        _tv = tuple(float(x) for x in a.tversky.split(','))
        # fail loudly: `sbatch --export=ALL,TV=0.6,0.4` splits on the comma and delivers only
        # "0.6", which used to reach the loss as a 1-tuple and die with IndexError 7 minutes in
        if len(_tv) != 2:
            raise SystemExit(f"--tversky needs 'alpha,beta', got {a.tversky!r} -> {_tv}. "
                             f"If launching via sbatch --export, the comma is a separator: "
                             f"set the default inside the script instead.")
    if _tv:
        log(f"[loss] Tversky alpha={_tv[0]} (FN) beta={_tv[1]} (FP) "
            f"-- alpha>beta forgives predicting outside the annotation")
    scaler = torch.amp.GradScaler("cuda")

    for ep in range(start_ep, a.epochs):
        for g in opt.param_groups:
            # lr_mult keeps the retargeted-fusion group hot; without it this loop would silently
            # overwrite the raised LR back to the base value every epoch.
            g["lr"] = a.lr * g.get("lr_mult", 1.0) * (1 - ep / a.epochs) ** 0.9
        model.train()
        tot = n = 0
        for img, m, txt, cat, use_cat, valid, apos, tn, cap, regt, rval in dl:
            img, m = img.flatten(0, 1), m.flatten(0, 1)      # (B,n_crops,...) -> (P,...)
            txt, cat = txt.flatten(0, 1), cat.flatten(0, 1)
            use_cat, valid = use_cat.flatten(0, 1), valid.flatten(0, 1)
            tn = tn.flatten(0, 1); cap = cap.flatten(0, 1)
            regt = regt.flatten(0, 1); rval = rval.flatten(0, 1)
            apos = apos.flatten(0, 1) if a.anat_pos else None
            P = img.shape[0]
            # One 192^3 patch costs ~15 GB of activations, so the n_crops patches from a single
            # volume read cannot go through in one forward. Accumulate over micro-batches and
            # step once -- keeps the I/O amortisation without the OOM.
            opt.zero_grad(set_to_none=True)
            nmicro = max(1, (P + a.micro - 1) // a.micro)
            for k in range(0, P, a.micro):
                sl = slice(k, min(k + a.micro, P))
                # arrives fp16 / bool from the loader (see PromptPatchesFull._payload_f32);
                # the promotion is a GPU-side cast, off the critical CPU path.
                i_ = img[sl].to(dev, non_blocking=True).float()
                m_ = m[sl].to(dev, non_blocking=True).float()
                t_, c_ = txt[sl].to(dev, non_blocking=True), cat[sl].to(dev, non_blocking=True)
                u_, v_ = use_cat[sl].to(dev, non_blocking=True), valid[sl].to(dev, non_blocking=True)
                tn_ = tn[sl].to(dev, non_blocking=True) if a.tn_weight else None
                tv_ = _tv
                if a.cap_forgive > 0:
                    # NB: NOT c_ -- that is the category tensor fed to cat_embed
                    cf_ = cap[sl].to(dev, non_blocking=True)
                    b_ = torch.where(cf_ > 0, torch.full_like(cf_, a.cap_forgive),
                                     torch.full_like(cf_, 0.5))
                    tv_ = (1.0 - b_, b_)          # alpha = 1-beta; 0.5/0.5 == plain Dice
                # the anatomy-relative encoding must be sliced with the SAME micro-batch slice,
                # otherwise patches get another patch's coordinates -- silently, and it would
                # look like the frame simply does not help
                (model.module if ddp else model).set_pos(
                    apos[sl].to(dev, non_blocking=True) if apos is not None else None)
                sync = (k + a.micro) >= P            # only sync grads on the last micro-batch
                ctx = model.no_sync() if (ddp and not sync) else contextlib.nullcontext()
                with ctx:
                    with torch.amp.autocast("cuda"):
                        out = model(i_, t_, c_, u_)
                        loss = deep_sup_loss(out, m_, v_,
                                             excl_weight=a.excl_weight,
                                             tversky=tv_) / nmicro
                        if a.boundary_weight > 0:
                            finest = out if torch.is_tensor(out) else max(out, key=lambda o: o.shape[-1])
                            gtb = m_.float()
                            dil = torch.nn.functional.max_pool3d(gtb, 3, 1, 1)
                            ero = -torch.nn.functional.max_pool3d(-gtb, 3, 1, 1)
                            band = (dil - ero).clamp(0, 1)          # ~1-voxel shell around GT edges
                            lb = torch.nn.functional.binary_cross_entropy_with_logits(
                                finest.float(), gtb, reduction="none")
                            w_ = band * v_[:, :, None, None, None]
                            den = w_.sum().clamp(min=1.0)
                            loss = loss + (a.boundary_weight * (lb * w_).sum() / den) / nmicro
                        if a.region_aux > 0:
                            core_ = model.module if ddp else model
                            ro = core_.region_out                       # (mb, N, X, Y, Z)
                            if ro is not None:
                                rt_ = regt[sl].to(dev, non_blocking=True).float()
                                rw_ = (v_ * rval[sl].to(dev, non_blocking=True))
                                w_ = rw_[:, :, None, None, None]
                                lr_ = nn.functional.binary_cross_entropy_with_logits(
                                    ro.float(), rt_, reduction="none")
                                # mean BCE per voxel over VALID prompts: (sum of weighted BCE)
                                # / (n_valid_prompts * voxels_per_prompt)
                                den = (w_.sum() * ro.shape[2] * ro.shape[3]
                                       * ro.shape[4]).clamp(min=1.0)
                                loss = loss + (a.region_aux * (lr_ * w_).sum() / den) / nmicro
                        if a.detect_aux > 0:
                            core_ = model.module if ddp else model
                            do_ = core_.detect_out
                            if do_ is not None:
                                un_ = (m_.float() * v_[:, :, None, None, None]).amax(1)
                                wd_ = un_ + a.detect_neg * (1.0 - un_)
                                ld_ = nn.functional.binary_cross_entropy_with_logits(
                                    do_.float(), un_, reduction="none")
                                loss = loss + (a.detect_aux * (ld_ * wd_).sum()
                                               / wd_.sum().clamp(min=1.0)) / nmicro
                        if a.tn_weight:
                            loss = loss + (a.tn_weight * tn_penalty(out, tn_, v_)) / nmicro
                        if a.attn_excl:
                            mm = model.module if ddp else model
                            att = mm._attn
                            if isinstance(att, (list, tuple)):
                                att = att[-1]            # last decoder layer
                            loss = loss + (a.attn_excl *
                                           attention_exclusion(att, m_, v_)) / nmicro
                    scaler.scale(loss).backward()
                tot += loss.item() * nmicro; n += 1
            scaler.step(opt); scaler.update()
        if ddp:                                    # average the reported loss across ranks
            t = torch.tensor([tot, n], dtype=torch.float64, device=dev)
            dist.all_reduce(t); tot, n = t[0].item(), t[1].item()
        core = model.module if ddp else model
        log(f"[ep {ep+1}/{a.epochs}] loss={tot/max(n,1):.4f} lr={opt.param_groups[0]['lr']:.2e} "
            f"|cat_emb|={float(core.cat_embed.weight.norm()):.3f}")
        if main_proc:
            # `film_*`/`gate_gain` live on T1Model, NOT on net, so saving only net.state_dict()
            # silently drops them and the checkpoint reloads as an UNCONDITIONED model whose
            # encoder was trained expecting conditioning. That mismatch is what invalidated the
            # first --text-film run (EXPERIMENTS.md 4o, retracted). Persist the extra pathway AND
            # the stage lists needed to rebuild it.
            if (ep % a.save_every == 0) or (ep >= a.epochs - 1):
                _extra = {k: v for k, v in core.state_dict().items()
                          if k.startswith(("film_trunk", "film_heads", "gate_gain", "region_proj", "detect_head"))}
                # ATOMIC save: write to .tmp then rename. A failed in-place save (disk full --
                # the 2026-08-21 quota incident) destroys the PREVIOUS good checkpoint too;
                # os.replace either succeeds completely or leaves the old file untouched.
                # --save-every N skips the 1.8G NFS write on non-multiple epochs (GPU-util fix).
                _tmp = a.out + ".tmp"
                torch.save({"network_weights": core.net.state_dict(),
                            "cat_embed": core.cat_embed.state_dict(), "epoch": ep,
                            "cond_extra": _extra,
                            "film_stages": list(core.film_stages),
                            "gate_stages": list(core.gate_stages),
                            "region_aux": bool(core.region_aux)}, _tmp)
                os.replace(_tmp, a.out)
    if ddp:
        dist.destroy_process_group()
    log(f"[done] -> {a.out}")


if __name__ == "__main__":
    main()

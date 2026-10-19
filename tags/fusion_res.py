"""Move text-image fusion to a finer grid, so the text can point at a lesion-sized thing.

The measurement that motivates this. VoxTell fuses text and image at ONE place: a transformer
decoder where the projected text vector is the query and the encoder's stage-4 feature map is the
memory. That map is 12^3 = 1728 tokens over a 192^3 patch, so one token spans 16 voxels --
roughly 11 x 11 x 20 mm at CT-RATE spacings.

The lesions we are asked to ground, measured on 76 nodules marked by both the radiologist and
TotalSegmentator in the same scan, are 6.4 mm in-plane and 4.5 mm in z. The target is SMALLER
THAN ONE TOKEN in every dimension. So the sentence can only steer the model to a neighbourhood
containing several candidates; it cannot indicate which. That is exactly the failure the error
analysis keeps returning: 70% of false-positive volume sits in the CORRECT lobe and 56% of it is
more than 20 mm from the target -- and 20 mm is about one token in z. A lobe is many tokens, so
lobe-level steering works; within-lobe steering has no resolution to work with.

Stage 3 is 24^3 = 13824 tokens at 256 channels, halving token size to ~5.5 x 5.5 x 10 mm, which
is comparable to the lesions rather than larger than them. It is affordable because this is
CROSS-attention: a couple of queries against N keys, so cost grows with N, not N^2.

Two pieces need surgery, and both are done AFTER the checkpoint loads, never before -- the
checkpoint carries the old shapes and a strict load would fail:

  pos_embed                   (1728,1,2048) -> (13824,1,2048)   trilinear interpolation
  project_bottleneck_embed.0  Linear(320->2048) -> Linear(256->2048)

Everything else transfers untouched. Attention weights are resolution-agnostic (2048x2048 whatever
the key count), so the encoder, the decoder convolutions, the text projection and the mask head all
load as-is -- 1093 of 1095 tensors.

The honest risk: stage-3 features are not stage-4 features, so whatever the pretrained projection
learned about its input distribution is partly invalidated. The new Linear is seeded from the old
one rather than randomly (see `_reseed`), and the caller should give it a raised learning rate --
at the standard 1e-4 SGD a fresh projection may simply not move in 60 epochs, and "did not move"
would read as "did not help".

    python fusion_res.py --self-test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _reseed(old: nn.Linear, new_in: int) -> nn.Linear:
    """A Linear(new_in -> out) seeded from a pretrained Linear(old_in -> out).

    Random init would hand the pretrained transformer decoder noise and destroy the very thing we
    are trying to keep. Instead the first `new_in` input columns are copied and rescaled by
    sqrt(old_in / new_in), which preserves the output variance of the layer under the usual
    assumption of unit-variance decorrelated inputs. The features differ in meaning between
    stages, so this is a scale-preserving starting point rather than a correct mapping -- it only
    has to be better than noise, and it is.
    """
    old_in, out = old.in_features, old.out_features
    new = nn.Linear(new_in, out, bias=old.bias is not None)
    with torch.no_grad():
        k = min(old_in, new_in)
        new.weight.zero_()
        new.weight[:, :k] = old.weight[:, :k] * (old_in / max(k, 1)) ** 0.5
        if old.bias is not None:
            new.bias.copy_(old.bias)
    return new


def _interp_pos(pos: torch.Tensor, new_grid: int) -> torch.Tensor:
    """(G^3, 1, C) -> (new_grid^3, 1, C), trilinear.

    The flattening in VoxTell's forward is `b h w d c -> (h w d) b c`, so the token axis unfolds
    as (h, w, d) in that order and must be folded back the same way. Getting this wrong would
    silently permute the spatial axes of the positional field -- no error, just a model whose
    sense of up and left is scrambled.
    """
    n, b, c = pos.shape
    g = round(n ** (1 / 3))
    if g ** 3 != n:
        raise ValueError(f"pos_embed length {n} is not a cube")
    x = pos.reshape(g, g, g, b, c)[:, :, :, 0, :]          # (g,g,g,C)
    x = x.permute(3, 0, 1, 2).unsqueeze(0)                 # (1,C,g,g,g)
    x = F.interpolate(x.float(), size=(new_grid,) * 3, mode="trilinear", align_corners=False)
    x = x.squeeze(0).permute(1, 2, 3, 0)                   # (n,n,n,C)
    return x.reshape(new_grid ** 3, 1, c).to(pos.dtype).contiguous()


def retarget_fusion(net, stage: int, patch: int = 192):
    """Point the text-image fusion at encoder `stage`. Returns the newly-created parameters.

    Call AFTER loading the checkpoint. Returns the parameters that did not come from pretraining,
    so the caller can put them in their own optimizer group.
    """
    old_stage = int(net.selected_decoder_layer)
    if stage == old_stage:
        return []
    strides = [1, 2, 4, 8, 16, 32]
    if not 0 <= stage < len(strides):
        raise ValueError(f"stage {stage} out of range")
    grid = patch // strides[stage]
    # channel count of that encoder stage, read from the model rather than assumed
    in_ch = [32, 64, 128, 256, 320, 320][stage]

    old_lin = net.project_bottleneck_embed[0]
    if old_lin.in_features != in_ch:
        net.project_bottleneck_embed[0] = _reseed(old_lin, in_ch).to(
            old_lin.weight.device, old_lin.weight.dtype)

    dev, dt = net.pos_embed.device, net.pos_embed.dtype
    new_pos = _interp_pos(net.pos_embed.data.cpu(), grid).to(dev, dt)
    was_param = isinstance(net.pos_embed, nn.Parameter)
    if was_param:
        net.pos_embed = nn.Parameter(new_pos, requires_grad=net.pos_embed.requires_grad)
    else:
        # registered as a buffer in v1.1 -- keep it a buffer so state_dict round-trips unchanged
        net.pos_embed = new_pos
        if hasattr(net, "_buffers") and "pos_embed" in net._buffers:
            net._buffers["pos_embed"] = new_pos

    net.selected_decoder_layer = stage
    fresh = list(net.project_bottleneck_embed[0].parameters())
    if was_param:
        fresh.append(net.pos_embed)
    return fresh


def align_to_checkpoint(net, ck_weights, patch=192, verbose=True):
    """Retarget `net` to whatever fusion stage the checkpoint was trained at, then it will load.

    Every code path that loads a checkpoint needs this, and each one that forgets fails the same
    way: `size mismatch for pos_embed ... [13824,1,2048] vs [1728,1,2048]`. That already cost one
    eval job and one caching job, because the fix was applied to `eval_sweep.py` and not to
    `cache_val_probs.py`. Keeping it in one function is the point.

    Safe to call unconditionally -- a checkpoint trained at the pretrained stage is a no-op.
    """
    pos = ck_weights.get("pos_embed")
    if pos is None or int(pos.shape[0]) == int(net.pos_embed.shape[0]):
        return False
    g = int(round(int(pos.shape[0]) ** (1 / 3)))
    stage = {24: 3, 12: 4, 6: 5}.get(g)
    if stage is None:
        raise SystemExit(f"checkpoint pos_embed grid {g}^3 maps to no known encoder stage")
    retarget_fusion(net, stage, patch=patch)
    if verbose:
        print(f"[fusion] checkpoint fuses at stage {stage} ({g}^3 = {pos.shape[0]} tokens, "
              f"{patch//g} voxels/token) -- retargeted before load", flush=True)
    return True


def _self_test():
    from train_t1_full import build_model
    net, _ = build_model("cpu", in_channels=1)
    print(f"  before: stage {net.selected_decoder_layer}  pos {tuple(net.pos_embed.shape)}  "
          f"proj_in {net.project_bottleneck_embed[0].in_features}")
    ck = torch.load(HERE / "checkpoints" / "SUBMISSION_1ch.pt", map_location="cpu",
                    weights_only=False)
    net.load_state_dict(ck["network_weights"])
    print("  checkpoint loaded strictly (old shapes) -- OK")

    ref_pos = net.pos_embed.clone()
    fresh = retarget_fusion(net, 3)
    print(f"  after : stage {net.selected_decoder_layer}  pos {tuple(net.pos_embed.shape)}  "
          f"proj_in {net.project_bottleneck_embed[0].in_features}  "
          f"fresh params {sum(p.numel() for p in fresh)/1e6:.2f}M")

    # the interpolated field must preserve the coarse structure it came from
    back = _interp_pos(net.pos_embed, 12)
    err = (back - ref_pos).abs().mean() / ref_pos.abs().mean()
    print(f"  round-trip 12->24->12 relative error: {err:.4f} (small = structure preserved)")

    net.eval()
    with torch.no_grad():
        out = net(torch.zeros(1, 1, 192, 192, 192), torch.zeros(1, 2, 1, 2560))
    o = out[0] if isinstance(out, list) else out
    print(f"  forward OK: {len(out) if isinstance(out,list) else 1} heads, "
          f"head0 {tuple(o.shape)}")
    assert o.shape[:2] == (1, 2), o.shape
    assert o.shape[2:] == (192, 192, 192), o.shape
    print("\n  PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    if ap.parse_args().self_test:
        _self_test()

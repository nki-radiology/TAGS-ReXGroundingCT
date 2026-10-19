"""VoxTellTrainer_T1_base -- faithful VoxTell promptable fine-tune for Track 1 (free-text).

The model consumes the EXACT free-text finding as a Qwen3-Embedding-4B vector and predicts a
binary mask for it. Warm-started from VoxTell v1.1 (never from an earlier local checkpoint --
a model that learned to ignore text has that pathology in its weights).

Recipe (Rokuss 2025, published SOTA on this benchmark = 0.282). The earlier local attempt
scored 0.055 because it deviated on all four of these:
  * deep supervision at 5 scales, lambda = [1, 1/2, 1/4, 1/8, 1/16]   (was: disabled)
  * 2 positive + 1 NEGATIVE prompt per image                          (was: no negatives)
  * SGD lr 1e-4 poly, momentum 0.99 nesterov                          (was: AdamW)
  * 85% foreground oversampling                                       (was: 66%)
plus NO left-right mirroring: mirroring a volume while the prompt says "right upper lobe"
actively trains the model to treat laterality as noise.

The negative prompt is the mechanism that makes text load-bearing: without it, "segment any
abnormality" is never punished and the model is free to ignore the prompt entirely.

GEOMETRY: images come from CT-RATE `_fixed` (proper HU, LPS affine) while masks come from
ReXGroundingCT (identity affine). Their voxel arrays are identical, so both are read as RAW
arrays with NO canonical reorientation -- reorienting only the image would silently misalign
it against the mask.

    python train_t1_base.py --epochs 250 --batch 2
"""
import argparse
import json
import pydoc
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voxtell.model.voxtell_model import VoxTellModel  # noqa: E402

HERE = Path(__file__).resolve().parent
from paths import VOXTELL_DIR as MODEL_DIR, TEXT_BANK_DIR as BANK_DIR  # noqa: E402
DS_WEIGHTS = [1.0, 0.5, 0.25, 0.125, 0.0625]        # deep supervision, coarse->fine ordering fixed at use


def build_model(device, deep_supervision=True, text_dim=2560):
    plans = json.load(open(MODEL_DIR / "plans.json"))
    cfg = plans["configurations"]["3d_fullres"]
    ak = dict(cfg["architecture"]["arch_kwargs"])
    for k in cfg["architecture"]["_kw_requires_import"]:
        if ak[k] is not None:
            ak[k] = pydoc.locate(ak[k])
    net = VoxTellModel(input_channels=1, **ak, decoder_layer=4, text_embedding_dim=text_dim,
                       num_maskformer_stages=5, num_heads=32, query_dim=2048,
                       project_to_decoder_hidden_dim=2048, deep_supervision=deep_supervision)
    ck = torch.load(MODEL_DIR / "fold_0" / "checkpoint_final.pth", map_location="cpu",
                    weights_only=False)
    net.load_state_dict(dict(ck["network_weights"]))
    return net.to(device), [int(x) for x in cfg["patch_size"]]


def zscore(x):
    """Z-score over foreground (voxels above the volume minimum).

    Kept as the straightforward boolean-index version. A "no-copy" rewrite using
    sum/sum-of-squares reductions was tried and REVERTED: it was no faster (numpy's boolean
    reduction is already efficient here) and it lost precision -- float32 accumulation in
    `dot(x,x)` over ~50M voxels made var = E[x^2] - E[x]^2 catastrophically cancel, giving
    max|diff| ~0.15 against this reference.
    """
    fg = x > x.min()
    m = x[fg].mean() if fg.any() else x.mean()
    s = x[fg].std() if fg.any() else 1.0
    return (x - m) / (s + 1e-6)


class PromptPatches(Dataset):
    """One item = (ONE CT patch, N target masks, N text embeddings).

    VoxTell takes N prompts per image in a single forward (text `(B,N,1,D)` -> masks `(B,N,...)`),
    sharing one image encode, so all prompts must describe the SAME patch.

    ONLY REAL ANNOTATED FINDINGS ARE USED AS PROMPTS -- no fabricated cross-case negatives.
    The "this text describes nothing here" signal that makes the text load-bearing already
    arises, and is VERIFIED, from two places:
      * a co-occurring finding located outside the crop -> its target in this patch is
        genuinely empty (we know where it actually is);
      * the 1-fg_prob fraction of random crops -> the prompt's own finding is often outside.

    Fabricated negatives (a finding text borrowed from another scan, target forced to zero)
    are UNSOUND here: ReXGroundingCT annotates only what the report mentioned, so absence of
    an annotation is not absence of the pathology. A borrowed "nodule in the left lower lobe"
    may well be present but unannotated, which would train a real nodule as background -- and
    the risk grows the more semantically plausible (harder) the negative is.
    """

    def __init__(self, rows, text2vec, patch, n_items, fg_prob=0.85, n_pos=2):
        self.by_case = {}
        for r in rows:
            self.by_case.setdefault(r["case"], []).append(r)
        self.cases = sorted(self.by_case)
        self.rows = rows
        self.text2vec, self.patch = text2vec, patch
        self.fg_prob, self.n_pos = fg_prob, n_pos
        self.n_items = n_items

    def __len__(self):
        return self.n_items

    @staticmethod
    def _load_vol(path):
        return zscore(np.asanyarray(nib.load(path).dataobj).astype(np.float32))   # RAW, no reorient

    @staticmethod
    def _load_mask(path, fidx):
        """Read ONLY this finding's channel (lazy) -- cases can carry many channels."""
        obj = nib.load(path).dataobj
        arr = obj[fidx] if len(obj.shape) == 4 else np.asanyarray(obj)
        return (np.asanyarray(arr) > 0).astype(np.float32)

    def _pad_amt(self, shape):
        ph, pw, pd = self.patch
        return [(0, max(0, ph - shape[0])), (0, max(0, pw - shape[1])), (0, max(0, pd - shape[2]))]

    def _origin(self, shape, m, force_fg):
        ph, pw, pd = self.patch
        H, W, D = shape
        if force_fg and m is not None and m.any():
            c = np.argwhere(m > 0)
            c = c[np.random.randint(len(c))]
            return (int(np.clip(c[0] - ph // 2, 0, H - ph)), int(np.clip(c[1] - pw // 2, 0, W - pw)),
                    int(np.clip(c[2] - pd // 2, 0, D - pd)))
        return (np.random.randint(0, H - ph + 1), np.random.randint(0, W - pw + 1),
                np.random.randint(0, D - pd + 1))

    def __getitem__(self, _):
        rows = self.by_case[self.cases[np.random.randint(len(self.cases))]]
        vol = self._load_vol(rows[0]["image"])
        pad = self._pad_amt(vol.shape)
        vol = np.pad(vol, pad)
        ph, pw, pd = self.patch

        anchor = rows[np.random.randint(len(rows))]            # the finding the crop centres on
        m_anchor = np.pad(self._load_mask(anchor["seg"], anchor["fidx"]), pad)
        y, x, z = self._origin(vol.shape, m_anchor, np.random.rand() < self.fg_prob)
        sl = (slice(y, y + ph), slice(x, x + pw), slice(z, z + pd))
        ct = vol[sl]

        # Co-occurring findings of THIS case, without replacement. A case with one finding
        # simply contributes one prompt -- we never duplicate it or invent a second.
        others = [r for r in rows if r is not anchor]
        np.random.shuffle(others)
        picks = [anchor] + others[: self.n_pos - 1]
        targets = [m_anchor[sl]] + [np.pad(self._load_mask(r["seg"], r["fidx"]), pad)[sl]
                                    for r in picks[1:]]
        texts = [self.text2vec[r["text"]] for r in picks]

        # Pad to a fixed n_pos so batches collate; padded slots are masked out of the loss.
        valid = np.zeros(self.n_pos, np.float32)
        valid[: len(picks)] = 1.0
        while len(targets) < self.n_pos:
            targets.append(np.zeros_like(ct))
            texts.append(texts[0])

        return (torch.from_numpy(ct)[None].float(),                        # (1, ph,pw,pd)
                torch.from_numpy(np.stack(targets)).float(),               # (n_pos, ph,pw,pd)
                torch.from_numpy(np.stack(texts)).float()[:, None],        # (n_pos, 1, D)
                torch.from_numpy(valid))                                   # (n_pos,)


def dice_bce(logit, target, valid=None, eps=1e-5):
    """logit/target: (B, N, ...). Dice is PER PROMPT, never pooled across prompts -- pooling
    would let a confident mask on one prompt hide a wrong answer on another. `valid` (B,N)
    zeroes out padding slots for cases that had fewer findings than n_pos.

    Dice is applied ONLY where the target is non-empty. For an empty ground truth the soft-Dice
    term degenerates to `1 - eps/(p.sum()+eps)`: exactly 0 if the model predicts nothing and 1
    if it predicts anything, i.e. pure "output zeros everywhere" pressure. Roughly 40-50% of our
    prompts have empty targets in their patch (a co-occurring finding outside the crop, or a
    random crop), which collapsed the first run to all-empty predictions -- loss rose from 0.95
    to 1.17 over 45 epochs as the non-empty targets then scored Dice ~1.
    BCE still runs on every prompt, so false positives on empty prompts are penalised; only the
    degenerate Dice term is dropped.
    """
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    bce = bce.flatten(0, 1).flatten(1).mean(1)                 # (B*N,)
    p = torch.sigmoid(logit).flatten(0, 1).flatten(1)
    t = target.flatten(0, 1).flatten(1)
    tsum = t.sum(1)
    num = 2 * (p * t).sum(1) + eps
    den = p.sum(1) + tsum + eps
    per = bce + (1 - num / den) * (tsum > 0).float()           # Dice only on non-empty targets
    if valid is None:
        return per.mean()
    w = valid.flatten()
    return (per * w).sum() / w.sum().clamp(min=1.0)


def tversky_bce(logit, target, valid=None, alpha=0.6, beta=0.4, eps=1e-5):
    """Dice+BCE with the overlap term made ASYMMETRIC: false negatives cost `alpha`, false
    positives `beta`. alpha>beta forgives predicting outside the annotation.

    WHY. The dataset is not fully annotated -- measured on val: 20% of "bilateral" findings are
    labelled in one lung only, predictions span 2.1x the annotated slice range with 57% bracketing
    it, and diffuse findings get point-marker masks (emphysema "in both lungs" = 0.4 mL). So a
    voxel outside the annotation is often a CORRECT segmentation of an unannotated part of the
    same lesion. Punishing it as a hard negative gives contradictory gradients on identical visual
    patterns and corrupts the representation, not just the score.

    THE PRIOR IS NEGATIVE AND THIS IS DELIBERATELY MILD. Recall-weighted Tversky has been tried
    twice at beta=0.9 (9:1): our own 501 encoder ("blobs on 40-85% of the scan") and the dHT
    project ("still paints, pf 0.979", marked actively harmful and removed). Both ran away into
    over-segmentation -- which is already our worst failure. 0.6/0.4 is a 1.5:1 nudge, not 9:1.

    Dice is still skipped on EMPTY targets: with tp=fn=0 the Tversky index has no gradient at all
    (the bug both projects hit), and 40-50% of our prompts have empty targets in their patch.
    BCE keeps running everywhere, so empty prompts are still pushed toward zero.
    """
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    bce = bce.flatten(0, 1).flatten(1).mean(1)
    p = torch.sigmoid(logit).flatten(0, 1).flatten(1)
    t = target.flatten(0, 1).flatten(1)
    tp = (p * t).sum(1)
    fn = ((1 - p) * t).sum(1)
    fp = (p * (1 - t)).sum(1)
    # alpha/beta may be TENSORS of shape (B*N,) -- one setting per prompt. Used by --cap-forgive:
    # a training finding with entity_count==3 hit the annotators' 3-instance cap, so voxels outside
    # its mask are often REAL lesions that were never drawn. Punishing those teaches the model to
    # stop finding things (measured: it scores 0.269 on seen train data vs 0.319 on unseen val).
    # Those prompts get beta<alpha; every other prompt keeps 0.5/0.5, which IS plain Dice.
    if torch.is_tensor(alpha):
        alpha = alpha.to(tp.device).reshape(-1)
    if torch.is_tensor(beta):
        beta = beta.to(tp.device).reshape(-1)
    tv = 1 - (tp + eps) / (tp + alpha * fn + beta * fp + eps)
    per = bce + tv * (t.sum(1) > 0).float()
    if valid is None:
        return per.mean()
    w = valid.flatten()
    return (per * w).sum() / w.sum().clamp(min=1.0)


def exclusion_loss(logit, target, valid=None):
    """Punish predicting ANOTHER prompt's finding. THE mechanism that makes text matter.

    With plain per-prompt Dice, a model that outputs A-union-B for BOTH prompts still scores
    ~0.67 on each -- so ignoring the text is a comfortable local optimum, and nothing anywhere
    in the loss says the two outputs must DIFFER. That is exactly the observed failure: swap
    the lobe in the prompt and the mask does not move (Dice(A,B)=1.000), while the model
    over-segments wildly (14781 voxels vs the pretrained model's 0-57).

    For prompt i this penalises probability mass inside the OTHER prompts' targets but outside
    its own: region_i = (union_{j != i} target_j) AND NOT target_i. Segmenting the other
    finding is then strictly costly, and the only way to avoid it is to read the prompt.
    """
    p = torch.sigmoid(logit)                              # (B,N,...)
    B, N = p.shape[0], p.shape[1]
    if N < 2:
        return p.sum() * 0.0
    tot = target.sum(dim=1, keepdim=True)                 # how many prompts cover each voxel
    others = (tot - target).clamp(0, 1)                   # union of the OTHER prompts' targets
    region = others * (1.0 - target)                      # theirs, and not mine
    num = (p * region).flatten(2).sum(-1)                 # leaked probability mass  (B,N)
    den = region.flatten(2).sum(-1).clamp(min=1.0)
    per = (num / den).flatten()
    if valid is None:
        return per.mean()
    w = valid.flatten()
    return (per * w).sum() / w.sum().clamp(min=1.0)


def deep_sup_loss(out, target, valid=None, excl_weight=0.0, tversky=None):
    """out: tensor or list of logits (matched to target by spatial size, finest -> coarsest)."""
    if torch.is_tensor(out):
        base = (dice_bce(out, target, valid) if tversky is None
                else tversky_bce(out, target, valid, *tversky))
        return base + excl_weight * exclusion_loss(out, target, valid) if excl_weight else base
    outs = sorted(out, key=lambda t: -t.shape[-1])            # finest -> coarsest
    total, wsum = 0.0, 0.0
    for w, o in zip(DS_WEIGHTS, outs):
        t = target if o.shape[-3:] == target.shape[-3:] else F.interpolate(
            target, size=o.shape[-3:], mode="nearest")
        term = (dice_bce(o, t, valid) if tversky is None
                else tversky_bce(o, t, valid, *tversky))
        if excl_weight:
            term = term + excl_weight * exclusion_loss(o, t, valid)
        total = total + w * term
        wsum += w
    return total / wsum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--iters", type=int, default=250, help="cases sampled per epoch")
    ap.add_argument("--batch", type=int, default=2, help="cases per step (x3 prompts each)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(HERE / "VoxTellTrainer_T1_base.pt"))
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [json.loads(l) for l in open(HERE / "index" / "t1_index_train.jsonl")]
    bank_texts = json.load(open(BANK_DIR / "texts.json"))
    bank_emb = np.load(BANK_DIR / "embeddings.npy")
    text2vec = {t: bank_emb[i] for i, t in enumerate(bank_texts)}
    missing = {r["text"] for r in rows} - set(text2vec)
    assert not missing, f"{len(missing)} finding texts absent from the embedding bank"

    net, patch = build_model(dev, deep_supervision=True)
    print(f"[model] VoxTell v1.1 loaded, deep_supervision=True, patch={patch}", flush=True)
    print(f"[data] {len(rows)} findings over {len({r['case'] for r in rows})} cases", flush=True)

    ds = PromptPatches(rows, text2vec, patch, n_items=a.iters * a.batch)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True, drop_last=True)
    opt = torch.optim.SGD(net.parameters(), lr=a.lr, momentum=0.99, nesterov=True)
    scaler = torch.amp.GradScaler("cuda")

    for ep in range(a.epochs):
        for g in opt.param_groups:                       # poly schedule
            g["lr"] = a.lr * (1 - ep / a.epochs) ** 0.9
        net.train()
        tot = n = 0
        for ct, m, emb, valid in dl:
            ct = ct.to(dev, non_blocking=True)         # (B,1,ph,pw,pd)
            m = m.to(dev, non_blocking=True)           # (B,N,ph,pw,pd)
            emb = emb.to(dev, non_blocking=True)       # (B,N,1,D) -- N prompts share one encode
            valid = valid.to(dev, non_blocking=True)   # (B,N) -- padded slots excluded
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = deep_sup_loss(net(ct, emb), m, valid)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item(); n += 1
        print(f"[ep {ep+1}/{a.epochs}] loss={tot/max(n,1):.4f} lr={opt.param_groups[0]['lr']:.2e}",
              flush=True)
        torch.save({"network_weights": net.state_dict(), "epoch": ep}, a.out)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()

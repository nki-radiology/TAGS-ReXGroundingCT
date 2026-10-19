"""Lesion copy-paste augmentation for text-conditioned segmentation.

Two modes, both text-consistency-safe:

ADD (counter the annotation cap): pick a prompt whose text is PLURAL-SAFE (plural nouns or
  count-agnostic wording) and paste 1-2 extra same-category lesions INSIDE the region its text
  names (whole lung when no location is named). The pasted voxels are ADDED to that prompt's
  target. Train findings are capped at 3 annotated instances while val/test average 3.8 -- this
  makes training targets look like test targets, attacking the cap from the DATA side (the loss
  side, cap-forgive, failed: EXPERIMENTS.md 4k).

DISTRACTOR (attack precision 0.373): paste a lesion that no prompt's mask includes, so the model
  must NOT segment it for any prompt in the crop. Safety rules, checked against EVERY valid
  prompt: (i) same-category distractors only OUTSIDE a prompt's named region -- and only when the
  prompt NAMES a region; a location-free same-category prompt semantically includes any pasted
  instance, so such crops get cross-category distractors only. (ii) never overlap any target.
  The same-category-elsewhere case is the `category-AMBIGUOUS` lesson (two lesions, text location
  is the only disambiguator) -- available naturally in only ~35% of cases, synthesised here.

Placement: centre sampled inside the TS lung mask (intersected with the allowed lobe set for ADD),
rejecting positions whose pasted mask would overlap existing targets or leave the lung by more
than half. Hard paste of masked voxels with a 1-voxel feather; source and destination are both
z-scored volumes, so intensities transfer directly.
"""
import json, re
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent
LOBES = [10, 11, 12, 13, 14]
LEFT, RIGHT = [10, 11], [12, 13, 14]
PLURAL = re.compile(r"\b(nodules|opacities|densities|lesions|areas|changes|infiltrat|"
                    r"multiple|several|few|scattered|numerous|both|bilateral)\b", re.I)

_L = {"upper": {"left": 10, "right": 12}, "middle": {"right": 13},
      "lower": {"left": 11, "right": 14}, "lingula": 11}

def named_region(text):
    """Lobe ids the text names, or None. Mirrors eval_t1_val200.named_region (copied to avoid a
    circular import: eval imports train_t1_full, which imports this module)."""
    t = text.lower()
    out = set()
    left = "left" in t; right = "right" in t
    if "lingula" in t or "lingular" in t:
        out.add(11)
    for part, sides in (("upper", ("left", "right")), ("middle", ("right",)),
                        ("lower", ("left", "right"))):
        if part + " lobe" in t or part + " lobes" in t or f"{part} zone" in t:
            for s in sides:
                if (s == "left" and left) or (s == "right" and right) or (not left and not right):
                    out.add(_L[part][s] if isinstance(_L[part], dict) else _L[part])
    if not out:
        if left and not right:
            out.update(LEFT)
        elif right and not left:
            out.update(RIGHT)
    return sorted(out) or None


class PasteBank:
    def __init__(self, root=None):
        root = Path(root or H / "paste_bank")
        self.root = root
        self.rows = [json.loads(l) for l in open(root / "index.jsonl")]
        self.by_cat = {}
        for i, r in enumerate(self.rows):
            self.by_cat.setdefault(r["category"], []).append(i)

    def sample(self, category, rng, max_vox=8000):
        idxs = self.by_cat.get(category)
        if not idxs:
            return None
        for _ in range(6):
            r = self.rows[idxs[rng.integers(len(idxs))]]
            if r["vox"] <= max_vox:
                z = np.load(self.root / "lesions" / f"{r['id']}.npz")
                return dict(ct=z["ct"].astype(np.float32), mask=z["mask"].astype(bool),
                            category=r["category"], case=r["case"])
        return None


def _feather(mask):
    """1-voxel soft edge so the paste boundary is not a step function."""
    from scipy import ndimage
    er = ndimage.binary_erosion(mask)
    w = mask.astype(np.float32)
    w[mask & ~er] = 0.6
    return w


def _try_place(les, allowed, targets_any, crop_shape, rng, tries=8):
    d, h, w = les["mask"].shape
    D, Hh, W = crop_shape
    if d >= D or h >= Hh or w >= W:
        return None
    cand = np.argwhere(allowed)
    if len(cand) == 0:
        return None
    for _ in range(tries):
        cy, cx, cz = cand[rng.integers(len(cand))]
        y0, x0, z0 = cy - d // 2, cx - h // 2, cz - w // 2
        if y0 < 0 or x0 < 0 or z0 < 0 or y0 + d > D or x0 + h > Hh or z0 + w > W:
            continue
        sl = (slice(y0, y0 + d), slice(x0, x0 + h), slice(z0, z0 + w))
        if (targets_any[sl] & les["mask"]).any():
            continue
        inside = allowed[sl][les["mask"]].mean()
        if inside < 0.5:
            continue
        return sl
    return None


def apply_copy_paste(img, targets, picks, valid_n, ts_crop, bank, rng, p_add, p_dis):
    """Mutates img (C,D,H,W) and targets (list of (D,H,W) bool) in place.

    picks: the dataset rows for the crop's prompts (length == len(targets) may exceed picks for
    padding -- only the first valid_n are real). ts_crop: TotalSegmentator labels for this crop,
    or None (then no pasting -- placement needs the lung mask)."""
    if ts_crop is None or bank is None:
        return 0, 0
    lung = np.isin(ts_crop, LOBES)
    if lung.sum() < 5000:
        return 0, 0
    targets_any = np.zeros_like(lung)
    for t in targets[:valid_n]:
        targets_any |= t.astype(bool)
    n_add = n_dis = 0

    # ---- ADD ----
    if rng.random() < p_add:
        order = rng.permutation(valid_n)
        for j in order:
            r = picks[j]
            if not PLURAL.search(r["text"]):
                continue
            reg = named_region(r["text"])
            allowed = lung & np.isin(ts_crop, reg) if reg else lung
            les = bank.sample(r["category"], rng)
            if les is None:
                continue
            sl = _try_place(les, allowed, targets_any, lung.shape, rng)
            if sl is None:
                continue
            wgt = _feather(les["mask"])
            img[0][sl] = img[0][sl] * (1 - wgt) + les["ct"] * wgt
            targets[j][sl] |= les["mask"]
            targets_any[sl] |= les["mask"]
            n_add = 1
            break

    # ---- DISTRACTOR ----
    if rng.random() < p_dis:
        cats_free = {picks[j]["category"] for j in range(valid_n)
                     if named_region(picks[j]["text"]) is None}
        excl = np.zeros_like(lung)
        cand_cats = list(bank.by_cat.keys())
        rng.shuffle(cand_cats)
        for cat in cand_cats:
            if cat in cats_free:
                continue                        # location-free prompt of this cat would include it
            allowed = lung.copy()
            ok = True
            for j in range(valid_n):
                if picks[j]["category"] == cat:
                    reg = named_region(picks[j]["text"])
                    if reg is None:
                        ok = False; break
                    allowed &= ~np.isin(ts_crop, reg)   # keep distractor OUT of its named region
            if not ok or allowed.sum() < 5000:
                continue
            les = bank.sample(cat, rng)
            if les is None:
                continue
            sl = _try_place(les, allowed, targets_any, lung.shape, rng)
            if sl is None:
                continue
            wgt = _feather(les["mask"])
            img[0][sl] = img[0][sl] * (1 - wgt) + les["ct"] * wgt
            targets_any[sl] |= les["mask"]      # block later pastes; targets themselves untouched
            n_dis = 1
            break
    return n_add, n_dis

"""Auxiliary CT-RATE + TotalSegmentator patches, mixed into Track-1 training.

ReXGroundingCT has 1,743 nodule (2d) and 237 effusion (2e) findings. CT-RATE reports mapped by
the LLM mapper, paired with the TotalSegmentator `lung_nodules` / `pleural_pericard_effusion`
tasks, give 19,537 + 4,612 more of the SAME image domain with REAL free-text findings.

Three things this handles that a naive loader would get wrong:

* **Label, not `>0`.** The TS masks are multi-label; `>0` selects the whole lung (~8.7M voxels)
  rather than the lesion (~380). The row carries `mask_label` (2 for both tasks: `lung_nodules`
  and `pleural_effusion` — NOT 3, which is pericardial effusion, a different structure).

* **Empty masks are skipped.** ~15% of findings say "nodule" where TotalSegmentator found none.
  Training text->empty would teach the model to suppress the very thing it should segment, so
  those are resampled rather than emitted.

* **Mask is per-SCAN, finding is per-LESION.** The TS mask covers every nodule in the volume
  while the text names one ("...in the right lower lobe"). Where the finding's `lobes` are
  known, the mask is intersected with them via ts_total, making the target finding-specific.

These are PSEUDO-labels. Mix them with real annotations (`--aux-prob`), never train on them
alone: the model would inherit TotalSegmentator's ceiling and its false negatives.
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from train_t1_full import CAT2IDX, LOBES, zscore


class AuxPatches(Dataset):
    """One item = (CT patch, mask patch, text vec, category) from the CT-RATE aux pool."""

    def __init__(self, index_path, text2vec, patch, n_items, n_pos=2, n_crops=8,
                 fg_prob=0.85, min_vox=20, use_anat=False, dilate=0):
        self.rows = [json.loads(l) for l in open(index_path)]
        self.rows = [r for r in self.rows if r["text"] in text2vec]
        self.text2vec = text2vec
        self.patch, self.n_items, self.n_pos = patch, n_items, n_pos
        self.n_crops, self.fg_prob, self.min_vox = n_crops, fg_prob, min_vox
        self.use_anat = use_anat
        self.dilate = int(dilate)
        if not self.rows:
            raise RuntimeError(f"no aux rows with embedded text in {index_path}")

    def __len__(self):
        return self.n_items

    def _inplane_dilate(self, m, affine):
        """Widen the mask IN-PLANE to match how the annotations are drawn.

        Measured over 250 masks from each source (largest component, so multi-nodule findings do
        not confound the shape):

                        eq_diam   in-plane      z     aniso
            pseudo-label   4.7        4.8       4.8    0.98
            annotation     6.6        7.9       5.0    0.62

        The annotations are not shorter in z -- 5.0 against 4.8. They are much wider IN-PLANE,
        7.9 against 4.8. A radiologist draws a generous outline on each slice; TotalSegmentator
        draws a tight one. So the two differ by a drawing convention, and a model trained on the
        tight version is scored against the generous one. This is also why trimming predictions
        in z post-hoc returned ~0.000 per category: there was never excess z to remove.

        `calibrate_convention.py` swept the radius: 2 matches SIZE (6.5 vs 6.6), 3 matches SHAPE
        (aniso 0.63 vs 0.62). Dilation is in-plane only -- the through-plane axis is the coarsest
        spacing, read from the affine rather than assumed, since CT-RATE is not consistently
        oriented and a hardcoded axis would widen the wrong way.
        """
        sp = np.abs(nib.affines.voxel_sizes(affine))
        zax = int(np.argmax(sp))
        st = np.zeros((3, 3, 3), bool)
        sl = [slice(None)] * 3
        sl[zax] = 1
        st[tuple(sl)] = ndimage.generate_binary_structure(2, 1)
        # dilate inside a padded crop: the lesion is a few hundred voxels in a 512x512x300 volume
        # and dilating the whole array would cost more than the rest of the loader combined.
        idx = np.argwhere(m)
        k = self.dilate
        lo = np.maximum(idx.min(0) - k - 1, 0)
        hi = np.minimum(idx.max(0) + k + 2, m.shape)
        sub = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        out = m.copy()
        out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = ndimage.binary_dilation(sub, st, iterations=k)
        return out

    def _load(self, r):
        """(ct, mask) at full res, or None if this row is unusable (empty lesion mask)."""
        mimg = nib.load(r["mask"])
        m_all = np.asanyarray(mimg.dataobj)
        if "fidx" in r:
            # model pseudo-labels: 4D per-case file, one BINARY channel per finding
            m = m_all[int(r["fidx"])] > 0
        else:
            m = (m_all == int(r.get("mask_label", 2)))
        if m.sum() < self.min_vox:
            return None                                   # segmenter found nothing -> skip
        if r.get("lobes") and r.get("ts_total"):
            try:                                          # make the mask finding-specific
                ts = np.asanyarray(nib.load(r["ts_total"]).dataobj)
                mi = m & np.isin(ts, r["lobes"])
                if mi.sum() >= self.min_vox:
                    m = mi
            except Exception:
                pass
        if self.dilate:
            m = self._inplane_dilate(m, mimg.affine)
        vol = zscore(np.asanyarray(nib.load(r["image"]).dataobj).astype(np.float32))
        if vol.shape != m.shape:
            return None
        return vol.astype(np.float16), m

    def __getitem__(self, _):
        ph, pw, pd = self.patch
        for _try in range(12):                            # resample past empty/broken rows
            r = self.rows[np.random.randint(len(self.rows))]
            try:
                got = self._load(r)
            except Exception:
                got = None
            if got is None:
                continue
            vol, m = got
            pad = [(0, max(0, ph - vol.shape[0])), (0, max(0, pw - vol.shape[1])),
                   (0, max(0, pd - vol.shape[2]))]
            if any(p[1] for p in pad):
                vol, m = np.pad(vol, pad), np.pad(m, pad)
            H, W, D = vol.shape
            imgs, tgts, txts, cats, uses, vals = [], [], [], [], [], []
            for _c in range(self.n_crops):
                if np.random.rand() < self.fg_prob and m.any():
                    c = np.argwhere(m)
                    c = c[np.random.randint(len(c))]
                    y = int(np.clip(c[0] - ph // 2, 0, H - ph))
                    x = int(np.clip(c[1] - pw // 2, 0, W - pw))
                    z = int(np.clip(c[2] - pd // 2, 0, D - pd))
                else:
                    y = np.random.randint(0, H - ph + 1); x = np.random.randint(0, W - pw + 1)
                    z = np.random.randint(0, D - pd + 1)
                sl = (slice(y, y + ph), slice(x, x + pw), slice(z, z + pd))
                imgs.append(vol[sl][None].astype(np.float32))
                t = np.zeros((self.n_pos,) + (ph, pw, pd), np.float32)
                t[0] = m[sl]
                tgts.append(t)
                v = self.text2vec[r["text"]]
                txts.append(np.stack([v] * self.n_pos)[:, None])
                ci = CAT2IDX[r["category"]]
                cats.append(np.array([ci] * self.n_pos))
                uses.append(np.array([1.0] + [0.0] * (self.n_pos - 1), np.float32))
                val = np.zeros(self.n_pos, np.float32); val[0] = 1.0   # only slot 0 is real
                vals.append(val)
            return (torch.from_numpy(np.stack(imgs)).float(),
                    torch.from_numpy(np.stack(tgts)).float(),
                    torch.from_numpy(np.stack(txts)).float(),
                    torch.tensor(np.stack(cats), dtype=torch.long),
                    torch.tensor(np.stack(uses), dtype=torch.float32),
                    torch.from_numpy(np.stack(vals)),
                    # 7th slot: anatomy-relative token positions. The main dataset grew this when
                    # --anat-pos was added and the training loop now unpacks seven values, so an
                    # aux batch returning six crashes the epoch before it starts. Aux rows carry
                    # no anatomy frame, so this is the same zero placeholder the main dataset
                    # emits when --anat-pos is off, and it is never read in that case.
                    torch.zeros(len(imgs), 1, 1),
                    # 8th slot: the true-negative restriction mask, added with --tn-weight AFTER
                    # aux_2e and aux_dil2 were trained -- which is why their original runs
                    # succeeded and a rerun on the same command line does not. THE SAME BUG AS
                    # THE 7th SLOT, one release later. `tn` derives from a TotalSegmentator
                    # parse of the finding's sentence; CT-RATE pseudo-label rows have no such
                    # sentence, so aux is exempt and emits the (n_pos,1,1,1) placeholder the main
                    # dataset uses when tn is off. Anything appended to the main dataset's tuple
                    # must be mirrored here or every aux run dies in its first epoch.
                    torch.zeros(len(imgs), self.n_pos, 1, 1, 1),
                    # 9th-11th slots: cap-forgive flag, region-aux target, region-aux validity --
                    # added to the main dataset 2026-08-14..20. Aux rows are exempt from all
                    # three; shapes mirror the main dataset's off placeholders. (Reader: this is
                    # the THIRD instance of the mirror-the-tuple bug the two comments above
                    # describe. If you append to PromptPatchesFull's return, append HERE too.)
                    torch.zeros(len(imgs), self.n_pos),
                    torch.zeros(len(imgs), self.n_pos, 1, 1, 1),
                    torch.zeros(len(imgs), self.n_pos))
        raise RuntimeError("no usable aux sample in 12 tries")


def load_aux_bank(bank_dir):
    """text -> vector for the aux findings (embedded by prep_aux_texts.py)."""
    bank_dir = Path(bank_dir)
    texts = json.load(open(bank_dir / "texts.json"))
    emb = np.load(bank_dir / "embeddings.npy")
    return {t: emb[i] for i, t in enumerate(texts)}

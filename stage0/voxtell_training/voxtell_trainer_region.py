# Modified from VoxTell (https://github.com/MIC-DKFZ/VoxTell, Apache-2.0) by the fomo.fo team, 2026:
# trainer variants for the 14-category ReXGroundingCT region model. Cluster paths replaced by env vars.
"""Region-based (multi-label sigmoid) VoxTell fine-tuning + focal oversampling.

Why: the 14-way softmax model suppresses small/minority classes — they dump their
voxels into background or a dominant neighbour for lower loss (nodules 2d scored 0.000
on val-200 despite being 35% of findings). Region-based training gives one independent
sigmoid channel per category (DC+BCE per channel), so each class competes only with its
own background. FocalDataLoader additionally forces a fixed fraction of foreground
patches to be centred on nodule/micronodule voxels, which sigmoid heads alone don't
guarantee (default oversampling picks uniformly among present classes, dominated by the
big parenchymal classes).

Region mode itself is triggered by the dataset.json (an `all_fg` region with len>1).
Loss + #output-heads are then selected automatically by nnUNetTrainer.
"""
from __future__ import annotations

import numpy as np
import torch

from voxtell.training.voxtell_trainer import VoxTellTrainer
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader

# class_locations keys are region tuples (see LabelManager.foreground_regions).
# labels: 1e micronodules = 5, 2d nodules/masses = 10  -> region tuples (5,), (10,)
FOCAL_REGIONS = [(5,), (10,)]
FOCAL_P = 0.5  # fraction of foreground patches forced onto a focal class (when present)


class FocalDataLoader(nnUNetDataLoader):
    """Bias foreground oversampling toward small focal classes via overwrite_class."""

    def get_bbox(self, data_shape, force_fg, class_locations, overwrite_class=None, verbose=False):
        if force_fg and overwrite_class is None and class_locations is not None \
                and np.random.uniform() < FOCAL_P:
            present = [k for k in FOCAL_REGIONS
                       if k in class_locations and len(class_locations[k]) > 0]
            if present:
                overwrite_class = present[np.random.randint(len(present))]
        return super().get_bbox(data_shape, force_fg, class_locations, overwrite_class, verbose)


class VoxTellTrainer_region_250ep(VoxTellTrainer):
    """250-epoch region-based fine-tune with focal (nodule/micronodule) oversampling."""

    DDP_BATCH = 8  # global batch; VoxTellTrainer hardcodes 2 (too small for >2-GPU DDP)

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.oversample_foreground_percent = 0.5  # default 0.33; more fg patches overall
        # VoxTellTrainer.__init__ forces cfg["batch_size"]=2. Under DDP raise it so the
        # global batch can split across GPUs; on a single GPU keep 2 (else OOM).
        if self.is_ddp:
            self.configuration_manager.configuration["batch_size"] = self.DDP_BATCH

    def get_dataloaders(self):
        # nnUNetTrainer.get_dataloaders instantiates the module-global nnUNetDataLoader;
        # swap in FocalDataLoader for the duration so both tr/val use focal sampling.
        import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as M
        orig = M.nnUNetDataLoader
        M.nnUNetDataLoader = FocalDataLoader
        try:
            return super().get_dataloaders()
        finally:
            M.nnUNetDataLoader = orig


# ----------------------- class-balanced focal sampling ----------------------- #
import json as _json  # noqa: E402
import os as _os      # noqa: E402

_WEIGHTS_PATH = __import__("os").path.join(__import__("os").environ.get("TAGS_STAGE0_DATA", "."), "class_weights_501.json")
_CODES = ["1a", "1b", "1c", "1d", "1e", "1f", "2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h"]
# CLASS_WEIGHTS[label 1..14] = inverse-frequency weight (median-freq balanced, measured).
if _os.path.isfile(_WEIGHTS_PATH):
    _w = _json.load(open(_WEIGHTS_PATH))["weights"]          # {code: weight}
    CLASS_WEIGHTS = {i + 1: float(_w[c]) for i, c in enumerate(_CODES)}
else:                                                        # fallback: uniform
    CLASS_WEIGHTS = {k: 1.0 for k in range(1, 15)}


class BalancedFocalDataLoader(nnUNetDataLoader):
    """Every forced-foreground patch is centred on a class sampled from a distribution
    weighted inversely to class frequency (CLASS_WEIGHTS) among the classes actually
    present in that case. Rare/small classes (nodule, micronodule, effusion) get heavy
    weight; common classes keep a floor so no channel starves. Falls back to nnU-Net's
    default (random present class) if the case has no single-label foreground region."""

    def get_bbox(self, data_shape, force_fg, class_locations, overwrite_class=None, verbose=False):
        if force_fg and overwrite_class is None and class_locations is not None:
            try:
                present = [k for k in class_locations
                           if isinstance(k, tuple) and len(k) == 1 and 1 <= k[0] <= 14
                           and len(class_locations[k]) > 0]
                if present:
                    w = np.array([CLASS_WEIGHTS.get(k[0], 1.0) for k in present], dtype=np.float64)
                    # robust weighted pick: searchsorted on the cumulative weights. Avoids
                    # np.random.choice(p=...), which RAISES (and silently hangs the worker)
                    # if the probabilities don't sum to exactly 1 due to float error.
                    c = np.cumsum(w)
                    overwrite_class = present[int(np.searchsorted(c, np.random.random() * c[-1]))]
            except Exception:
                overwrite_class = None   # never crash a dataloader worker -> fall back to default
        return super().get_bbox(data_shape, force_fg, class_locations, overwrite_class, verbose)


class VoxTellTrainer_region_bal_250ep(VoxTellTrainer_region_250ep):
    """Region/sigmoid fine-tune with class-BALANCED focal oversampling + per-epoch logging
    of per-channel positive-voxel counts across sampled patches (starvation observable)."""

    def get_dataloaders(self):
        import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as M
        orig = M.nnUNetDataLoader
        M.nnUNetDataLoader = BalancedFocalDataLoader
        try:
            # skip the parent's FocalDataLoader swap -> go straight to nnUNetTrainer's impl
            return VoxTellTrainer.get_dataloaders(self)
        finally:
            M.nnUNetDataLoader = orig

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._cls_vox = {k: 0 for k in range(1, 15)}

    def train_step(self, batch):
        t = batch["target"]
        t = t[0] if isinstance(t, list) else t              # (b, C, ...)
        with torch.no_grad():
            if t.shape[1] >= 15:                             # one-hot regions: channel k = label k
                for k in range(1, 15):
                    self._cls_vox[k] += int(t[:, k].sum().item())
            else:                                            # integer label map
                for k in range(1, 15):
                    self._cls_vox[k] += int((t == k).sum().item())
        return super().train_step(batch)

    def on_train_epoch_end(self, train_outputs):
        self.print_to_log_file(
            "patch_class_voxels " + " ".join(f"{_CODES[k - 1]}={self._cls_vox[k]}"
                                             for k in range(1, 15)))
        super().on_train_epoch_end(train_outputs)

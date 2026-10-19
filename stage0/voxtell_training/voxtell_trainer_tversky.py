# Modified from VoxTell (https://github.com/MIC-DKFZ/VoxTell, Apache-2.0) by the fomo.fo team, 2026:
# trainer variants for the 14-category ReXGroundingCT region model. Cluster paths replaced by env vars.
"""Tversky-loss region fine-tune for VoxTell.

Fixes the rare-class (effusion=0.000) per-channel sigmoid *loss collapse* by replacing
the region Dice with a Tversky loss that penalises false negatives more than false
positives (alpha<beta). Keeps the 0.229 "uniform sampling" recipe (VoxTellTrainer_250epochs)
untouched otherwise — sampling is NOT changed, because balanced sampling was already shown
not to fix the collapse (model card 05). Only the loss aggregation changes.
"""
import json
import os

import numpy as np
import torch

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, AllGatherGrad
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader

from voxtell.training.voxtell_trainer import VoxTellTrainer, VoxTellTrainer_250epochs

# ---- presence-based weights (fix for rare-PRESENCE collapse, e.g. effusion) ----
_CODES = ["1a", "1b", "1c", "1d", "1e", "1f", "2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h"]
try:
    _pw = json.load(open(os.path.join(os.environ["TAGS_STAGE0_DATA"], "class_weights_presence.json")))["weights"]
    PRESENCE_WEIGHTS = {i + 1: float(_pw[c]) for i, c in enumerate(_CODES)}  # label 1..14 -> weight
except Exception:
    PRESENCE_WEIGHTS = {k: 1.0 for k in range(1, 15)}
try:
    CASE_WEIGHTS = json.load(open(os.path.join(os.environ["TAGS_STAGE0_DATA"], "case_sampling_weights.json")))
except Exception:
    CASE_WEIGHTS = {}


class MemoryEfficientTverskyLoss(MemoryEfficientSoftDiceLoss):
    """Tversky index  TP / (TP + alpha*FP + beta*FN).  Dice == alpha=beta=0.5.
    alpha<beta penalises false negatives (the 'rare channel predicts empty' failure).
    Same forward signature (x, y, loss_mask) as the Dice it replaces, so it drops into
    DC_and_BCE_loss as `dice_class`. Body copied from MemoryEfficientSoftDiceLoss.forward;
    only the final aggregation differs."""

    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 ddp=True, alpha=0.3, beta=0.7):
        super().__init__(apply_nonlin=apply_nonlin, batch_dice=batch_dice, do_bg=do_bg,
                         smooth=smooth, ddp=ddp)
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        axes = tuple(range(2, x.ndim))
        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))
            if x.shape == y.shape:
                y_onehot = y.to(torch.float32)
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                y_onehot.scatter_(1, y.long(), 1)
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            sum_gt = y_onehot.sum(axes, dtype=torch.float32) if loss_mask is None \
                else (y_onehot * loss_mask).sum(axes, dtype=torch.float32)
        if not self.do_bg:
            x = x[:, 1:]
        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes, dtype=torch.float32)
            sum_pred = x.sum(axes, dtype=torch.float32)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes, dtype=torch.float32)
            sum_pred = (x * loss_mask).sum(axes, dtype=torch.float32)
        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0, dtype=torch.float32)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0, dtype=torch.float32)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0, dtype=torch.float32)
            intersect = intersect.sum(0, dtype=torch.float32)
            sum_pred = sum_pred.sum(0, dtype=torch.float32)
            sum_gt = sum_gt.sum(0, dtype=torch.float32)
        tp = intersect
        fp = sum_pred - intersect
        fn = sum_gt - intersect
        tv = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn
                                   + float(self.smooth)).clamp_min(1e-8)
        return -tv.mean()


class VoxTellTrainer_region_tversky(VoxTellTrainer_250epochs):
    """0.229 recipe (uniform sampling, 250 ep) with the region loss's Dice term replaced by
    Tversky to keep rare channels alive. alpha/beta via env TVERSKY_ALPHA/BETA (default
    0.3/0.7); beta>alpha penalises false negatives (the 'rare channel predicts empty' failure)."""

    _alpha = None   # subclasses set these for distinct output dirs; else env; else default
    _beta = None

    def _build_loss(self):
        self.alpha = self._alpha if self._alpha is not None else float(os.environ.get("TVERSKY_ALPHA", "0.3"))
        self.beta = self._beta if self._beta is not None else float(os.environ.get("TVERSKY_BETA", "0.7"))
        self.print_to_log_file(f"[Tversky] alpha={self.alpha} beta={self.beta}")
        assert self.label_manager.has_regions, "Tversky variant expects region (sigmoid) mode"
        loss = DC_and_BCE_loss(
            {},
            {'batch_dice': self.configuration_manager.batch_dice, 'do_bg': True,
             'smooth': 1e-5, 'ddp': self.is_ddp, 'alpha': self.alpha, 'beta': self.beta},
            use_ignore_label=self.label_manager.ignore_label is not None,
            dice_class=MemoryEfficientTverskyLoss,
        )
        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(scales))])
            weights[-1] = 1e-6 if (self.is_ddp and not self._do_i_compile()) else 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class PresenceFocalDataLoader(nnUNetDataLoader):
    """Two-level oversampling by scan-PRESENCE (fixes the rare-presence collapse the
    voxel-count weighting caused for effusion):
      (1) CASE level: sampling_probabilities upweight cases containing rare-presence classes;
      (2) WITHIN case: forced-fg patches centred on a class picked by PRESENCE_WEIGHTS."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            ids = list(self.indices)
            med = float(np.median(list(CASE_WEIGHTS.values()))) if CASE_WEIGHTS else 1.0
            w = np.array([float(CASE_WEIGHTS.get(i, med)) for i in ids], dtype=np.float64)
            if w.sum() > 0:
                self.sampling_probabilities = w / w.sum()   # case-level oversampling
        except Exception:
            pass

    def get_bbox(self, data_shape, force_fg, class_locations, overwrite_class=None, verbose=False):
        if force_fg and overwrite_class is None and class_locations is not None:
            try:
                present = [k for k in class_locations
                           if isinstance(k, tuple) and len(k) == 1 and 1 <= k[0] <= 14
                           and len(class_locations[k]) > 0]
                if present:
                    ww = np.array([PRESENCE_WEIGHTS.get(k[0], 1.0) for k in present], dtype=np.float64)
                    c = np.cumsum(ww)
                    overwrite_class = present[int(np.searchsorted(c, np.random.random() * c[-1]))]
            except Exception:
                overwrite_class = None
        return super().get_bbox(data_shape, force_fg, class_locations, overwrite_class, verbose)


class VoxTellTrainer_region_presence_tversky(VoxTellTrainer_region_tversky):
    """Tversky loss (env alpha/beta) + presence-based case & within-case oversampling."""

    def get_dataloaders(self):
        import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as M
        orig = M.nnUNetDataLoader
        M.nnUNetDataLoader = PresenceFocalDataLoader
        try:
            return VoxTellTrainer.get_dataloaders(self)
        finally:
            M.nnUNetDataLoader = orig


# ---- distinct classes per setting (each -> its own nnU-Net results dir) ----
class VoxTellTrainer_region_tversky_b90(VoxTellTrainer_region_tversky):          # uniform, Tversky .1/.9
    _alpha, _beta = 0.1, 0.9


class VoxTellTrainer_region_tversky_b95(VoxTellTrainer_region_tversky):          # uniform, Tversky .05/.95
    _alpha, _beta = 0.05, 0.95


class VoxTellTrainer_region_presence_tversky_b90(VoxTellTrainer_region_presence_tversky):  # presence + .1/.9
    _alpha, _beta = 0.1, 0.9

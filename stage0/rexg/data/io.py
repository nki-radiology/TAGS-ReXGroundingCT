"""Load ReXGroundingCT volumes (in HU) and their finding masks.

CT-RATE volumes are stored as raw values; true HU = value * RescaleSlope +
RescaleIntercept, with per-scan slope/intercept in the CT-RATE metadata CSVs
(e.g. train_10000_a_1 uses intercept -8192). TAP-CT clips in HU, so this
conversion is mandatory.

Array conventions (matching the dataset's own layout):
    volume: (H, W, D)
    mask:   (F, H, W, D)  -- F findings; within a finding, each integer value
                             is a distinct entity/instance.
"""

from __future__ import annotations

import functools
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from .paths import REXG_ROOT, parse_name


@functools.lru_cache(maxsize=4)
def _metadata(split: str, root: Path = REXG_ROOT) -> pd.DataFrame:
    fname = "meta_train_metadata.csv" if split == "train" else "meta_validation_metadata.csv"
    df = pd.read_csv(root / fname).set_index("VolumeName")
    return df


def rescale_to_hu(name: str, root: Path = REXG_ROOT) -> tuple[float, float]:
    """Return (slope, intercept) for converting stored values to HU."""
    split = parse_name(name)["split"]
    row = _metadata(split, root).loc[name if name.endswith(".nii.gz") else f"{name}.nii.gz"]
    return float(row["RescaleSlope"]), float(row["RescaleIntercept"])


def load_volume_hu(name: str, root: Path = REXG_ROOT) -> np.ndarray:
    """Load a CT volume as float32 HU, shape (H, W, D)."""
    path = root / "volumes" / (name if name.endswith(".nii.gz") else f"{name}.nii.gz")
    vol = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    slope, intercept = rescale_to_hu(name, root)
    return vol * slope + intercept


def load_mask(name: str, root: Path = REXG_ROOT) -> np.ndarray:
    """Load a finding mask, shape (F, H, W, D), integer entity labels."""
    path = root / "segmentations" / (name if name.endswith(".nii.gz") else f"{name}.nii.gz")
    arr = np.asarray(nib.load(str(path)).dataobj)
    if arr.ndim == 3:
        arr = arr[None]
    return arr.astype(np.int16)


def category_mask(name: str, channels: list[int], root: Path = REXG_ROOT) -> np.ndarray:
    """Binary union mask (H, W, D) over the given finding channels of one scan."""
    m = load_mask(name, root)                    # (F, H, W, D)
    sel = m[channels]                            # (len(channels), H, W, D)
    return (sel > 0).any(axis=0)


def spacing(name: str, root: Path = REXG_ROOT) -> tuple[float, float, float]:
    """Voxel spacing in mm as (x, y, z)."""
    split = parse_name(name)["split"]
    row = _metadata(split, root).loc[name if name.endswith(".nii.gz") else f"{name}.nii.gz"]
    xy = eval(row["XYSpacing"]) if isinstance(row["XYSpacing"], str) else row["XYSpacing"]
    return float(xy[0]), float(xy[1]), float(row["ZSpacing"])

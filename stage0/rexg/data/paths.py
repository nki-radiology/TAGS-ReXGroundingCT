"""Resolve ReXGroundingCT mask filenames to the source CT-RATE volume on disk.

A challenge entry's ``name`` (e.g. ``train_1741_b_2.nii.gz``) is a CT-RATE
filename. CT-RATE lays the volumes out as::

    <CT_RATE_ROOT>/dataset/<split>/train_1741/train_1741_b/train_1741_b_2.nii.gz

where ``<split>`` is ``train`` or ``valid``. The challenge mask of the same
name lives flat in ``<REXG_ROOT>/segmentations/<name>``.
"""

from __future__ import annotations

import re
from pathlib import Path

import os
CT_RATE_ROOT = Path(os.environ.get("TAGS_CTRATE_ROOT", "data/CT-RATE/dataset"))


def _resolve_rexg_root() -> Path:
    """Original ReXGroundingCT release (dataset.json, volumes/, segmentations/)."""
    return Path(os.environ.get("TAGS_REXG_ROOT", "data/ReXGroundingCT"))


REXG_ROOT = _resolve_rexg_root()

# train_1741_b_2  ->  groups: patient="train_1741", scan="train_1741_b"
_NAME_RE = re.compile(r"^(?P<split>train|valid)_(?P<pid>\d+)_(?P<scan>[a-z])_(?P<recon>\d+)$")


def parse_name(name: str) -> dict:
    """Split a CT-RATE/ReXGrounding filename into its components."""
    stem = name[:-7] if name.endswith(".nii.gz") else name
    m = _NAME_RE.match(stem)
    if not m:
        raise ValueError(f"Unrecognised CT-RATE name: {name!r}")
    d = m.groupdict()
    d["stem"] = stem
    return d


def ct_rate_volume_path(name: str, ct_rate_root: Path = CT_RATE_ROOT) -> Path:
    """Map a challenge ``name`` to its CT-RATE volume path."""
    p = parse_name(name)
    patient = f"{p['split']}_{p['pid']}"
    scan = f"{patient}_{p['scan']}"
    fname = f"{p['stem']}.nii.gz"
    # CT-RATE challenge uses the 'train' folder for the released train+val splits;
    # held-out test volumes are distributed separately. Try the obvious split first,
    # then fall back to the sibling split.
    for split in (p["split"], "valid" if p["split"] == "train" else "train"):
        cand = ct_rate_root / "dataset" / split / patient / scan / fname
        if cand.exists():
            return cand
    # Return the primary guess even if missing, so the caller can report it.
    return ct_rate_root / "dataset" / p["split"] / patient / scan / fname


def mask_path(name: str, rexg_root: Path = REXG_ROOT) -> Path:
    fname = name if name.endswith(".nii.gz") else f"{name}.nii.gz"
    return rexg_root / "segmentations" / fname

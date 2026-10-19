"""Build Dataset501_ReXGround14Region: region-based (multi-label) version of 500.

- imagesTr: SYMLINKED from Dataset500 (identical HU volumes, no rebuild).
- labelsTr: rebuilt painting findings LARGEST-FIRST so small findings (nodules,
  micronodules) are written last and survive overlaps in their own category channel.
- dataset.json: region-based. 14 per-category regions (each an independent sigmoid
  channel) + one `all_fg` region (len>1) that flips nnU-Net into region mode.

Then: nnUNetv2_plan_and_preprocess -d 501 -pl nnUNetPlannerResEncL -c 3d_fullres ;
      voxtell-finetune 501 3d_fullres 0 -p nnUNetResEncUNetLPlans -tr VoxTellTrainer_region_250ep ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rexg.data.io import load_mask  # noqa: E402
from rexg.data.paths import REXG_ROOT  # noqa: E402

CODES14 = ["1a", "1b", "1c", "1d", "1e", "1f", "2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h"]
CODE2IDX = {c: i + 1 for i, c in enumerate(CODES14)}     # 1..14 (0 = background)
NNUNET_RAW = Path(__import__("os").environ["nnUNet_raw"])
SRC = "Dataset500_ReXGround14"
DST = "Dataset501_ReXGround14Region"
BYNAME: dict = {}


def build_case(name: str, src_img: Path, img_dir: Path, lbl_dir: Path) -> str:
    stem = name.replace(".nii.gz", "")
    ip, lp = img_dir / f"{stem}_0000.nii.gz", lbl_dir / f"{stem}.nii.gz"
    # symlink the HU image from Dataset500
    if not ip.exists():
        s = src_img / f"{stem}_0000.nii.gz"
        if s.exists():
            ip.symlink_to(s)
    if lp.exists():
        return "skip"
    masks = load_mask(name)                                  # (F,H,W,D)
    e = BYNAME[name]
    cats = e.get("categories", {})
    # paint largest finding first, smallest last -> small findings survive overlaps
    order = sorted(range(masks.shape[0]), key=lambda fi: int((masks[fi] > 0).sum()), reverse=True)
    lab = np.zeros(masks.shape[1:], np.uint8)
    for fi in order:
        code = cats.get(str(fi))
        if code in CODE2IDX:
            lab[masks[fi] > 0] = CODE2IDX[code]
    aff = nib.load(str(ip)).affine
    nib.save(nib.Nifti1Image(lab, aff), str(lp))
    return "ok"


def write_dataset_json(droot: Path, n: int):
    labels = {"background": 0, "all_fg": list(range(1, 15))}   # all_fg has len>1 -> region mode
    for c in CODES14:
        labels[c] = [CODE2IDX[c]]                             # one independent sigmoid channel each
    regions_class_order = [0] + [CODE2IDX[c] for c in CODES14]  # all_fg->0 (first, overwritten at export)
    ds = {"channel_names": {"0": "CT"}, "labels": labels,
          "regions_class_order": regions_class_order,
          "numTraining": n, "file_ending": ".nii.gz"}
    json.dump(ds, open(droot / "dataset.json", "w"), indent=2)
    print(f"wrote region dataset.json: {len(regions_class_order)} regions, numTraining={n}", flush=True)


def main():
    global BYNAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    dj = json.load(open(REXG_ROOT / "dataset.json"))
    entries = [e for e in dj["train"] if (REXG_ROOT / "volumes" / e["name"]).exists()]
    BYNAME = {e["name"]: e for e in entries}

    src_img = NNUNET_RAW / SRC / "imagesTr"
    droot = NNUNET_RAW / DST
    img_dir, lbl_dir = droot / "imagesTr", droot / "labelsTr"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    mine = entries[args.shard::args.nshards]
    print(f"shard {args.shard}/{args.nshards}: {len(mine)} cases", flush=True)
    done = err = 0
    for e in mine:
        try:
            build_case(e["name"], src_img, img_dir, lbl_dir)
        except Exception as ex:  # noqa: BLE001
            err += 1
            if err <= 5:
                print(f"  ERROR {e['name']}: {ex!r}", flush=True)
        done += 1
        if done % 100 == 0:
            print(f"  shard{args.shard}: {done}/{len(mine)} (err {err})", flush=True)

    if args.shard == 0:
        write_dataset_json(droot, len(entries))
    else:
        print(f"shard {args.shard} done ({err} errors)", flush=True)


if __name__ == "__main__":
    main()

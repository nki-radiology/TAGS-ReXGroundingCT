"""Build an nnU-Net raw dataset for category-based fine-tuning (path A).

Each training scan -> imagesTr/<id>_0000.nii.gz (HU volume, our rescale) +
labelsTr/<id>.nii.gz (14-class label map: each voxel = category index of the
finding it belongs to, 1..14; 0 = background). Plus dataset.json.

Then: nnUNetv2_plan_and_preprocess -d <id> ... ; voxtell-finetune <id> 3d_fullres 0
  -pretrained_weights .../voxtell_v1.1/fold_0/checkpoint_final.pth

    python scripts/24_build_nnunet_dataset.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rexg.data.io import load_mask, rescale_to_hu  # noqa: E402
from rexg.data.paths import REXG_ROOT  # noqa: E402

CODES14 = ["1a", "1b", "1c", "1d", "1e", "1f", "2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h"]
CODE2IDX = {c: i + 1 for i, c in enumerate(CODES14)}     # 1..14 (0 = background)
NNUNET_RAW = Path(__import__("os").environ["nnUNet_raw"])
DATASET = "Dataset500_ReXGround14"
BYNAME: dict = {}                                          # name -> dataset.json entry (set in main)


def build_case(name: str, img_dir: Path, lbl_dir: Path) -> str:
    stem = name.replace(".nii.gz", "")
    ip, lp = img_dir / f"{stem}_0000.nii.gz", lbl_dir / f"{stem}.nii.gz"
    if ip.exists() and lp.exists():
        return f"skip {stem}"
    ct = nib.load(str(REXG_ROOT / "volumes" / name))
    slope, intercept = rescale_to_hu(name)
    hu = np.asanyarray(ct.dataobj).astype(np.float32) * slope + intercept
    nib.save(nib.Nifti1Image(np.round(hu).astype(np.int16), ct.affine), str(ip))  # imagesTr (HU int16)
    # label map: paint each finding's mask voxels with its category index
    masks = load_mask(name)                                 # (F,H,W,D)
    e = BYNAME[name]
    cats = e.get("categories", {})
    lab = np.zeros(hu.shape, np.uint8)
    for fi in range(masks.shape[0]):
        code = cats.get(str(fi))
        if code in CODE2IDX:
            lab[masks[fi] > 0] = CODE2IDX[code]            # later findings overwrite (rare overlaps)
    nib.save(nib.Nifti1Image(lab, ct.affine), str(lp))
    return f"ok {stem}"


def main():
    global BYNAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1, help="run N separate processes (CPU/gzip bound)")
    args = ap.parse_args()

    dj = json.load(open(REXG_ROOT / "dataset.json"))
    entries = [e for e in dj["train"] if (REXG_ROOT / "volumes" / e["name"]).exists()]
    BYNAME = {e["name"]: e for e in entries}

    droot = NNUNET_RAW / DATASET
    img_dir, lbl_dir = droot / "imagesTr", droot / "labelsTr"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    mine = entries[args.shard::args.nshards]                # shard the cases across processes
    print(f"shard {args.shard}/{args.nshards}: {len(mine)} cases -> {droot}", flush=True)
    done = err = 0
    for e in mine:
        try:
            build_case(e["name"], img_dir, lbl_dir)
        except Exception as ex2:  # noqa: BLE001
            err += 1
            if err <= 5:
                print(f"  ERROR {e['name']}: {ex2!r}", flush=True)
        done += 1
        if done % 50 == 0:
            print(f"  shard{args.shard}: {done}/{len(mine)} (errors {err})", flush=True)

    if args.shard != 0:
        print(f"shard {args.shard} done ({err} errors)", flush=True)
        return
    labels = {"background": 0, **{c: CODE2IDX[c] for c in CODES14}}
    ds = {"channel_names": {"0": "CT"}, "labels": labels,
          "numTraining": len(entries), "file_ending": ".nii.gz"}
    json.dump(ds, open(droot / "dataset.json", "w"), indent=2)
    print(f"wrote dataset.json (14 classes), numTraining={len(entries)}", flush=True)
    print("NEXT: nnUNetv2_plan_and_preprocess -d 500 --verify_dataset_integrity ; "
          "voxtell-finetune 500 3d_fullres 0 -pretrained_weights "
          "<voxtell_v1.1>/fold_0/checkpoint_final.pth", flush=True)


if __name__ == "__main__":
    main()

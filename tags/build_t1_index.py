"""Build the Track-1 (free-text grounding) training index: one row per (case, finding).

Each ReXGroundingCT case carries a 4D mask `(F, H, W, D)` -- one channel per finding -- plus a
`findings` dict mapping the channel index to the free-text string. Track 1 conditions the
segmentation on that exact string, so the unit of training is the (volume, text, mask-channel)
triplet, not the case.

Images come from CT-RATE `_fixed` (proper HU + real affine), NOT the ReXGroundingCT `volumes/`
copy (CT-RATE's original release: raw values + stripped affine).

val-200 is the benchmark split and is written separately -- never mixed into train.

    python build_t1_index.py
"""
import json
import os
from collections import Counter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import REXG_ROOT, CTRATE_ROOT  # noqa: E402
RX = str(REXG_ROOT)
CT = str(CTRATE_ROOT)
OUT = str(Path(__file__).resolve().parent / "index")
Path(OUT).mkdir(exist_ok=True)
DS = f"{RX}/MICCAI_challenge_dataset.json"


def image_path(case: str) -> str:
    """CT-RATE `_fixed` path for a ReXGroundingCT case name (no .nii.gz)."""
    split = case.split("_")[0]
    pid = "_".join(case.split("_")[:2])
    acc = "_".join(case.split("_")[:3])
    sub = "valid_fixed" if split == "valid" else "train_fixed"
    return f"{CT}/{sub}/{pid}/{acc}/{case}.nii.gz"


def main() -> None:
    ds = json.load(open(DS))
    for split in ("train", "val"):
        rows, missing_img, missing_seg = [], set(), set()
        cats, per_case = Counter(), []
        for case in ds[split]:
            name = case["name"]
            stem = name[:-7] if name.endswith(".nii.gz") else name
            img = image_path(stem)
            seg = f"{RX}/segmentations/{name}"
            if not os.path.exists(img):
                missing_img.add(stem)
            if not os.path.exists(seg):
                missing_seg.add(stem)
            findings = case["findings"]
            per_case.append(len(findings))
            for fidx, text in findings.items():
                code = case.get("categories", {}).get(fidx)
                cats[code] += 1
                rows.append({
                    "case": stem,
                    "fidx": int(fidx),                       # channel index into the 4D mask
                    "text": text,                            # the EXACT free-text prompt
                    "category": code,                        # auxiliary signal only
                    "entity_count": case.get("entity_counts", {}).get(fidx),
                    "shape": case.get("shape"),
                    "image": img,
                    "seg": seg,
                })

        path = f"{OUT}/t1_index_{split}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        print(f"=== {split} ===")
        print(f"  cases            : {len(ds[split])}")
        print(f"  findings (rows)  : {len(rows)}   ({sum(per_case)/len(per_case):.2f} per case)")
        print(f"  missing images   : {len(missing_img)}  {sorted(missing_img)[:3] if missing_img else ''}")
        print(f"  missing segs     : {len(missing_seg)}  {sorted(missing_seg)[:3] if missing_seg else ''}")
        print(f"  categories       : {dict(sorted((k, v) for k, v in cats.items() if k))}")
        print(f"  -> {path}\n")


if __name__ == "__main__":
    main()

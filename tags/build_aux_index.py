"""Build an auxiliary training index from CT-RATE reports + TotalSegmentator masks.

ReXGroundingCT gives only 1743 nodule (2d) and 237 effusion (2e) findings. CT-RATE has the
SAME images (ReXGroundingCT is annotations on top of CT-RATE) with ~25k reports, every finding
already mapped by the LLM mapper to a 14-category label + TotalSegmentator `ts_labels`. Two
TotalSegmentator tasks give matching masks:

    ts_lung_nodules               16,300 studies   -> category 2d
    ts_pleural_pericard_effusion  13,135 studies   -> category 2e

So we can mint (image, REAL free-text finding, mask) triples at scale, in the exact challenge
domain, with no prompt synthesis and no DICOM conversion.

Two things this does carefully:

* **Leakage.** ReXGroundingCT is built ON CT-RATE, so any CT-RATE study that is also a
  challenge val/test case must be excluded or the val score is meaningless. Every case in the
  Track-1 val index is dropped, and `--exclude-train` additionally drops challenge TRAIN cases
  so the aux set is disjoint from all annotated data.

* **The mask is per-SCAN, the finding is per-LESION.** The TS mask covers every nodule in the
  volume, while the finding describes one ("nodule in the right lower lobe"). Where the
  finding's `ts_labels` name lobes, the mask is intersected with those lobes to make it
  finding-specific. Findings naming no lobe keep the whole mask (correct for effusion, which
  is not inside a lobe; lossy for a nodule, hence `--require-lobe` for 2d).

These are PSEUDO-labels (TotalSegmentator predictions, not radiologist ground truth) and should
be treated as weak supervision — pretrain/mix, do not fine-tune to convergence on them alone.

    python build_aux_index.py --out index/aux_ctrate.jsonl --limit 0
"""
import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(HERE)); from paths import LLM_OUT, CTRATE_ROOT  # noqa: E402,E401
MAP_OUT = LLM_OUT
CTRATE = CTRATE_ROOT
TS_SEG = CTRATE / "ts_seg"
# TotalSegmentator emits MULTI-LABEL masks; taking `>0` grabs the whole LUNG (~8.7M voxels)
# instead of the lesion (~200). Authoritative map (totalsegmentator.map_to_binary.class_map):
#   lung_nodules              -> {1: lung,         2: lung_nodules}
#   pleural_pericard_effusion -> {1: lung_pleural, 2: pleural_effusion, 3: pericardial_effusion}
# Category 2e is PLEURAL effusion/thickening -> label 2, NOT 3 (pericardial sits around the
# heart and is a different structure entirely).
TASK_FOR_CAT = {"2d": ("ts_lung_nodules", 2), "2e": ("ts_pleural_pericard_effusion", 2)}
# TotalSegmentator lung-lobe label ids (in the ts_total map), for the lobe intersection
LOBE_ID = {"lung_upper_lobe_left": 10, "lung_lower_lobe_left": 11,
           "lung_upper_lobe_right": 12, "lung_middle_lobe_right": 13,
           "lung_lower_lobe_right": 14}


def split_of(scan: str) -> str:
    """'train_1_a_1.nii.gz' -> 'train_fixed' / 'valid_fixed' (CT-RATE dir naming)."""
    return "valid_fixed" if scan.startswith("valid") else "train_fixed"


def paths_for(scan: str, task: str):
    """(image, mask, ts_total) absolute paths for a CT-RATE scan filename."""
    stem = scan[:-len(".nii.gz")] if scan.endswith(".nii.gz") else scan
    parts = stem.split("_")                       # train_1_a_1 -> train_1 / train_1_a
    study, series = "_".join(parts[:2]), "_".join(parts[:3])
    sp = split_of(scan)
    img = CTRATE / sp / study / series / scan
    mask = TS_SEG / task / sp / study / series / scan
    total = TS_SEG / "ts_total" / sp / study / series / scan
    return img, mask, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "index" / "aux_ctrate.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="max records (0 = all)")
    ap.add_argument("--require-lobe", action="store_true",
                    help="keep 2d findings only when ts_labels name a lobe (mask is then "
                         "finding-specific rather than every nodule in the scan)")
    ap.add_argument("--exclude-train", action="store_true",
                    help="also drop challenge TRAIN cases, not just val (fully disjoint aux set)")
    a = ap.parse_args()

    # ---- leakage guard: every case the challenge annotates -------------------------------
    banned = set()
    for fn in (["t1_index_val.jsonl"] + (["t1_index_train.jsonl"] if a.exclude_train else [])):
        p = HERE / "index" / fn
        if not p.exists():
            continue
        for line in open(p):
            r = json.loads(line)
            # 'train_13082_a_1' -> study 'train_13082'
            banned.add("_".join(r["case"].split("_")[:2]))
    print(f"[leakage] excluding {len(banned)} challenge studies", flush=True)

    out_rows, stats = [], {"kept": 0, "no_mask": 0, "leak": 0, "no_lobe": 0, "no_cat": 0}
    with open(MAP_OUT / "ctrate_categories.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            scans = r.get("scan_ids") or []
            if not scans:
                continue
            study = "_".join(scans[0].split("_")[:2])
            if study in banned:
                stats["leak"] += 1
                continue
            for fi in r.get("findings", []):
                cat = fi.get("category")
                if cat not in TASK_FOR_CAT or fi.get("polarity") != "present":
                    stats["no_cat"] += 1
                    continue
                lobes = sorted({LOBE_ID[n] for n in (fi.get("ts_labels") or []) if n in LOBE_ID})
                if cat == "2d" and a.require_lobe and not lobes:
                    stats["no_lobe"] += 1
                    continue
                task, mlabel = TASK_FOR_CAT[cat]
                for scan in scans:
                    img, mask, total = paths_for(scan, task)
                    if not (os.path.exists(mask) and os.path.exists(img)):
                        stats["no_mask"] += 1
                        continue
                    out_rows.append({
                        "case": scan[:-len(".nii.gz")], "text": fi["text"], "category": cat,
                        "image": str(img), "mask": str(mask), "mask_label": mlabel,
                        "ts_total": str(total) if os.path.exists(total) else None,
                        "lobes": lobes,           # intersect the mask with these at load time
                        "source": "ctrate_aux", "pseudo": True,
                    })
                    stats["kept"] += 1
            if a.limit and len(out_rows) >= a.limit:
                break

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as fh:
        for row in out_rows:
            fh.write(json.dumps(row) + "\n")
    ncase = len({r["case"] for r in out_rows})
    n2d = sum(1 for r in out_rows if r["category"] == "2d")
    nlobe = sum(1 for r in out_rows if r["lobes"])
    print(f"[aux] wrote {len(out_rows)} findings / {ncase} scans -> {outp}")
    print(f"      2d(nodule)={n2d}  2e(effusion)={len(out_rows)-n2d}  with-lobe={nlobe}")
    print(f"      skipped: no_mask={stats['no_mask']} leak_studies={stats['leak']} "
          f"no_lobe={stats['no_lobe']}")


if __name__ == "__main__":
    main()

"""Validate the 300 test predictions against the challenge spec, then zip them FLAT.

Checks each file against the challenge json before packaging, because a malformed submission
fails silently at evaluation time: channel count must equal the case's finding count and be in
the json's key order, spatial shape must match the case's declared shape, dtype uint8, values
in {0,1}.
"""
import argparse, json, zipfile
from pathlib import Path
import numpy as np, nibabel as nib

HERE = Path(__file__).resolve().parent
_ap = argparse.ArgumentParser()
_ap.add_argument("--src", default=str(HERE / "submission" / "SUBMISSION_1ch"),
                 help="directory of .nii.gz predictions to validate and zip")
_ap.add_argument("--zip", dest="zip_out", default="",
                 help="output path; defaults to <src>.zip alongside it")
_a = _ap.parse_args()
SRC = Path(_a.src)
ZIP = Path(_a.zip_out) if _a.zip_out else SRC.with_suffix(".zip")
import sys; sys.path.insert(0, str(HERE)); from paths import CHALLENGE_JSON  # noqa: E402,E401
d = json.load(open(CHALLENGE_JSON))
items = list(d['test'].values()) if isinstance(d['test'], dict) else d['test']

bad, empty, tot = [], 0, 0
files = []
for e in items:
    stem = e['name'][:-7] if e['name'].endswith('.nii.gz') else e['name']
    p = SRC / f"{stem}.nii.gz"
    if not p.exists():
        bad.append(f"MISSING {stem}"); continue
    im = nib.load(p); a = np.asanyarray(im.dataobj)
    nf = len(e['findings'])
    if a.ndim != 4 or a.shape[0] != nf: bad.append(f"{stem}: shape {a.shape}, expected ({nf},...)")
    if a.dtype != np.uint8: bad.append(f"{stem}: dtype {a.dtype}")
    if a.max() > 1: bad.append(f"{stem}: values >1 present (max {int(a.max())})")
    if e.get('shape') and tuple(a.shape[1:]) != tuple(e['shape']):
        bad.append(f"{stem}: spatial {a.shape[1:]} vs declared {tuple(e['shape'])}")
    for j in range(a.shape[0]):
        tot += 1
        if not a[j].any(): empty += 1
    files.append(p)

print(f"[check] {len(files)}/{len(items)} files | {tot} findings | empty masks {empty} ({100*empty/max(tot,1):.1f}%)")
print(f"[check] problems: {len(bad)}")
for b in bad[:10]: print("   ", b)
if bad:
    print("[check] NOT zipping — fix the problems above first"); raise SystemExit(1)

with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for p in files:
        z.write(p, arcname=p.name)          # arcname = flat, no directory prefix
print(f"[zip] {ZIP}  ({ZIP.stat().st_size/1e6:.0f} MB, {len(files)} files, flat)")
with zipfile.ZipFile(ZIP) as z:
    names = z.namelist()
    print(f"[zip] verify: {len(names)} entries, no folders: {all('/' not in n for n in names)}")
    print(f"[zip] first entries: {names[:3]}")

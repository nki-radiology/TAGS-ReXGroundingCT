#!/usr/bin/env python3
# lung_regions.py -- SELF-CONTAINED region-target generator for ReXGroundingCT.
#
# Given a CT-RATE case (CT.nii.gz for the affine, TotalSegmentator ts_total
# multilabel for the lung mask) and a finding's free text, produce a SOFT 3D
# region-target volume in the scan's voxel grid: the predefined anatomical
# region the finding should fall in. Use as the auxiliary-head target
# (soft-Dice / BCE), replacing the old lobe auxiliary.
#
# Design (fixed, calibrated on val):
#   - Anchors: lung mask + left/right side only. NO fissures, NO lobe gates
#     (measured: hard lobe clip keeps only 77% of finding mass, laterality 86%).
#   - Frame per lung voxel: SI% (base->apex, per side), AP% (ant->post),
#     ML% (medial->lateral, per side), depth_mm (EDT from pleural surface).
#   - 12 feathered primitives; vocabulary regions = products (intersections).
#   - All boundaries are SOFT (sigmoid feather) so spillover is down-weighted,
#     never zeroed.
#
# Usage:
#   from lung_regions import RegionGenerator
#   gen = RegionGenerator(ct_path, ts_path)        # once per scan
#   target = gen.target_for_text(finding_text)     # float32 volume in [0,1]
#
# CLI self-test on one case (prints per-finding coverage lift):
#   python lung_regions.py <case_dir> <ts_path>
import gzip, struct, re, os, sys
import numpy as np
from scipy import ndimage

LUNG_LABELS = (10, 11, 12, 13, 14)          # TS v2 lobes; used ONLY as lung+side
FEATHER = 6.0                                # % feather on SI/AP/ML boundaries
FEATHER_MM = 3.0                             # mm feather on depth boundaries


# ---------------------------------------------------------------- nii reading
def read_nii(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    end = "<" if struct.unpack_from("<i", raw, 0)[0] == 348 else ">"
    dim = struct.unpack_from(end + "8h", raw, 40)
    bitpix = struct.unpack_from(end + "h", raw, 72)[0]
    vox = int(struct.unpack_from(end + "f", raw, 108)[0])
    srow = np.array([struct.unpack_from(end + "4f", raw, off) for off in (280, 296, 312)])
    ni, nj, nk = dim[1], dim[2], dim[3]
    dt = {8: np.uint8, 16: np.int16, 32: np.float32}[bitpix]
    arr = np.frombuffer(raw[vox:vox + ni * nj * nk * (bitpix // 8)], dt).reshape(nk, nj, ni)
    return arr, srow

def read_srow(path):
    with gzip.open(path, "rb") as f:
        raw = f.read(400)
    end = "<" if struct.unpack_from("<i", raw, 0)[0] == 348 else ">"
    return np.array([struct.unpack_from(end + "4f", raw, off) for off in (280, 296, 312)])


def _sig(x, w):
    return 1.0 / (1.0 + np.exp(-np.clip(x / max(w, 1e-6), -60, 60)))


# ---------------------------------------------------------------- vocabulary
# term -> list of (axis, op, thr) soft constraints; laterality handled separately.
# Compounds are parsed BEFORE their parts (order matters).
VOCAB = [
    ("posterobasal",      [("SI", "<", 40), ("AP", ">", 50)]),
    ("laterobasal",       [("SI", "<", 40), ("ML", ">", 50)]),
    ("mediobasal",        [("SI", "<", 40), ("ML", "<", 50)]),
    ("anterobasal",       [("SI", "<", 40), ("AP", "<", 50)]),
    ("apicoposterior",    [("SI", ">", 70), ("AP", ">", 50)]),
    ("superior segment",  [("SI", ">", 40), ("SI", "<", 70), ("AP", ">", 50)]),
    ("anterior segment",  [("AP", "<", 50)]),
    ("posterior segment", [("AP", ">", 50)]),
    ("medial segment",    [("ML", "<", 50)]),
    ("lateral segment",   [("ML", ">", 50)]),
    ("lingular",          [("AP", "<", 50), ("SI", ">", 15), ("SI", "<", 45)]),   # + LEFT
    ("lingula",           [("AP", "<", 50), ("SI", ">", 15), ("SI", "<", 45)]),   # + LEFT
    ("middle lobe",       [("AP", "<", 55), ("SI", ">", 25), ("SI", "<", 55)]),   # + RIGHT
    ("upper lobe",        [("SI", ">", 45)]),
    ("lower lobe",        [("SI", "<", 55)]),
    ("subpleural",        [("DEPTH", "<", 8)]),
    ("pleural-based",     [("DEPTH", "<", 8)]),
    ("peripheral",        [("DEPTH", "<", 12)]),
    ("perihilar",         [("DEPTH", ">", 12), ("ML", "<", 50)]),
    ("central",           [("DEPTH", ">", 12)]),
    ("apices",            [("SI", ">", 70)]),
    ("apical",            [("SI", ">", 70)]),
    ("apex",              [("SI", ">", 70)]),
    ("basal",             [("SI", "<", 40)]),
    ("basilar",           [("SI", "<", 40)]),
    ("juxtadiaphragmatic",[("SI", "<", 25)]),
    ("dependent",         [("AP", ">", 55)]),          # supine
    ("retrocardiac",      [("AP", ">", 50), ("ML", "<", 45)]),  # + LEFT
    ("paramediastinal",   [("ML", "<", 35)]),
    ("paraspinal",        [("AP", ">", 55), ("ML", "<", 40)]),
    ("paravertebral",     [("AP", ">", 55), ("ML", "<", 40)]),
    ("posterior",         [("AP", ">", 50)]),
    ("anterior",          [("AP", "<", 50)]),
    ("superior",          [("SI", ">", 55)]),
    ("inferior",          [("SI", "<", 45)]),
    ("medial",            [("ML", "<", 50)]),
    ("lateral",           [("ML", ">", 50)]),
    # adverb / adjective variants and indirect location cues
    ("peripherally",      [("DEPTH", "<", 12)]),
    ("periphery",         [("DEPTH", "<", 12)]),
    ("centrally",         [("DEPTH", ">", 12)]),
    ("apically",          [("SI", ">", 70)]),
    ("basally",           [("SI", "<", 40)]),
    ("posteriorly",       [("AP", ">", 50)]),
    ("anteriorly",        [("AP", "<", 50)]),
    ("medially",          [("ML", "<", 50)]),
    ("laterally",         [("ML", ">", 50)]),
    ("superiorly",        [("SI", ">", 55)]),
    ("inferiorly",        [("SI", "<", 45)]),
    ("paraseptal",        [("DEPTH", "<", 8)]),        # paraseptal = subpleural by definition
    ("pleural based",     [("DEPTH", "<", 8)]),
    ("peribronchovascular", [("DEPTH", ">", 10)]),     # airway-centered -> soft central
    ("peribronchial",     [("DEPTH", ">", 10)]),
    ("bronchovascular",   [("DEPTH", ">", 10)]),
]
FORCE_LEFT = {"lingula", "lingular", "retrocardiac"}
FORCE_RIGHT = {"middle lobe"}


def parse_text(text):
    """-> (constraints list, side in {'left','right','both',None})"""
    t = text.lower()
    def has(p):
        return re.search(r"(?<![a-z])" + p + r"(?![a-z])", t) is not None
    both = has(r"both lungs?") or has(r"bilateral(?:ly)?") or has(r"both (?:upper|lower) lobes?") \
        or has(r"in both")
    right = has(r"right")
    left = has(r"left")
    side = "both" if both else ("right" if right and not left else
                                ("left" if left and not right else
                                 ("both" if right and left else None)))
    # bilateral but asymmetric: "more prominent/pronounced/marked on the right"
    m = re.search(r"more (?:prominent|pronounced|marked|severe)(?:ly)?\s+(?:on|in)\s+the\s+(right|left)", t)
    if m and side in ("both", None):
        side = m.group(1) + "-lean"
    terms, used_spans = [], []
    for term, c in VOCAB:
        for m in re.finditer(r"(?<![a-z])" + re.escape(term) + r"s?(?![a-z])", t):
            if any(m.start() >= s and m.end() <= e for s, e in used_spans):
                continue
            used_spans.append((m.start(), m.end()))
            terms.append((term, c))
            if term in FORCE_LEFT and side is None:
                side = "left"
            if term in FORCE_RIGHT and side is None:
                side = "right"
            break
    return terms, side


# ---------------------------------------------------------------- generator
class RegionGenerator:
    def __init__(self, ct_path, ts_path):
        ts, _ = read_nii(ts_path)
        srow = read_srow(ct_path)
        lung = np.isin(ts, LUNG_LABELS)
        if not lung.any():
            raise ValueError("no lung labels in %s" % ts_path)
        self.shape = lung.shape
        k, j, i = np.nonzero(lung)
        self.idx = (k, j, i)
        ijk = np.stack([i, j, k], 1).astype(np.float64)
        ras = ijk @ srow[:, :3].T + srow[:, 3]
        R, A, S = ras[:, 0], ras[:, 1], ras[:, 2]
        lab = ts[k, j, i]
        isR = lab >= 12
        if not isR.any() or isR.all():          # fallback: split at lung R midpoint
            isR = R > np.median(R)
        # per-side extents
        def ext(v, m):
            return v[m].min(), v[m].max()
        SmnR, SmxR = ext(S, isR); SmnL, SmxL = ext(S, ~isR)
        Amn, Amx = S_ = A.min(), A.max()
        right_high = R[isR].mean() > R[~isR].mean()
        medR, latR = (R[isR].min(), R[isR].max()) if right_high else (R[isR].max(), R[isR].min())
        medL, latL = (R[~isR].max(), R[~isR].min()) if right_high else (R[~isR].min(), R[~isR].max())
        self.SI = np.where(isR, (S - SmnR) / (SmxR - SmnR + 1e-6),
                                (S - SmnL) / (SmxL - SmnL + 1e-6)) * 100
        self.AP = (Amx - A) / (Amx - Amn + 1e-6) * 100
        self.ML = np.where(isR, (R - medR) / (latR - medR + 1e-6),
                                (R - medL) / (latL - medL + 1e-6)) * 100
        self.isR = isR
        # depth: EDT on 2x-downsampled lung bbox (fast, ~3mm accurate)
        sp = np.linalg.norm(srow[:, :3], axis=0)            # (i,j,k) spacings
        samp = (sp[2] * 2, sp[1] * 2, sp[0] * 2)
        pad = 8
        k0, k1 = max(0, k.min() - pad), min(lung.shape[0], k.max() + pad + 1)
        j0, j1 = max(0, j.min() - pad), min(lung.shape[1], j.max() + pad + 1)
        i0, i1 = max(0, i.min() - pad), min(lung.shape[2], i.max() + pad + 1)
        dt2 = ndimage.distance_transform_edt(lung[k0:k1, j0:j1, i0:i1][::2, ::2, ::2],
                                             sampling=samp)
        kk = np.clip((k - k0) // 2, 0, dt2.shape[0] - 1)
        jj = np.clip((j - j0) // 2, 0, dt2.shape[1] - 1)
        ii = np.clip((i - i0) // 2, 0, dt2.shape[2] - 1)
        self.DEPTH = dt2[kk, jj, ii]

    def _axis(self, name):
        return {"SI": self.SI, "AP": self.AP, "ML": self.ML, "DEPTH": self.DEPTH}[name]

    def _term_mask(self, cons):
        w = np.ones(len(self.SI), dtype=np.float32)
        for axis, op, thr in cons:
            v = self._axis(axis)
            f = FEATHER_MM if axis == "DEPTH" else FEATHER
            w *= _sig(v - thr, f) if op == ">" else _sig(thr - v, f)
        return w

    def target_for_text(self, text):
        """Soft region volume (float32, full grid, 0 outside lung).

        Terms are combined conflict-aware: a term that overlaps the running
        region refines it (product = intersection); a term that is spatially
        disjoint from it names an ADDITIONAL site (max = union) — e.g.
        "scarring in the apices and posterobasal segments".
        """
        terms, side = parse_text(text)
        w = None
        for _, cons in terms:
            tw = self._term_mask(cons)
            if w is None:
                w = tw
                continue
            inter = float((w * tw).mean())
            lo = min(float(w.mean()), float(tw.mean()))
            if lo > 1e-6 and inter / lo < 0.2:      # disjoint -> additional site
                w = np.maximum(w, tw)
            else:                                    # overlapping -> refinement
                w = w * tw
        if w is None:
            w = np.ones(len(self.SI), dtype=np.float32)
        if side == "right":
            w *= np.where(self.isR, 1.0, 0.15)      # soft laterality (not a hard gate)
        elif side == "left":
            w *= np.where(self.isR, 0.15, 1.0)
        elif side == "right-lean":                   # bilateral, more prominent right
            w *= np.where(self.isR, 1.0, 0.5)
        elif side == "left-lean":
            w *= np.where(self.isR, 0.5, 1.0)
        out = np.zeros(self.shape, dtype=np.float32)
        out[self.idx] = w
        return out


# ---------------------------------------------------------------- self-test
def _selftest(case_dir, ts_path):
    gen = RegionGenerator(os.path.join(case_dir, "CT.nii.gz"), ts_path)
    fmeta = {}
    with open(os.path.join(case_dir, "findings.txt")) as f:
        for n, line in enumerate(f):
            p = line.rstrip("\n").split("\t")
            if n == 0 or len(p) < 4:
                continue
            fmeta[int(p[0])] = p[3]
    print("%-4s %-9s %-9s %-6s  text" % ("val", "coverage", "volfrac", "lift"))
    for fn in sorted(os.listdir(case_dir)):
        m = re.match(r"f(\d+)_.*\.nii\.gz$", fn)
        if not m:
            continue
        value = int(m.group(1)) + 1
        text = fmeta.get(value, "")
        mask, _ = read_nii(os.path.join(case_dir, fn))
        mask = mask > 0
        if not mask.any():
            continue
        tgt = gen.target_for_text(text)
        cov = float(tgt[mask].mean())                       # region prob at GT voxels
        volfrac = float(tgt[gen.idx].mean())                # region prob over lung
        lift = cov / max(volfrac, 1e-6)
        print("%-4d %-9.3f %-9.3f %-6.2f  %s" % (value, cov, volfrac, lift, text[:70]))

if __name__ == "__main__":
    _selftest(sys.argv[1], sys.argv[2])

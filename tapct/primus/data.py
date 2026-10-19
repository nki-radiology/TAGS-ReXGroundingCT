"""Crop sampler + volume loader over the primus_cache format (see build_cache.py)."""
import json, os, random
import numpy as np
import torch
from torch.utils.data import Dataset

CLIP = (-1008.0, 822.0)
MEAN, STD = -86.80862426757812, 322.63470458984375     # TAP-CT normalisation
WIN = (int(os.environ.get('PRIMUS_WIN_Z', 12)),
       int(os.environ.get('PRIMUS_WIN_YX', 224)), int(os.environ.get('PRIMUS_WIN_YX', 224)))   # (z, y, x); env-overridable per run
NORM = os.environ.get('PRIMUS_NORM', 'tapct')   # 'tapct' (clip/mean/std) or 'spectre' (HU [-1000,1000] -> [0,1])
SPACING = (float(os.environ.get('PRIMUS_Z_MM', 1.5)), 1.5, 1.5)   # cache voxel size (z, y, x) mm
VOX_MM3 = SPACING[0] * SPACING[1] * SPACING[2]


def normalise(ct_int16):
    x = torch.as_tensor(np.asarray(ct_int16, dtype=np.float32))
    if NORM == 'spectre':                      # SPECTRE's own ScaleIntensityRange: HU [-1000, 1000] -> [0, 1]
        return (x.clamp(-1000.0, 1000.0) + 1000.0) / 2000.0
    if NORM == 'ctclip':                       # CT-CLIP data.py: clip HU to [-1000, 1000], divide by 1000
        return x.clamp(-1000.0, 1000.0) / 1000.0
    if NORM == 'ctfm':                         # CT-FM README: ScaleIntensityRange HU [-1024, 2048] -> [0, 1], clipped
        return (x.clamp(-1024.0, 2048.0) + 1024.0) / 3072.0
    if NORM == 'hu':                           # raw HU; the encoder does its own windowing (Pillar-0)
        return x
    return (x.clamp(*CLIP) - MEAN) / STD


def list_cache(root):
    stems = sorted(f[:-10] for f in os.listdir(root) if f.endswith('_meta.json'))
    return stems


def load_meta(root, stem):
    return json.load(open(os.path.join(root, f'{stem}_meta.json')))


def load_volume(root, stem, mmap=True):
    mode = 'r' if mmap else None
    ct = np.load(os.path.join(root, f'{stem}_ct.npy'), mmap_mode=mode)
    seg = np.load(os.path.join(root, f'{stem}_seg.npy'), mmap_mode=mode)
    inst = np.load(os.path.join(root, f'{stem}_inst.npy'), mmap_mode=mode)
    return ct, seg, inst


def split_stems(root, dataset, val_list=None, seed=0, val_frac=0.15):
    """Returns (train_stems, val_stems). rexground: challenge split from meta; pancancer: from val_list (nnU-Net
    fold-0 validation case names) else a seeded random split."""
    stems = list_cache(root)
    if dataset == 'rexground':
        tr, va = [], []
        for s in stems:
            (va if load_meta(root, s).get('split') == 'val' else tr).append(s)
        return tr, va
    if val_list:
        vs = set(val_list)
        va = [s for s in stems if s in vs]; tr = [s for s in stems if s not in vs]
        return tr, va
    rng = random.Random(seed); st = list(stems); rng.shuffle(st)
    n = int(len(st) * val_frac)
    return sorted(st[n:]), sorted(st[:n])


class CropDataset(Dataset):
    """Infinite sampler of (12,224,224) windows. fg_prob of the draws are centred (with jitter) on a foreground
    voxel of a uniformly chosen instance; the rest are uniform within the body bounding box."""

    def __init__(self, root, stems, win=WIN, fg_prob=0.5, length=10**7, seed=0, jitter=(4, 64, 64), flip=True, binary=False):
        self.root, self.stems, self.win, self.fg_prob, self.length = root, list(stems), win, fg_prob, length
        self.binary = binary
        self.seed, self.jitter, self.flip = seed, jitter, flip
        self.meta = {s: load_meta(root, s) for s in self.stems}
        self.with_fg = [s for s in self.stems if self.meta[s]['instances']]
        self._vols = {}

    def __len__(self):
        return self.length

    def _vol(self, s):
        # Open per item and let the memmaps be released right after slicing. Keeping memmaps open across items
        # made every worker progressively map the whole cache (161 GB for ReXGroundingCT): mapped file pages count
        # against the job's cgroup and the job was OOM-killed at 120 GB host RAM (2026-08-23).
        return load_volume(self.root, s)

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1000003 + i + os.getpid() * 7919)
        fg = rng.random() < self.fg_prob and self.with_fg
        s = rng.choice(self.with_fg if fg else self.stems)
        m = self.meta[s]; shape = m['shape']
        w = self.win
        if fg:
            ins = rng.choice(m['instances'])
            c = rng.choice(ins['fg_samples'])
            c = [c[k] + rng.randint(-self.jitter[k], self.jitter[k]) for k in range(3)]
            lo = [min(max(c[k] - w[k] // 2, 0), max(shape[k] - w[k], 0)) for k in range(3)]
        else:
            b = m.get('body') or [[0, 0, 0], [s_ - 1 for s_ in shape]]
            lo = []
            for k in range(3):
                a, z = b[0][k], max(b[1][k] - w[k] + 1, b[0][k])
                lo.append(min(max(rng.randint(a, z), 0), max(shape[k] - w[k], 0)))
        ct, seg, inst = self._vol(s)
        sl = tuple(slice(lo[k], lo[k] + w[k]) for k in range(3))
        x = np.array(ct[sl]); y = np.array(seg[sl])
        del ct, seg, inst
        # pad (volumes thinner than the window in some axis)
        pad = [(0, w[k] - x.shape[k]) for k in range(3)]
        if any(p[1] > 0 for p in pad):
            x = np.pad(x, pad, constant_values=-1024); y = np.pad(y, pad, constant_values=0)
        if self.binary:
            y = (y > 0).astype(np.uint8)
        x = normalise(x); y = torch.from_numpy(y.astype(np.int64))
        if self.flip and rng.random() < 0.5:   # L/R flip only (anatomy-plausible)
            x = x.flip(-1); y = y.flip(-1)
        return x[None], y


def sliding_windows(shape, win=WIN, stride=(6, 112, 112)):
    """Start indices covering `shape` with windows `win` (last window clamped to the edge)."""
    starts = []
    for k in range(3):
        n, w, st = shape[k], win[k], stride[k]
        if n <= w:
            s = [0]
        else:
            s = list(range(0, n - w, st)) + [n - w]
        starts.append(sorted(set(s)))
    return [(a, b, c) for a in starts[0] for b in starts[1] for c in starts[2]]


def gaussian_weight(win=WIN, sigma_scale=1 / 8):
    ws = []
    for w in win:
        x = np.arange(w) - (w - 1) / 2
        ws.append(np.exp(-0.5 * (x / (w * sigma_scale)) ** 2))
    g = ws[0][:, None, None] * ws[1][None, :, None] * ws[2][None, None, :]
    g = g / g.max()
    return torch.from_numpy(g.astype(np.float32)).clamp_min(1e-3)

"""Build a node-local, isotropic (1.5 mm) int16 cache for the Primus-TAP block-depth experiments.

Two datasets, one format:
  <out>/<stem>_ct.npy    int16  (D,H,W)  HU clipped to [-1024, 3071], LPS axis order (z,y,x)
  <out>/<stem>_seg.npy   uint8  (D,H,W)  class map (pancancer: 0/1; rexground: 0..14 = bg + 14 categories)
  <out>/<stem>_inst.npy  uint8  (D,H,W)  instance map (0 = bg). pancancer: 26-connected components of the
                                         lesion mask; rexground: finding index+1, smallest finding wins overlaps
  <out>/<stem>_meta.json                 spacing, shape, body bbox, per-instance {id, cls, n_vox, centroid, lo, hi,
                                         fg_samples (<=64 random fg voxels for crop sampling)}, source paths
Resampling: trilinear for CT, nearest for labels, to (1.5,1.5,1.5) mm.
Orientation: arrays are brought to (z,y,x) with L->R along x, A->P along y, I->S along z, i.e. the same axis
convention the lattice/TAP-CT probing code used (SimpleITK LPS arrays).
"""
import argparse, json, os, re, sys, time, traceback
import numpy as np

TARGET = (float(os.environ.get('PRIMUS_Z_MM', 1.5)), 1.5, 1.5)   # (z, y, x) mm; PRIMUS_Z_MM=5 for thick slices
HU_MIN, HU_MAX = -1024, 3071
MAX_INST = 250

REX_SEG = os.path.join(os.environ.get("TAGS_REXG_ROOT", "data/ReXGroundingCT"), "segmentations")
REX_JSON = os.environ.get("TAGS_CHALLENGE_JSON", "data/ReXGroundingCT/MICCAI_challenge_dataset.json")
CT_TRAIN = os.path.join(os.environ.get("TAGS_CTRATE_ROOT", "data/CT-RATE/dataset"), 'train_fixed')
CT_VALID = os.path.join(os.environ.get("TAGS_CTRATE_ROOT", "data/CT-RATE/dataset"), 'valid_fixed')
REX_CATS = ['1a', '1b', '1c', '1d', '1e', '1f', '2a', '2b', '2c', '2d', '2e', '2f', '2g', '2h']
REX_CLS = {c: i + 1 for i, c in enumerate(REX_CATS)}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def resample(ct, labs, spacing):
    """ct (D,H,W) float32, labs list of (D,H,W) int arrays, spacing (z,y,x) -> resampled to TARGET."""
    import torch, torch.nn.functional as F
    torch.set_num_threads(2)
    D, H, W = ct.shape
    new = tuple(max(1, int(round(s * n / t))) for s, n, t in zip(spacing, (D, H, W), TARGET))
    t = torch.from_numpy(ct)[None, None]
    ct_r = F.interpolate(t, size=new, mode='trilinear', align_corners=False)[0, 0].numpy()
    out = []
    for l in labs:
        lt = torch.from_numpy(l.astype(np.float32))[None, None]
        out.append(F.interpolate(lt, size=new, mode='nearest')[0, 0].numpy().astype(l.dtype))
    return ct_r, out


def body_bbox(ct):
    above = ct > -377
    proj = [np.nonzero(above.any(axis=tuple(j for j in range(3) if j != k)))[0] for k in range(3)]
    if not all(a.size for a in proj):
        return None
    return [[int(a.min()) for a in proj], [int(a.max()) for a in proj]]


def instances_meta(inst, seg, rng, max_samples=64):
    ids = np.unique(inst); ids = ids[ids > 0]
    out = []
    for i in ids:
        nz = np.nonzero(inst == i)
        n = int(nz[0].size)
        cls = int(np.bincount(seg[nz]).argmax())
        sel = rng.choice(n, size=min(max_samples, n), replace=False)
        out.append(dict(id=int(i), cls=cls, n_vox=n,
                        centroid=[float(a.mean()) for a in nz],
                        lo=[int(a.min()) for a in nz], hi=[int(a.max()) for a in nz],
                        fg_samples=[[int(nz[k][j]) for k in range(3)] for j in sel]))
    return out


def write(out, stem, ct, seg, inst, meta):
    os.makedirs(out, exist_ok=True)
    ct16 = np.clip(np.round(ct), HU_MIN, HU_MAX).astype(np.int16)
    for name, arr in (('ct', ct16), ('seg', seg.astype(np.uint8)), ('inst', inst.astype(np.uint8))):
        p = os.path.join(out, f'{stem}_{name}.npy')
        np.save(p + '.tmp.npy', arr); os.replace(p + '.tmp.npy', p)
    meta.update(shape=list(ct16.shape), spacing=list(TARGET), body=body_bbox(ct16),
                hu_range=[int(ct16.min()), int(ct16.max())], version=1)
    mp = os.path.join(out, f'{stem}_meta.json')
    with open(mp + '.tmp', 'w') as fh:
        json.dump(meta, fh)
    os.replace(mp + '.tmp', mp)


def done(out, stem):
    return os.path.exists(os.path.join(out, f'{stem}_meta.json'))


# ----------------------------------------------------------------------------- pancancer
def to_lps(img):
    import SimpleITK as sitk
    d = np.array(img.GetDirection()).reshape(3, 3)
    if not np.allclose(d, np.eye(3), atol=1e-3):
        img = sitk.DICOMOrient(img, 'LPS')
    return img


def build_pancancer_one(label_path, image_path, out, stem, rng, dataset='pancancer'):
    import SimpleITK as sitk
    from scipy import ndimage
    im = to_lps(sitk.ReadImage(image_path)); lb = to_lps(sitk.ReadImage(label_path))
    ct = sitk.GetArrayFromImage(im).astype(np.float32)
    lab = sitk.GetArrayFromImage(lb)
    assert ct.shape == lab.shape, (ct.shape, lab.shape)
    # after reorientation both must describe the same physical grid, else the label is misaligned
    assert np.allclose(im.GetSpacing(), lb.GetSpacing(), atol=1e-3), (im.GetSpacing(), lb.GetSpacing())
    assert np.allclose(im.GetOrigin(), lb.GetOrigin(), atol=1.0), ('origin mismatch', im.GetOrigin(), lb.GetOrigin())
    assert np.allclose(im.GetDirection(), lb.GetDirection(), atol=1e-3), ('direction mismatch', im.GetDirection(), lb.GetDirection())
    sp = im.GetSpacing()[::-1]  # (z,y,x)
    assert ct.min() < -500 and ct.max() > 200, f'HU canary failed: {ct.min()} {ct.max()}'
    # instances: components of the ORIGINAL multi-label file if it carries instance ids, else 26-cc
    seg = (lab > 0).astype(np.uint8)
    inst, n = ndimage.label(seg, structure=np.ones((3, 3, 3)))
    if n > MAX_INST:  # keep the largest MAX_INST
        sizes = ndimage.sum(seg, inst, range(1, n + 1)); keep = np.argsort(sizes)[::-1][:MAX_INST] + 1
        m = np.isin(inst, keep); inst = np.where(m, inst, 0)
        inst = np.unique(inst, return_inverse=True)[1].reshape(inst.shape)
    ct_r, (seg_r, inst_r) = resample(ct, [seg, inst.astype(np.uint16)], sp)
    meta = dict(dataset=dataset, source_image=image_path, source_label=label_path,
                orig_spacing=list(sp), orig_shape=list(ct.shape), n_inst_orig=int(n))
    meta['instances'] = instances_meta(inst_r, seg_r, rng)
    write(out, stem, ct_r, seg_r, inst_r, meta)


def run_pancancer(args):
    pairs = json.load(open(args.pairs))
    items = sorted(pairs.items())
    rng = np.random.default_rng(args.task)
    for i, (lab, img) in enumerate(items):
        if i % args.n_tasks != args.task:
            continue
        stem = re.sub(r'[^A-Za-z0-9]+', '_', lab[:-5]).strip('_')
        if done(args.out, stem):
            continue
        t0 = time.time()
        try:
            build_pancancer_one(os.path.join(args.labels, lab), img, args.out, stem, rng)
            log(f'[{i}] {stem} ok {time.time()-t0:.1f}s')
        except Exception as e:
            log(f'[{i}] {stem} FAIL {e}'); traceback.print_exc()


def run_msd(args):
    """MSD / nnU-Net-raw task folder. Pairs labels<split>/X.nii.gz with images<split>/X.nii.gz or X_0000.nii.gz
    (single-channel). Binary foreground; instances = 26-connected components, exactly as for pancancer, so the
    caches are interchangeable for the block-depth evaluators. Used as an EXTERNAL test set (no training)."""
    split = args.msd_split
    lab_dir, img_dir = os.path.join(args.msd_dir, 'labels' + split), os.path.join(args.msd_dir, 'images' + split)
    labs = sorted(f for f in os.listdir(lab_dir) if f.endswith('.nii.gz') and not f.startswith('.'))
    if args.limit:
        labs = labs[:args.limit]
    rng = np.random.default_rng(args.task)
    for i, f in enumerate(labs):
        if i % args.n_tasks != args.task:
            continue
        img = os.path.join(img_dir, f)
        if not os.path.exists(img):
            img = os.path.join(img_dir, f[:-7] + '_0000.nii.gz')
        stem = re.sub(r'[^A-Za-z0-9]+', '_', f[:-7]).strip('_')
        if done(args.out, stem):
            continue
        t0 = time.time()
        try:
            build_pancancer_one(os.path.join(lab_dir, f), img, args.out, stem, rng,
                                dataset=os.path.basename(args.msd_dir.rstrip('/')).lower())
            log(f'[{i}] {stem} ok {time.time()-t0:.1f}s')
        except Exception as e:
            log(f'[{i}] {stem} FAIL {e}'); traceback.print_exc()


# ----------------------------------------------------------------------------- rexground
def rex_ct_path(name):
    base = name[:-7] if name.endswith('.nii.gz') else name
    parts = base.split('_')
    root = CT_TRAIN if parts[0] == 'train' else CT_VALID
    return os.path.join(root, '_'.join(parts[:2]), '_'.join(parts[:3]), base + '.nii.gz')


def nib_to_lps_zyx(arr, affine):
    """arr indexed (x,y,z) voxel axes [+ leading channel]; flip so codes become (L,P,S) then transpose to (z,y,x)."""
    import nibabel as nib
    codes = nib.aff2axcodes(affine)
    want = ('L', 'P', 'S')
    flips = [ax for ax, (c, w) in enumerate(zip(codes, want)) if c != w]
    for ax, (c, w) in enumerate(zip(codes, want)):
        assert c in (w, {'L': 'R', 'P': 'A', 'S': 'I'}[w]), f'unexpected axcodes {codes}'
    off = arr.ndim - 3
    if flips:
        arr = np.flip(arr, axis=[f + off for f in flips])
    perm = list(range(off)) + [off + 2, off + 1, off + 0]
    return np.ascontiguousarray(np.transpose(arr, perm))


def build_rex_one(entry, out, stem, rng):
    import nibabel as nib
    name = entry['name']
    ctimg = nib.load(rex_ct_path(name))
    ct = np.asanyarray(ctimg.dataobj).astype(np.float32)
    zooms = ctimg.header.get_zooms()[:3]  # (x,y,z)
    ct = nib_to_lps_zyx(ct, ctimg.affine)
    sp = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    assert ct.min() < -500 and ct.max() > 200, f'HU canary failed: {ct.min()} {ct.max()}'
    mimg = nib.load(os.path.join(REX_SEG, name))
    m = np.asanyarray(mimg.dataobj)
    if m.ndim == 3:
        m = m[None]
    assert m.ndim == 4 and m.shape[1:] == ctimg.shape[:3], (m.shape, ctimg.shape)   # (F, X, Y, Z) on the CT grid
    if not np.allclose(mimg.affine[:3, :3], ctimg.affine[:3, :3], atol=1e-2):
        log(f'WARN {name}: mask affine differs from CT affine; using CT affine for orientation')
    m = nib_to_lps_zyx(m, ctimg.affine)
    F_ = m.shape[0]
    cats = entry['categories']
    # instance map: smallest finding wins on overlap
    sizes = [(int((m[f] > 0.5).sum()), f) for f in range(F_)]
    inst = np.zeros(ct.shape, np.uint16); seg = np.zeros(ct.shape, np.uint8)
    for n, f in sorted(sizes, reverse=True):   # paint large first, small last -> small wins
        if n == 0:
            continue
        mk = m[f] > 0.5
        inst[mk] = f + 1
        seg[mk] = REX_CLS[cats[str(f)]]
    ct_r, (seg_r, inst_r) = resample(ct, [seg, inst], sp)
    meta = dict(dataset='rexground', name=name, split=entry.get('protocol'), source_image=rex_ct_path(name),
                orig_spacing=list(sp), orig_shape=list(ct.shape), n_findings=F_,
                findings={str(f): dict(category=cats[str(f)], text=entry['findings'].get(str(f), ''),
                                       n_orig=sizes[[s[1] for s in sizes].index(f)][0]) for f in range(F_)})
    meta['instances'] = instances_meta(inst_r, seg_r, rng)
    write(out, stem, ct_r, seg_r, inst_r, meta)


def run_rexground(args):
    d = json.load(open(REX_JSON))
    entries = []
    for split in ('train', 'val'):
        e = d[split]
        e = list(e.values()) if isinstance(e, dict) else e
        for x in e:
            x = dict(x); x['protocol'] = split; entries.append(x)
    entries.sort(key=lambda x: x['name'])
    if args.limit:
        entries = entries[:args.limit]
    rng = np.random.default_rng(args.task)
    for i, e in enumerate(entries):
        if i % args.n_tasks != args.task:
            continue
        stem = e['name'][:-7]
        if done(args.out, stem):
            continue
        t0 = time.time()
        try:
            build_rex_one(e, args.out, stem, rng)
            log(f'[{i}] {stem} ok {time.time()-t0:.1f}s')
        except Exception as ex:
            log(f'[{i}] {stem} FAIL {ex}'); traceback.print_exc()


def build_rex_test_one(entry, out, stem):
    """Challenge TEST scan: CT only (masks withheld). seg/inst are all-zero so the shared loaders work; the finding
    texts come from the challenge json so export_challenge.py can look up their embeddings."""
    import nibabel as nib
    name = entry['name']
    ctimg = nib.load(rex_ct_path(name))
    ct = np.asanyarray(ctimg.dataobj).astype(np.float32)
    zooms = ctimg.header.get_zooms()[:3]
    ct = nib_to_lps_zyx(ct, ctimg.affine)
    sp = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    assert ct.min() < -500 and ct.max() > 200, f'HU canary failed: {ct.min()} {ct.max()}'
    zeros = np.zeros(ct.shape, np.uint8)
    ct_r, (seg_r, inst_r) = resample(ct, [zeros, zeros], sp)
    cats = entry.get('categories', {})
    meta = dict(dataset='rexground', name=name, split='test', source_image=rex_ct_path(name),
                orig_spacing=list(sp), orig_shape=list(ct.shape), n_findings=len(entry['findings']),
                findings={str(f): dict(category=cats.get(str(f)), text=t, n_orig=-1) for f, t in entry['findings'].items()},
                instances=[])
    write(out, stem, ct_r, seg_r, inst_r, meta)


def run_rexground_test(args):
    entries = json.load(open(REX_JSON))['test']
    entries = list(entries.values()) if isinstance(entries, dict) else entries
    entries.sort(key=lambda x: x['name'])
    if args.limit:
        entries = entries[:args.limit]
    for i, e in enumerate(entries):
        if i % args.n_tasks != args.task:
            continue
        stem = e['name'][:-7]
        if done(args.out, stem):
            continue
        t0 = time.time()
        try:
            build_rex_test_one(e, args.out, stem)
            log(f'[{i}] {stem} ok {time.time()-t0:.1f}s')
        except Exception as ex:
            log(f'[{i}] {stem} FAIL {ex}'); traceback.print_exc()


# ----------------------------------------------------------------------------- ALIVE (NKI lung lesions)
ALIVE = os.environ.get('ALIVE_DIR', '')   # in-house dataset, not used for the challenge


def build_alive_one(stem, out, rng):
    """ALIVE training scan: one binary lesion label -> one instance per 26-connected component (>= 3 voxels,
    largest 250 kept, ids 1..K by size), rexground-style meta with a size-based category ('2d' nodule <= 30 mm
    equivalent diameter, else '2a' mass). Texts are added afterwards by primus/alive_texts.py."""
    import nibabel as nib
    from scipy.ndimage import label as cc_label
    ct_p = os.path.join(ALIVE, 'imagesTr', f'{stem}_0000.nii.gz'); lb_p = os.path.join(ALIVE, 'labelsTr', f'{stem}.nii.gz')
    ctimg = nib.load(ct_p)
    ct = np.asanyarray(ctimg.dataobj).astype(np.float32)
    zooms = ctimg.header.get_zooms()[:3]
    m = np.asanyarray(nib.load(lb_p).dataobj)
    while m.ndim > 3:                                     # some labels are stored (X, Y, Z, 1)
        m = m[..., 0]
    if ct.ndim > 3:
        ct = ct[..., 0]
    m = m > 0
    assert m.shape == ct.shape, (m.shape, ct.shape)
    ct = nib_to_lps_zyx(ct, ctimg.affine); m = nib_to_lps_zyx(m, ctimg.affine)
    sp = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    assert ct.min() < -500 and ct.max() > 200, f'HU canary failed: {ct.min()} {ct.max()}'
    cc, n = cc_label(m, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(cc.ravel(), minlength=n + 1); sizes[0] = 0
    order = [int(i) for i in np.argsort(-sizes) if sizes[i] >= 3][:250]
    inst = np.zeros(ct.shape, np.uint16); seg = np.zeros(ct.shape, np.uint8)
    vox_mm3 = float(np.prod(sp)); findings = {}
    for k, i in enumerate(order, 1):
        mk = cc == i
        d = (6.0 * sizes[i] * vox_mm3 / np.pi) ** (1.0 / 3.0)
        cat = '2d' if d <= 30 else '2a'
        inst[mk] = k; seg[mk] = REX_CLS[cat]
        nz = np.nonzero(mk)
        findings[str(k - 1)] = dict(category=cat, text='', n_orig=int(sizes[i]), d_mm=round(float(d), 1),
                                    centroid_orig=[float(a.mean()) for a in nz])
    ct_r, (seg_r, inst_r) = resample(ct, [seg, inst], sp)
    meta = dict(dataset='alive', name=stem, split='train', source_image=ct_p, label_image=lb_p,
                orig_spacing=list(sp), orig_shape=list(ct.shape), n_findings=len(order), findings=findings)
    meta['instances'] = instances_meta(inst_r, seg_r, rng)
    write(out, stem, ct_r, seg_r, inst_r, meta)


def run_alive(args):
    stems = sorted(f[:-7] for f in os.listdir(os.path.join(ALIVE, 'labelsTr')) if f.endswith('.nii.gz'))
    if args.limit:
        stems = stems[:args.limit]
    rng = np.random.default_rng(args.task)
    for i, stem in enumerate(stems):
        if i % args.n_tasks != args.task or done(args.out, stem):
            continue
        t0 = time.time()
        try:
            build_alive_one(stem, args.out, rng)
            log(f'[{i}] {stem} ok {time.time()-t0:.1f}s')
        except Exception as ex:
            log(f'[{i}] {stem} FAIL {ex}'); traceback.print_exc()


# ----------------------------------------------------------------------------- LIDC-IDRI (public nodules)
def build_lidc_one(stem, src, out, rng):
    import nibabel as nib
    ctimg = nib.load(os.path.join(src, f'{stem}_0000.nii.gz'))
    ct = np.asanyarray(ctimg.dataobj).astype(np.float32)
    inst = np.asanyarray(nib.load(os.path.join(src, f'{stem}_inst.nii.gz')).dataobj).astype(np.uint16)
    nod = json.load(open(os.path.join(src, f'{stem}_nodules.json')))['nodules']
    zooms = ctimg.header.get_zooms()[:3]
    ct = nib_to_lps_zyx(ct, ctimg.affine); inst = nib_to_lps_zyx(inst, ctimg.affine)
    sp = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    assert ct.min() < -500 and ct.max() > 200, f'HU canary failed: {ct.min()} {ct.max()}'
    seg = ((inst > 0) * REX_CLS['2d']).astype(np.uint8)
    findings = {}
    for fid, a in nod.items():
        k = int(fid) + 1
        nz = np.nonzero(inst == k)
        if nz[0].size == 0:
            continue
        findings[fid] = dict(category='2d', text='', n_orig=int(nz[0].size), d_mm=float(a['diameter_mm']),
                             centroid_orig=[float(v.mean()) for v in nz], texture=a['texture'],
                             calcification=a['calcification'], spiculation=a['spiculation'],
                             malignancy=a['malignancy'], n_readers=a['n_readers'])
    ct_r, (seg_r, inst_r) = resample(ct, [seg, inst], sp)
    meta = dict(dataset='lidc', name=stem, split='train', source_image=os.path.join(src, f'{stem}_0000.nii.gz'),
                orig_spacing=list(sp), orig_shape=list(ct.shape), n_findings=len(findings), findings=findings)
    meta['instances'] = instances_meta(inst_r, seg_r, rng)
    write(out, stem, ct_r, seg_r, inst_r, meta)


def run_lidc(args):
    src = args.lidc_dir
    stems = sorted(f[:-13] for f in os.listdir(src) if f.endswith('_nodules.json'))
    if args.limit:
        stems = stems[:args.limit]
    rng = np.random.default_rng(args.task)
    for i, stem in enumerate(stems):
        if i % args.n_tasks != args.task or done(args.out, stem):
            continue
        t0 = time.time()
        try:
            build_lidc_one(stem, src, args.out, rng)
            log(f'[{i}] {stem} ok {time.time()-t0:.1f}s')
        except Exception as ex:
            log(f'[{i}] {stem} FAIL {ex}'); traceback.print_exc()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['pancancer', 'rexground', 'msd', 'rexground_test', 'alive', 'lidc'])
    ap.add_argument('--msd-dir', default=None)
    ap.add_argument('--msd-split', default='Tr', choices=['Tr', 'Ts'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--pairs', default=None)
    ap.add_argument('--labels', default=None)
    ap.add_argument('--task', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)))
    ap.add_argument('--n-tasks', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1)))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--lidc-dir', default=None)
    args = ap.parse_args()
    log('start', vars(args))
    {'pancancer': run_pancancer, 'rexground': run_rexground, 'msd': run_msd, 'rexground_test': run_rexground_test, 'alive': run_alive, 'lidc': run_lidc}[args.dataset](args)
    log('done')

"""Generate the challenge TEST predictions: one .nii.gz per case, 300 files, ready to zip.

Output format mirrors the ground truth exactly (verified against a val seg):
  * 4D uint8, shape (n_findings, X, Y, Z) -- one binary channel per finding, in the order the
    findings appear in the challenge json (key "0", "1", ...);
  * the CT's own affine, so the mask overlays the volume it was predicted from.

Inference matches how the model was validated at 0.3218 on val-200: sliding window over the
full volume, threshold, then anatomical restriction to the lobe(s) the finding names.
"""
import argparse, json, os, sys
from pathlib import Path
import nibabel as nib, numpy as np, torch
nib.openers.Opener.default_compresslevel = 1  # prob intermediates; ~3x faster gzip

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_t1_full import (BANK_DIR, CAT2IDX, CODES14, LOBES, TEXT_DIM, T1Model,  # noqa: E402
                           build_model, ts_path, zscore)
from eval_t1_val200 import named_region, predict_case  # noqa: E402

from paths import CHALLENGE_JSON, CTRATE_ROOT  # noqa: E402
DS = str(CHALLENGE_JSON)
CTR = str(CTRATE_ROOT)


def img_path(name):
    s = name[:-7] if name.endswith('.nii.gz') else name
    sp = 'valid_fixed' if s.startswith('valid') else 'train_fixed'
    return f"{CTR}/{sp}/{'_'.join(s.split('_')[:2])}/{'_'.join(s.split('_')[:3])}/{s}.nii.gz", s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "checkpoints" / "SUBMISSION_1ch.pt"))
    ap.add_argument("--out-dir", default=str(HERE / "submission" / "SUBMISSION_1ch"))
    ap.add_argument("--thr", type=float, default=0.4)
    ap.add_argument("--save-prob", default="",
                    help="directory to write uint16 PROBABILITY maps instead of thresholded "
                         "masks. Required for an ensemble submission: members must be averaged "
                         "as probabilities and thresholded ONCE, not thresholded separately and "
                         "voted -- the latter throws away exactly the confidence information "
                         "that makes averaging work (val: +0.0100 +- 0.0021, t=+4.79).")
    ap.add_argument("--step-frac", type=float, default=0.5,
                    help="sliding-window stride as a fraction of the patch. 0.25 measured at "
                         "+0.0129 Dice on val (0.3218 -> 0.3347) for ~3.6x the inference cost. "
                         "MUST match whatever the val number being quoted was measured at -- "
                         "predicting at 0.5 and quoting a 0.25 val score is a silent mismatch.")
    ap.add_argument("--restrict", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0, help="0 = all")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    # text: train/val bank + the test bank built by prep_test_texts.py
    t2v = {}
    for b in (BANK_DIR, HERE / "test_bank"):
        if (Path(b) / "texts.json").exists():
            tx = json.load(open(Path(b) / "texts.json")); em = np.load(Path(b) / "embeddings.npy")
            t2v.update({t: em[i] for i, t in enumerate(tx)})

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    n_in = int(ck["network_weights"]["encoder.stem.convs.0.conv.weight"].shape[1])
    net, patch = build_model(dev, in_channels=n_in, deep_supervision=True)
    model = T1Model(net, len(CODES14), TEXT_DIM).to(dev)
    # Third loader to need this. `align_to_checkpoint` exists precisely so a checkpoint trained
    # with --fusion-stage can be loaded anywhere; it was added to eval_sweep.py and
    # cache_val_probs.py and missed here, and the failure is the same every time:
    #   size mismatch for pos_embed ... [13824,1,2048] vs [1728,1,2048]
    # It is a no-op for checkpoints trained at the pretrained stage, so it is always safe to call.
    from fusion_res import align_to_checkpoint
    if align_to_checkpoint(model.net, ck["network_weights"], patch=int(patch[0])):
        model.to(dev)
    model.net.load_state_dict(ck["network_weights"]); model.cat_embed.load_state_dict(ck["cat_embed"])
    model.eval()

    d = json.load(open(DS))
    items = list(d['test'].values()) if isinstance(d['test'], dict) else d['test']
    items = items[a.start: (a.end or len(items))]
    # step_frac MUST be logged. It silently defaulted to 0.5 once already, and because no run
    # script invokes this file (it is launched inline) the log is the only record of what a
    # submission was actually built with.
    print(f"[predict] {a.ckpt} thr={a.thr} restrict={a.restrict} step_frac={a.step_frac} "
          f"-> {out}  ({len(items)} cases)",
          flush=True)

    # ---- I/O overlap (2026-08-31): the loop was strictly serial, so the GPU idled through
    # every load (gunzip+zscore ~2 s), every TotalSeg read, and every ~300 MB gzip write.
    # Measured 41-61% mean GPU util across 5 H100 jobs. A depth-1 prefetch thread loads the
    # NEXT case during the current case's GPU work, and a writer thread owns nib.save so the
    # GPU never blocks on gzip. Output arrays are unchanged (same uint16 quantisation, same
    # filenames, same skip-if-exists); only wall-clock behaviour differs.
    import threading, queue as _queue

    def _iter_todo():
        """Yield (i, e, payload) applying the SAME skip/missing logic as the old loop."""
        for i, e in enumerate(items, 1):
            ip, stem = img_path(e['name'])
            dst = out / f"{stem}.nii.gz"
            # resume against the file this run actually writes (see original comment).
            done_p = (Path(a.save_prob) / f"{stem}.nii.gz") if a.save_prob else dst
            if done_p.exists():
                continue
            if not os.path.exists(ip):
                yield (i, e, ("missing", stem, dst))
                continue
            im = nib.load(ip)
            vol = zscore(np.asanyarray(im.dataobj).astype(np.float32))
            p_ = ts_path(stem)
            ts = np.asanyarray(nib.load(str(p_)).dataobj) if p_.exists() else None
            yield (i, e, ("case", stem, dst, im.affine.copy(), vol, ts))

    _pref = _queue.Queue(maxsize=1)          # depth 1: next case loads during current GPU work
    _PDONE = object()

    def _prefetcher():
        try:
            for it in _iter_todo():
                _pref.put(it)
        except Exception as ex:              # surface loader errors in the main thread
            _pref.put(("error", ex))
        _pref.put(_PDONE)

    _wq = _queue.Queue(maxsize=2)            # bounded: at most 2 pending saves in memory
    _WDONE = object()
    _werr = []

    def _writer():
        while True:
            job = _wq.get()
            if job is _WDONE:
                return
            try:
                img_, path_, msg = job
                nib.save(img_, path_)
                if msg:
                    print(msg, flush=True)
            except Exception as ex:
                _werr.append(ex)

    threading.Thread(target=_prefetcher, daemon=True).start()
    _wt = threading.Thread(target=_writer, daemon=True)
    _wt.start()

    while True:
        got = _pref.get()
        if got is _PDONE:
            break
        if isinstance(got, tuple) and got[0] == "error":
            raise got[1]
        i, e, payload = got
        if payload[0] == "missing":
            _, stem, dst = payload
            keys = sorted(e['findings'], key=int)
            print(f"  [{i}] MISSING IMAGE {stem} -- writing empty masks", flush=True)
            shp = tuple(e['shape']) if e.get('shape') else (512, 512, 100)
            _wq.put((nib.Nifti1Image(np.zeros((len(keys),) + shp, np.uint8), np.eye(4)), dst, ""))
            continue
        _, stem, dst, affine, vol, ts = payload
        keys = sorted(e['findings'], key=int)
        texts = [e['findings'][k] for k in keys]
        cats = [CAT2IDX.get((e.get('categories') or {}).get(k, ""), 0) for k in keys]
        if n_in == 1:
            arr = vol[None]                  # zscore already returns float32; no copy
        else:
            anat = (np.stack([(ts == l).astype(np.float32) for l in LOBES]) if ts is not None
                    else np.zeros((len(LOBES),) + vol.shape, np.float32))
            arr = np.concatenate([vol[None], anat], 0).astype(np.float32)
        embs = np.stack([t2v[t] for t in texts])
        probs = predict_case(model, arr, embs, cats, patch, dev, return_prob=True,
                             step_frac=a.step_frac)

        if a.save_prob:
            pd_ = Path(a.save_prob); pd_.mkdir(parents=True, exist_ok=True)
            q = np.clip(probs * 65535.0, 0, 65535).astype(np.uint16)
            _wq.put((nib.Nifti1Image(q, np.eye(4)), pd_ / f"{stem}.nii.gz",
                     f"  [{i}] {stem}: {q.shape[0]} findings, probs saved"))
            continue

        chans = []
        for j, t in enumerate(texts):
            m = probs[j] >= a.thr
            if a.restrict:
                reg = named_region(t)
                if reg and ts is not None:
                    m = m & np.isin(ts, reg)
            chans.append(m.astype(np.uint8))
        _wq.put((nib.Nifti1Image(np.stack(chans), affine), dst,
                 f"  [{i}/{len(items)}] {stem}: {len(keys)} findings, "
                 f"vox={[int(c.sum()) for c in chans]}"))
    _wq.put(_WDONE)
    _wt.join()
    if _werr:
        raise _werr[0]
    print("[predict] done", flush=True)


if __name__ == "__main__":
    main()

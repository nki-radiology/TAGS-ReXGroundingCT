"""Qwen3-Embedding-0.6B token states for augmentation texts, in the TokenTextBank format with variant keys:
  --aug-json <rexground_text_aug.json>   -> keys '<stem>|<fid>|p<k>' (paraphrases) and '<stem>|<fid>|n<k>' (hard negatives)
  --cache-root <dir>                     -> keys '<stem>|<fid>' (base text) and '<stem>|<fid>|p<k>' (template variants)
                                            for every meta in a cache (e.g. the ALIVE cache)
    python primus/embed_texts_tokens_aug.py --aug-json cache/rexground_text_aug.json --out cache/rexground_text_tokens_aug.npz
    python primus/embed_texts_tokens_aug.py --cache-root <alive cache> --out cache/alive_text_tokens_qwen06b.npz
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

L_MAX = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aug-json', default=None)
    ap.add_argument('--cache-root', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    keys, texts = [], []
    if a.aug_json:
        d = json.load(open(a.aug_json))
        for k, v in d.items():
            for i, t in enumerate(v.get('para', []), 1):
                keys.append(f'{k}|p{i}'); texts.append(t)
            for i, t in enumerate(v.get('neg', []), 1):
                keys.append(f'{k}|n{i}'); texts.append(t)
    if a.cache_root:
        for mp in sorted(glob.glob(os.path.join(a.cache_root, '*_meta.json'))):
            m = json.load(open(mp)); stem = os.path.basename(mp)[:-10]
            for fid, f in m.get('findings', {}).items():
                if not f.get('text'):
                    continue
                keys.append(f'{stem}|{fid}'); texts.append(f['text'])
                for i, t in enumerate(f.get('variants', []), 1):
                    keys.append(f'{stem}|{fid}|p{i}'); texts.append(t)
    print(f'{len(texts)} texts ({sum(k.count("|") == 1 for k in keys)} base, '
          f'{sum("|p" in k for k in keys)} positive variants, {sum("|n" in k for k in keys)} hard negatives)', flush=True)
    from transformers import AutoTokenizer, AutoModel
    from huggingface_hub import snapshot_download
    local = snapshot_download('Qwen/Qwen3-Embedding-0.6B', local_files_only=True)
    tok = AutoTokenizer.from_pretrained(local, padding_side='right')
    m = AutoModel.from_pretrained(local, torch_dtype=torch.float32).eval()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; m = m.to(dev)
    embs, masks, lens = [], [], []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            b = tok(texts[i:i + 32], padding='max_length', truncation=True, max_length=L_MAX, return_tensors='pt').to(dev)
            h = m(**b).last_hidden_state
            embs.append(h.cpu().to(torch.float16)); masks.append(b['attention_mask'].bool().cpu())
            lens += b['attention_mask'].sum(1).tolist()
            if i % 2048 == 0:
                print(i, flush=True)
    E = torch.cat(embs).numpy(); M = torch.cat(masks).numpy()
    print('len stats: mean %.1f p95 %d max %d truncated %.1f%%' % (np.mean(lens), np.percentile(lens, 95), max(lens),
          100 * np.mean(np.array(lens) >= L_MAX)), flush=True)
    # consistency: a base key that also exists in the train/val bank must embed identically
    ref_p = os.environ.get("TAPCT_TEXT_BANK", "cache/rexground_text_tokens_qwen06b.npz")
    if os.path.exists(ref_p):
        ref = np.load(ref_p); ref_idx = {k: i for i, k in enumerate(ref['keys'].tolist())}
        shared = [i for i, k in enumerate(keys) if k in ref_idx][:20]
        if shared:
            dmax = max(float(np.abs(E[i].astype(np.float32) - ref['emb'][ref_idx[keys[i]]].astype(np.float32)).max()) for i in shared)
            print(f'{len(shared)} keys shared with the train/val bank: max |diff| {dmax:.3e}', flush=True)
    np.savez_compressed(a.out, keys=np.array(keys), emb=E, mask=M)
    print('saved', a.out, E.shape, flush=True)


if __name__ == '__main__':
    main()

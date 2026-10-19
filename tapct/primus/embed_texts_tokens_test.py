"""Token-level Qwen3-Embedding-0.6B hidden states for the ReXGroundingCT MICCAI-2026 TEST findings (texts only,
no masks), in the exact format of embed_texts_tokens.py so the two banks can be merged at export time.
Output: cache/rexground_text_tokens_qwen06b_test.npz — keys '<stem>|<fid>', emb (n, L, 1024) fp16, mask (n, L)."""
import json
import os
import sys

import numpy as np
import torch

L_MAX = 40
JSON = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TAGS_CHALLENGE_JSON", "data/ReXGroundingCT/MICCAI_challenge_dataset.json")
SPLIT = sys.argv[2] if len(sys.argv) > 2 else 'test'
OUT = f'cache/rexground_text_tokens_qwen06b_{SPLIT}.npz'

entries = json.load(open(JSON))[SPLIT]
entries = list(entries.values()) if isinstance(entries, dict) else entries
keys, texts = [], []
for e in entries:
    stem = e['name'][:-7]
    for fid, text in e['findings'].items():
        if not text:
            continue
        keys.append(f'{stem}|{fid}'); texts.append(text)
print(f'{len(texts)} {SPLIT} texts from {len(entries)} scans', flush=True)

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
E = torch.cat(embs).numpy(); M = torch.cat(masks).numpy()
print('len stats: mean %.1f p95 %d max %d truncated %.1f%%' % (np.mean(lens), np.percentile(lens, 95), max(lens),
      100 * np.mean(np.array(lens) >= L_MAX)), flush=True)
# consistency check against the train/val bank: a finding text present in both must embed identically
ref = np.load(os.environ.get("TAPCT_TEXT_BANK", "cache/rexground_text_tokens_qwen06b.npz"))
ref_idx = {k: i for i, k in enumerate(ref['keys'].tolist())}
shared = [i for i, k in enumerate(keys) if k in ref_idx]
if shared:
    d = max(float(np.abs(E[i].astype(np.float32) - ref['emb'][ref_idx[keys[i]]].astype(np.float32)).max()) for i in shared[:20])
    print(f'{len(shared)} keys also in the train/val bank; max |diff| on 20 of them = {d:.3e}', flush=True)
np.savez_compressed(OUT, keys=np.array(keys), emb=E, mask=M)
print('saved', OUT, E.shape, flush=True)

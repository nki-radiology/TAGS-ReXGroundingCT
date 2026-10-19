"""Token-level Qwen3-Embedding-0.6B hidden states for every ReXGroundingCT finding text (for SteerViT K/V).
Output: cache/rexground_text_tokens_qwen06b.npz — keys, emb (n, L, 1024) fp16 right-padded, mask (n, L) bool."""
import json, os, sys, glob
import numpy as np
import torch
import torch.nn.functional as F

L_MAX = 40
ROOT = sys.argv[1] if len(sys.argv) > 1 else 'cache/primus_cache/rexground'
OUT = os.environ.get("TAPCT_TEXT_BANK", "cache/rexground_text_tokens_qwen06b.npz")

keys, texts = [], []
for mp in sorted(glob.glob(os.path.join(ROOT, '*_meta.json'))):
    m = json.load(open(mp)); stem = os.path.basename(mp)[:-10]
    for fid, f in m.get('findings', {}).items():
        if f['n_orig'] == 0 or not f['text']:
            continue
        keys.append(f'{stem}|{fid}'); texts.append(f['text'])
print(f'{len(texts)} texts', flush=True)

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
        h = m(**b).last_hidden_state                       # (B, L, 1024)
        embs.append(h.cpu().to(torch.float16)); masks.append(b['attention_mask'].bool().cpu())
        lens += b['attention_mask'].sum(1).tolist()
        if i % 1024 == 0:
            print(i, flush=True)
E = torch.cat(embs).numpy(); M = torch.cat(masks).numpy()
print('len stats: mean %.1f p95 %d max %d truncated %.1f%%' % (np.mean(lens), np.percentile(lens, 95), max(lens),
      100 * np.mean(np.array(lens) >= L_MAX)), flush=True)
np.savez_compressed(OUT, keys=np.array(keys), emb=E, mask=M)
print('saved', OUT, E.shape, flush=True)

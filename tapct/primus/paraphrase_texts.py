"""Text-side augmentation for ReXGroundingCT training findings.

  (a) paraphrases: Qwen2.5-7B-Instruct rewrites each training finding sentence N ways while keeping every clinical
      fact (laterality, lobe/segment, size, morphology, count) — positives that teach robustness to phrasing;
  (b) hard negatives (rule-based, no model): the same sentence with its laterality or craniocaudal word swapped
      (right<->left, upper<->lower, superior<->inferior, ...), skipped when another finding of the same scan
      mentions the swapped side — negatives that force the grounder to actually read the location words.

Output: JSON {"<stem>|<fid>": {"text": ..., "para": [...], "neg": [...]}} for embed_texts_tokens_aug.py.

    python primus/paraphrase_texts.py --out cache/rexground_text_aug.json [--n-para 3] [--limit 0] [--no-llm]
"""
import argparse
import json
import os
import re
import sys
import time

REX_JSON = os.environ.get("TAGS_CHALLENGE_JSON", "data/ReXGroundingCT/MICCAI_challenge_dataset.json")
SWAPS = [('right', 'left'), ('upper', 'lower'), ('superior', 'inferior'), ('anterior', 'posterior'),
         ('apical', 'basal'), ('anterobasal', 'posterobasal'), ('medial', 'lateral')]
PROMPT = ("You rewrite chest CT report findings. Rewrite the finding below in {n} different ways a radiologist might "
          "phrase it. Keep every clinical fact exactly the same: side (left/right), lobe or segment, size in mm, "
          "morphology (e.g. solid, ground-glass, calcified, cavitary), number of lesions, and any comparison. Do not "
          "add, remove or hedge any fact. Output exactly {n} lines, one rewrite per line, no numbering, no quotes.\n\n"
          "Finding: {text}")


def swap_word(text, a, b):
    """Swap whole-word occurrences of a<->b (case preserved), return None if neither occurs."""
    pat = re.compile(r'\b(%s|%s)\b' % (a, b), re.I)
    if not pat.search(text):
        return None

    def rep(m):
        w = m.group(0); tgt = b if w.lower() == a else a
        return tgt.capitalize() if w[0].isupper() else tgt
    return pat.sub(rep, text)


def hard_negatives(text, others):
    out = []
    for a, b in SWAPS:
        t = swap_word(text, a, b)
        if t is None or t == text:
            continue
        # which side words did the swap introduce? skip if any other finding of the scan already uses them
        introduced = {w for w in (a, b) if re.search(r'\b%s\b' % w, t, re.I) and not re.search(r'\b%s\b' % w, text, re.I)}
        if any(re.search(r'\b%s\b' % w, o, re.I) for w in introduced for o in others):
            continue
        out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='cache/rexground_text_aug.json')
    ap.add_argument('--n-para', type=int, default=3)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--no-llm', action='store_true', help='hard negatives only')
    a = ap.parse_args()
    d = json.load(open(REX_JSON))
    entries = d['train']; entries = list(entries.values()) if isinstance(entries, dict) else entries
    items = []
    for e in entries:
        stem = e['name'][:-7]; texts = e['findings']
        for fid, t in texts.items():
            if t:
                items.append((f'{stem}|{fid}', t, [v for k, v in texts.items() if k != fid and v]))
    if a.limit:
        items = items[:a.limit]
    print(f'{len(items)} training findings', flush=True)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for k, t, others in items:
        res.setdefault(k, {'text': t})
        res[k]['neg'] = hard_negatives(t, others)
    n_neg = sum(len(v['neg']) for v in res.values())
    print(f'hard negatives: {n_neg} for {sum(1 for v in res.values() if v["neg"])} findings', flush=True)
    if a.no_llm:
        json.dump(res, open(a.out, 'w')); return
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to('cuda').eval()
    todo = [(k, t) for k, t, _ in items if len(res[k].get('para', [])) < a.n_para]
    print(f'paraphrasing {len(todo)} findings with {a.model} on {torch.cuda.get_device_name(0)}', flush=True)
    t0 = time.time()
    for i in range(0, len(todo), a.batch):
        chunk = todo[i:i + a.batch]
        msgs = [[{'role': 'user', 'content': PROMPT.format(n=a.n_para, text=t)}] for _, t in chunk]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt', padding=True, return_dict=True).to('cuda')
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=160, do_sample=True, temperature=0.7, top_p=0.9,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        outs = tok.batch_decode(gen[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        for (k, t), o in zip(chunk, outs):
            lines = [re.sub(r'^\s*(\d+[.)]|[-*])\s*', '', ln).strip().strip('"') for ln in o.strip().split('\n')]
            lines = [ln for ln in lines if len(ln) > 10 and ln.lower() != t.lower()]
            res[k]['para'] = lines[:a.n_para]
        if (i // a.batch) % 20 == 0:
            json.dump(res, open(a.out, 'w'))
            print(f'[{i + len(chunk)}/{len(todo)}] {(time.time() - t0) / 60:.1f} min | e.g. {chunk[0][1][:70]!r} -> {res[chunk[0][0]]["para"][:1]}', flush=True)
    json.dump(res, open(a.out, 'w'))
    n_p = sum(len(v.get('para', [])) for v in res.values())
    print(f'done: {n_p} paraphrases, {n_neg} hard negatives -> {a.out} ({(time.time() - t0) / 60:.1f} min)', flush=True)


if __name__ == '__main__':
    main()

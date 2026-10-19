"""textsens -- does the model actually USE the free-text prompt?

Dice cannot answer this. Only ~23% of test findings have a sibling sharing their category, so a
model that ignores the text entirely still earns full credit on ~77% of the score. A good Dice
number is therefore NOT evidence that the method satisfies the main track's eligibility rule
("must generate distinct masks given two findings in separate locations").

So probe the two axes of the prompt separately, on the SAME volume, with controlled triplets
that differ in exactly one thing:

    A = "<pathology_a> in the <lobe_a>"
    B = "<pathology_a> in the <lobe_b>"   -> only the LOCATION changed
    C = "<pathology_b> in the <lobe_a>"   -> only the PATHOLOGY changed

Report Dice(A,B) and Dice(A,C) between the PREDICTIONS (not against ground truth):

    both ~1.0        -> text is ignored; the model segments whatever is in the patch
    A/B low, A/C ~1  -> anatomy matcher: uses location, blind to what it is looking for
    both low         -> the prompt is genuinely driving the output

Swapped prompts are new strings, so they are embedded live with Qwen3-Embedding-4B (the bank
only holds the original findings).

    python textsens.py --ckpt VoxTellTrainer_T1_full.pt --n 12
"""
import argparse
import json
import pydoc
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_t1_full import (CAT2IDX, CODES14, LOBES, MODEL_DIR, TEXT_DIM,  # noqa: E402
                           T1Model, build_model, ts_path, zscore, apply_lora)
from eval_t1_val200 import predict_case  # noqa: E402


def build_pretrained_1ch(dev):
    """VoxTell v1.1 exactly as released: 1 input channel, no category, no anatomy.
    Mirrors diagnostics/text_influence.py so this runs the released model on the same
    12-triplet yardstick we hold our fine-tunes to."""
    from voxtell.model.voxtell_model import VoxTellModel
    plans = json.load(open(MODEL_DIR / "plans.json"))
    cfg = plans["configurations"]["3d_fullres"]
    ak = dict(cfg["architecture"]["arch_kwargs"])
    for k in cfg["architecture"]["_kw_requires_import"]:
        if ak[k] is not None:
            ak[k] = pydoc.locate(ak[k])
    net = VoxTellModel(input_channels=1, **ak, decoder_layer=4, text_embedding_dim=TEXT_DIM,
                       num_maskformer_stages=5, num_heads=32, query_dim=2048,
                       project_to_decoder_hidden_dim=2048, deep_supervision=False)
    ck = torch.load(MODEL_DIR / "fold_0" / "checkpoint_final.pth", map_location="cpu",
                    weights_only=False)
    net.load_state_dict(dict(ck["network_weights"]))
    return net.to(dev).eval(), [int(x) for x in cfg["patch_size"]]


class PretrainedShim(torch.nn.Module):
    """Exposes the T1Model call signature model(img, txt, cat, use) over the raw pretrained
    net(img, txt), ignoring category so predict_case can drive it unchanged."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, img, txt, cat=None, use=None):
        return self.net(img, txt)

TEXT_MODEL = "Qwen/Qwen3-Embedding-4B"
LOBE_PHRASE = {10: "left upper lobe", 11: "left lower lobe", 12: "right upper lobe",
               13: "right middle lobe", 14: "right lower lobe"}
# one canonical phrase per challenge category, used to build the controlled prompts
CAT_TERM = {"1a": "bronchial wall thickening", "1b": "bronchiectasis", "1c": "emphysema",
            "1d": "interlobular septal thickening", "1e": "centrilobular micronodules",
            "1f": "mosaic attenuation", "2a": "linear atelectasis",
            "2b": "consolidation", "2c": "ground-glass opacity", "2d": "pulmonary nodule",
            "2e": "pleural effusion", "2f": "honeycombing", "2g": "pneumothorax",
            "2h": "a focal lesion"}


def embed_texts(texts, device, batch_size=16, max_len=512):
    """Qwen3-Embedding-4B via VoxTell's own pooling, so vectors match the training bank.

    MUST pass a LIST to wrap_with_instruction. Given a bare string it iterates over the
    CHARACTERS and returns one wrapped item per character -- taking [0] then embeds the first
    LETTER, not the finding. That produced banks whose cosine against the training bank was
    0.33 for identical strings (should be ~1.0), and silently degraded every downstream use.
    Batched exactly like 09_voxtell_text/30_precompute_text_embeddings.py, which built the
    training bank.
    """
    from transformers import AutoModel, AutoTokenizer
    from voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    mdl = AutoModel.from_pretrained(TEXT_MODEL, torch_dtype=torch.float16).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i:i + batch_size])            # LIST, never a bare string
            tk = tok(wrap_with_instruction(chunk), padding=True, truncation=True,
                     max_length=max_len, return_tensors="pt").to(device)
            h = mdl(**tk).last_hidden_state
            out.append(last_token_pool(h, tk["attention_mask"]).float().cpu().numpy())
    del mdl
    torch.cuda.empty_cache()
    return np.concatenate(out, 0)


def dice(a, b, eps=1e-6):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float((2 * (a & b).sum() + eps) / (s + eps))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "VoxTellTrainer_T1_full.pt"))
    ap.add_argument("--n", type=int, default=12, help="val cases to probe")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--pretrained", action="store_true",
                    help="probe released VoxTell v1.1 (1ch, no category) instead of a checkpoint")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rows = [json.loads(l) for l in open(HERE / "index" / "t1_index_val.jsonl")]
    seen, probes = set(), []
    for r in rows:                                   # one finding per case, needs a category
        if r["case"] in seen or r["category"] not in CAT_TERM:
            continue
        seen.add(r["case"])
        probes.append(r)
        if len(probes) >= a.n:
            break

    rng = np.random.default_rng(0)
    triplets = []
    for r in probes:
        lobe_a = int(rng.choice(LOBES))
        lobe_b = int(rng.choice([l for l in LOBES if l != lobe_a]))
        cat_a = r["category"]
        cat_b = str(rng.choice([c for c in CAT_TERM if c != cat_a]))
        triplets.append({
            "case": r["case"], "image": r["image"],
            "A": f"{CAT_TERM[cat_a]} in the {LOBE_PHRASE[lobe_a]}",
            "B": f"{CAT_TERM[cat_a]} in the {LOBE_PHRASE[lobe_b]}",
            "C": f"{CAT_TERM[cat_b]} in the {LOBE_PHRASE[lobe_a]}",
            "cat_a": CAT2IDX[cat_a], "cat_b": CAT2IDX[cat_b],
            "lobe_a": lobe_a, "lobe_b": lobe_b,
        })

    texts = sorted({t[k] for t in triplets for k in "ABC"})
    print(f"[text] embedding {len(texts)} controlled prompts with {TEXT_MODEL}", flush=True)
    vecs = embed_texts(texts, dev)
    t2v = {t: v for t, v in zip(texts, vecs)}

    if a.pretrained:
        net, patch = build_pretrained_1ch(dev)
        model = PretrainedShim(net).to(dev).eval()
        n_in = 1
        print("[ckpt] RELEASED VoxTell v1.1 (1ch, category OFF)", flush=True)
    else:
        # infer the checkpoint's channel count -- --no-anat models are 1-channel and would
        # otherwise fail to load into a 6-channel skeleton (autodiag reported '?' for them)
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        w = ck["network_weights"].get("encoder.stem.convs.0.conv.weight")
        n_in = int(w.shape[1]) if w is not None else 1 + len(LOBES)
        net, patch = build_model(dev, in_channels=n_in, deep_supervision=True)
        is_lora = any(k.endswith(".A") or ".base." in k for k in ck["network_weights"])
        if is_lora:
            r = next(v.shape[0] for k, v in ck["network_weights"].items() if k.endswith(".A"))
            apply_lora(net, ["project_text_embed", ".linear1", ".linear2"], r=r, alpha=r)
            net.to(dev)
            print(f"[lora] rebuilt rank-{r} adapters to match checkpoint", flush=True)
        model = T1Model(net, len(CODES14), TEXT_DIM).to(dev)
        model.net.load_state_dict(ck["network_weights"])
        model.cat_embed.load_state_dict(ck["cat_embed"])
        model.eval()
        print(f"[ckpt] {a.ckpt} (epoch {ck.get('epoch')})", flush=True)

    ab, ac, in_a, in_b = [], [], [], []
    for i, t in enumerate(triplets, 1):
        vol = zscore(np.asanyarray(nib.load(t["image"]).dataobj).astype(np.float32))
        p = ts_path(t["case"])
        ts = np.asanyarray(nib.load(str(p)).dataobj) if p.exists() else np.zeros_like(vol, np.uint8)
        if n_in == 1:
            img = vol[None].astype(np.float32)
        else:
            anat = np.stack([(ts == l).astype(np.float32) for l in LOBES])
            img = np.concatenate([vol[None], anat], 0)

        embs = np.stack([t2v[t["A"]], t2v[t["B"]], t2v[t["C"]]])
        cats = [t["cat_a"], t["cat_a"], t["cat_b"]]     # A,B share pathology; C differs
        pred = predict_case(model, img, embs, cats, patch, dev, thr=a.thr)
        pa, pb, pc = pred[0], pred[1], pred[2]
        ab.append(dice(pa, pb)); ac.append(dice(pa, pc))
        # does the prediction actually sit in the lobe the prompt names?
        if pa.sum():
            in_a.append(float((pa & (ts == t["lobe_a"])).sum()) / float(pa.sum()))
        if pb.sum():
            in_b.append(float((pb & (ts == t["lobe_b"])).sum()) / float(pb.sum()))
        print(f"  [{i}/{len(triplets)}] {t['case']}: Dice(A,B)={ab[-1]:.3f} Dice(A,C)={ac[-1]:.3f}"
              f"  |A|={int(pa.sum())} |B|={int(pb.sum())} |C|={int(pc.sum())}", flush=True)

    print("\n=== textsens ===")
    print(f"  Dice(A,B) location swapped : {np.mean(ab):.3f}   (LOW = mask moved. good)")
    print(f"  Dice(A,C) pathology swapped: {np.mean(ac):.3f}   (LOW = mask changed. good)")
    print(f"  frac of A inside its named lobe: {np.mean(in_a) if in_a else float('nan'):.3f}")
    print(f"  frac of B inside its named lobe: {np.mean(in_b) if in_b else float('nan'):.3f}")
    v = "TEXT IGNORED" if (np.mean(ab) > 0.8 and np.mean(ac) > 0.8) else (
        "ANATOMY MATCHER (pathology ignored)" if np.mean(ac) > 0.8 else "text is driving output")
    print(f"  --> {v}")


if __name__ == "__main__":
    main()

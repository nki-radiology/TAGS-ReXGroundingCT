"""Generate IN-DISTRIBUTION text variants per region_prior/augment_synonyms.json.

Single-swap variants only: each variant differs from its original by exactly ONE group
substitution. Composition would need every combination embedded; one swap per sample already
gives the phrasing diversity the corpus itself attests. Sampling probability (and the caution
cap of 0.2) is applied at TRAIN time -- this script only enumerates strings for the bank.

Never touched (by construction -- only listed swaps are performed): temporal words, certainty
tiers except the listed pair, named patterns, numbers (measurement templates keep {X} exact).

    python augment_texts.py            # writes region_prior/augment_variants.json + stats
"""
import json, re, collections
from pathlib import Path

H = Path(__file__).resolve().parent
SPEC = json.load(open(H / "region_prior" / "augment_synonyms.json"))

def load_groups():
    groups = []
    for sec, entries in SPEC.items():
        if sec.startswith("_") or sec == "measurement_templates":
            continue
        for name, g in entries.items():
            if not isinstance(g, dict):
                continue
            for key in ("swap", "plural"):
                sw = g.get(key)
                if not sw:
                    continue
                variants = [(t, c or 1) for t, c in sw.items() if c is not None]
                if len(variants) >= 2:
                    groups.append(dict(name=f"{sec}.{name}.{key}", variants=variants,
                                       caution=bool(g.get("caution"))))
    return groups

GROUPS = load_groups()
MEAS = [("measuring {X} mm", 519), ("{X} mm in diameter", 157), ("{X} mm in size", 28)]
MEAS_RE = [(re.compile(r"\bmeasuring (\d+(?:\.\d+)?) mm\b"), "measuring {X} mm"),
           (re.compile(r"\b(\d+(?:\.\d+)?) mm in diameter\b"), "{X} mm in diameter"),
           (re.compile(r"\b(\d+(?:\.\d+)?) mm in size\b"), "{X} mm in size")]

def variants_of(text):
    """[(variant_text, group_name, weight, caution)] -- one substitution each."""
    out = []
    low = text.lower()
    for g in GROUPS:
        # longest phrase first, so "right lower lobe" is not shadowed by a subphrase
        for src, _ in sorted(g["variants"], key=lambda v: -len(v[0])):
            i = low.find(src.lower())
            if i < 0:
                continue
            # word boundaries: avoid swapping inside a longer word
            if (i > 0 and low[i-1].isalnum()) or \
               (i + len(src) < len(low) and low[i+len(src)].isalnum()):
                continue
            for dst, w in g["variants"]:
                if dst.lower() == src.lower():
                    continue
                d_ = dst
                if text[i].isupper() and d_ and d_[0].islower():
                    d_ = d_[0].upper() + d_[1:]        # keep sentence-initial capitalisation
                v = text[:i] + d_ + text[i+len(src):]
                out.append((v, g["name"], w, g["caution"]))
            break                                   # one occurrence per group is enough
    for rex, tmpl in MEAS_RE:
        m = rex.search(text)
        if not m:
            continue
        x = m.group(1)
        for dst_t, w in MEAS:
            if dst_t == tmpl:
                continue
            v = text[:m.start()] + dst_t.replace("{X}", x) + text[m.end():]
            out.append((v, "measurement", w, False))
        break
    return out

def main():
    rows = [json.loads(l) for l in open(H / "index" / "t1_index_train.jsonl")]
    rows += [json.loads(l) for l in open(H / "index" / "t1_index_val.jsonl")]
    texts = sorted({r["text"] for r in rows})
    var_map, all_new = {}, set()
    per_group = collections.Counter()
    for t in texts:
        vs = variants_of(t)
        if vs:
            var_map[t] = [dict(text=v, group=g, w=w, caution=c) for v, g, w, c in vs]
            for v, g, w, c in vs:
                all_new.add(v); per_group[g.split(".")[0]] += 1
    json.dump(var_map, open(H / "region_prior" / "augment_variants.json", "w"))
    json.dump(sorted(all_new), open(H / "region_prior" / "augment_new_texts.json", "w"))
    print(f"{len(texts)} unique texts; {len(var_map)} ({100*len(var_map)/len(texts):.0f}%) have >=1 variant")
    print(f"{len(all_new)} distinct variant strings to embed")
    for k, v in per_group.most_common():
        print(f"   {k:14s} {v}")
    import random
    random.seed(0)
    print("\nspot checks:")
    for t in random.sample(list(var_map), 5):
        print(f"  ORIG {t[:90]}")
        print(f"   ->  {var_map[t][0]['text'][:90]}   [{var_map[t][0]['group']}]")

if __name__ == "__main__":
    main()

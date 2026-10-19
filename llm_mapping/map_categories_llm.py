"""Map report findings to (a) the 14 ReXGroundingCT challenge categories AND
(b) the TotalSegmentator structure(s) the finding sits in, with a local LLM.

Combines two mappings in one pass, so each atomic finding gets BOTH a challenge
category (where applicable) AND its anatomical location (needed to know where the
lesion is). Extends thorax_crop_findings/map_findings_llm.py's TotalSeg rules;
adds the category field and the "whole-lungs" fix.

WHOLE-LUNGS FIX: a lung finding with NO named lobe (e.g. "pulmonary nodules",
"scattered ground-glass") must map to group:lungs (all five lobes), not only to a
pathology token like lung_nodules. Never a lung finding -> pathology token alone.

Uses ALL reports of each dataset (no thorax filter). Text-only, resumable.
vLLM + Qwen3-32B-FP8, 2 GPU. Run via run_map_categories.sh.
"""
import argparse, hashlib, json, os, re
import pandas as pd

LABEL_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "totalseg_label_map.json")
EXTRA_LABELS = ["lung_nodules", "pleural_effusion", "pericardial_effusion"]

# --- the 14 challenge categories (ReXGroundingCT). Group 1 = DIFFUSE, 2 = FOCAL. ---
CATEGORIES = {
    "1a": "bronchial wall thickening (peribronchial/bronchial wall thickening)",
    "1b": "bronchiectasis (bronchial dilatation, traction bronchiectasis)",
    "1c": "emphysema (centrilobular/paraseptal/panlobular emphysema, bullae)",
    "1d": "interlobular septal thickening (reticular pattern, Kerley lines, septal lines)",
    "1e": "micronodules (tree-in-bud, centrilobular micronodules, miliary pattern, multiple tiny <3mm nodules)",
    "1f": "other diffuse parenchymal abnormality (mosaic attenuation, diffuse fibrosis/reticulation without honeycombing, crazy-paving pattern, diffuse air-trapping)",
    "2a": "linear opacities (fibrotic bands, scarring, linear/plate/discoid atelectasis, subsegmental linear opacity)",
    "2b": "atelectasis / consolidation (air-space consolidation, lobar/segmental atelectasis with volume loss, collapse)",
    "2c": "focal ground-glass opacity (localized GGO)",
    "2d": "nodule(s) / mass (pulmonary nodule, mass, focal tumour, discrete lung lesion >~3mm)",
    "2e": "pleural effusion or pleural thickening (pleural fluid, pleural thickening, pleural plaques)",
    "2f": "honeycombing (subpleural clustered cystic honeycomb, UIP honeycombing)",
    "2g": "pneumothorax (pleural air)",
    "2h": "other focal lung/pleural finding not covered above (focal cyst, cavity, focal calcification, mucus plug)",
}
CAT_SET = set(CATEGORIES)

# --- class groups: LLM emits one token, expanded to exact members in code (from the TotalSeg mapper) ---
GROUP_PREFIX = "group:"
GROUP_RULES = {
    "bones":          lambda n: n.startswith(("rib_", "vertebrae_", "humerus_", "femur_", "hip_"))
                                or n in {"sternum", "clavicula_left", "clavicula_right",
                                         "scapula_left", "scapula_right", "costal_cartilages",
                                         "sacrum", "skull"},
    "ribs":           lambda n: n.startswith("rib_"),
    "thoracic_spine": lambda n: n.startswith("vertebrae_T"),
    "spine":          lambda n: n.startswith("vertebrae_"),
    "lungs":          lambda n: n.startswith("lung_") and "lobe" in n,
    "adrenal_glands": lambda n: n.startswith("adrenal_gland_"),
    "kidneys":        lambda n: n.startswith("kidney_") and "cyst" not in n,
}


def expand_groups(labs, vocab):
    out = []
    for l in labs:
        if l.startswith(GROUP_PREFIX):
            rule = GROUP_RULES.get(l[len(GROUP_PREFIX):])
            if rule:
                out.extend(n for n in vocab if rule(n))
        else:
            out.append(l)
    return out


def build_vocab(label_map_path):
    with open(label_map_path) as f:
        d = json.load(f)
    return [d["index_to_name"][str(i)] for i in range(1, 118)] + EXTRA_LABELS


SYSTEM_PROMPT = """You are a radiology information-extraction system for CT reports.

You are given the FINDINGS/IMPRESSION section of a CT report, a VOCABULARY of
TotalSegmentator anatomical structure names, and a list of 14 CATEGORIES of lung/
pleural abnormalities.

Split the FINDINGS into ATOMIC findings: ONE observation about ONE entity/location.
Split a sentence whenever it carries several observations or several organs:
  "Emphysema and a nodule in the right upper lobe." -> 2 findings
  "Normal adrenal glands, pancreas and spleen."     -> 3 findings (one per organ)
Never merge unrelated observations, never invent anatomy.

Each finding must be a STANDALONE statement about THIS scan. Remove every comparison
with prior imaging (increased, decreased, unchanged, stable, new, progressive,
regressed, "previously ...", "compared to ..."). Keep current measurements; drop prior.
If a statement conveys ONLY a comparison and no current finding, omit it.

OMIT non-diagnostic / administrative statements ENTIRELY (do not emit them):
communication/notification, attestations ("agree with ..."), cross-references to other
reports, section headers, list numbering, pure follow-up recommendations.

For each atomic finding output FIVE things: text, source_text, ts_labels, category, polarity.

--- (1) ts_labels: WHERE the finding sits (TotalSegmentator structures) ---
Use ONLY names from VOCABULARY, spelled exactly. A NEGATED finding is still a finding
and must be emitted (polarity "absent"); NEVER drop a clinical negative.

PARENCHYMAL LUNG findings (nodule, mass, consolidation, atelectasis, ground-glass,
emphysema, bronchiectasis, septal thickening, micronodules, honeycombing, fibrosis,
linear/band opacity ...):
  - a specific LOBE is named -> that lobe label(s), e.g. lung_lower_lobe_right.
  - a SIDE is named but no lobe -> ONLY that side's lobes:
        left  -> lung_upper_lobe_left, lung_lower_lobe_left                          (2 lobes)
        right -> lung_upper_lobe_right, lung_middle_lobe_right, lung_lower_lobe_right (3 lobes)
  - LATERALITY IS STRICT: a one-sided finding must NEVER include a lobe of the other side.
  - NO side named (e.g. "pulmonary nodules", "scattered ground-glass") OR bilateral /
    "both lungs" -> group:lungs (ALL FIVE LOBES).
  - NEVER map a parenchymal finding to only a pathology token without its lobe location.
  - PRESENT pulmonary nodule/mass -> add lung_nodules AND the lobe location above.

PLEURAL / AIR findings do NOT sit in lung parenchyma -- NEVER give them lung lobes:
  - PRESENT pleural effusion (fluid) -> ts_labels = ["pleural_effusion"] only.
  - PRESENT pericardial effusion -> ts_labels = ["pericardial_effusion"] only.
  - pleural thickening -> ts_labels = []   (the pleura is not in VOCABULARY).
  - pneumothorax -> ts_labels = []         (air in the pleural space; no VOCABULARY structure).
  - if NEGATED, drop the pathology token: "No pleural effusion" -> [], absent.

Other rules: thyroid/struma/goitre -> thyroid_gland; a named vertebra -> that vertebrae_*
label. For a whole CLASS of structures emit ONE group token instead of listing members:
  group:bones, group:ribs, group:thoracic_spine, group:spine, group:lungs,
  group:adrenal_glands, group:kidneys
ts_labels = [] when the anatomy has no VOCABULARY entry (lymph nodes, bile ducts, the
pleura, peritoneum, bone marrow, breast, ascites) or the statement names no anatomy.

--- (2) category: WHICH of the 14 challenge classes, or null ---
Assign a CODE only for a LUNG-PARENCHYMAL or PLEURAL abnormality that is PRESENT. The
category is decided by the TYPE of abnormality, NOT by how focal or diffuse it looks
(the "typical" pattern is only a hint -- a scattered ground-glass is STILL 2c):
  1a bronchial wall thickening
  1b bronchiectasis
  1c emphysema (centrilobular, paraseptal, bullous, ...)
  1d interlobular septal thickening
  1e micronodules: tiny <3mm / tree-in-bud / centrilobular / miliary nodules
  1f other DIFFUSE lung abnormality fitting none of 1a-1e/2c (e.g. mosaic attenuation)
  2a linear / band / plate / discoid opacity, linear (subsegmental) atelectasis, scar
  2b atelectasis or consolidation (lobar/segmental collapse or air-space filling)
  2c ground-glass opacity  -- ALWAYS 2c, even if scattered / multifocal / bilateral
  2d pulmonary nodule or mass (discrete, >~3 mm)
  2e pleural effusion OR pleural thickening
  2f honeycombing
  2g pneumothorax
  2h other FOCAL lung abnormality fitting none of 2a-2g
Pick the MOST SPECIFIC type; use 1f / 2h only for a real lung/pleural finding that fits no
named type. Assign null when the finding is NEGATED/absent; OR not a lung/pleural finding
(mediastinum, hilum, LYMPH NODES, heart, pericardium, vessels/aorta, trachea, oesophagus,
thyroid, chest wall, breast, bones, or ANY abdominal/head-neck/musculoskeletal finding);
OR a normal descriptor. Never force a non-lung finding into a catch-all.

--- (3) polarity ---
"absent" if the finding negates/excludes something, else "present". An absent finding has
category null (nothing to segment) but MUST still carry its ts_labels location.

Return ONLY a JSON array of objects with exactly these five keys
("text", "source_text", "ts_labels", "category", "polarity"). No prose, no markdown fences."""


def make_user_prompt(findings, vocab):
    cats = "\n".join(f"  {c}: {d}" for c, d in CATEGORIES.items())
    groups = ", ".join(GROUP_PREFIX + g for g in GROUP_RULES)
    return (f"VOCABULARY ({len(vocab)} names):\n" + ", ".join(vocab)
            + f"\n\nGROUP TOKENS (use instead of listing many members):\n{groups}"
            + f"\n\n14 CATEGORIES:\n{cats}"
            + "\n\nFINDINGS:\n" + findings.strip() + "\n\nJSON array:")


LOBES_LEFT  = {"lung_upper_lobe_left", "lung_lower_lobe_left"}
LOBES_RIGHT = {"lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"}
ALL_LOBES   = LOBES_LEFT | LOBES_RIGHT
_L   = re.compile(r"\bleft\b|left-sided", re.I)
_R   = re.compile(r"\bright\b|right-sided", re.I)
_BIL = re.compile(r"\bbilateral\b|\bboth\b|\bbibasilar\b|\bbibasal\b", re.I)


def _sided(text):
    """left/right only if the finding names exactly one side and is not bilateral."""
    if _BIL.search(text):
        return None
    l, r = bool(_L.search(text)), bool(_R.search(text))
    return "left" if (l and not r) else "right" if (r and not l) else None


def parse_response(raw, vocab_set, vocab):
    s = raw.strip()
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
    i, j = s.find("["), s.rfind("]")
    if i == -1 or j == -1 or j <= i:
        return None, 0
    try:
        arr = json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None, 0
    if not isinstance(arr, list):
        return None, 0
    out, dropped = [], 0
    for it in arr:
        if not isinstance(it, dict) or "text" not in it:
            continue
        labs_in = it.get("ts_labels") or []
        if isinstance(labs_in, str):
            labs_in = [labs_in]
        labs_in = expand_groups(labs_in, vocab)
        labs = [l for l in labs_in if l in vocab_set]
        dropped += len(labs_in) - len(labs)
        pol = str(it.get("polarity", "present")).lower()
        if pol not in ("present", "absent"):
            pol = "present"
        cat = it.get("category")
        cat = cat if (isinstance(cat, str) and cat in CAT_SET) else None
        if pol == "absent":
            cat = None                       # nothing present to segment
        # hard anatomical guards (enforce the two invariants the LLM occasionally breaks):
        side = _sided(str(it["text"]))       # one-sided finding -> drop other-side lobes
        if side == "left":
            labs = [l for l in labs if l not in LOBES_RIGHT]
        elif side == "right":
            labs = [l for l in labs if l not in LOBES_LEFT]
        if cat in ("2e", "2g"):              # pleural effusion/thickening & pneumothorax: not parenchyma
            labs = [l for l in labs if l not in ALL_LOBES]
        out.append({"text": str(it["text"]).strip(),
                    "source_text": str(it.get("source_text", it["text"])).strip(),
                    "ts_labels": sorted(set(labs)), "category": cat, "polarity": pol})
    return out, dropped


def load_records(dataset, limit=0):
    """Normalize each dataset's reports to [{key, text, scan_ids}]. Uses ALL reports."""
    from collections import defaultdict
    recs = []
    if dataset == "ctrate":
        base = os.path.join(os.environ.get("TAGS_CTRATE_ROOT", "data/CT-RATE/dataset"), "radiology_text_reports")
        studies = defaultdict(lambda: {"text": None, "scans": []})
        for split in ("train_reports.csv", "validation_reports.csv"):
            df = pd.read_csv(f"{base}/{split}", usecols=["VolumeName", "Findings_EN"])
            for r in df.itertuples(index=False):
                if pd.isna(r.Findings_EN) or not str(r.Findings_EN).strip():
                    continue
                vn = str(r.VolumeName); stem = vn[:-7] if vn.endswith(".nii.gz") else vn
                study = "_".join(stem.split("_")[:3]); s = studies[study]
                s["scans"].append(vn)
                if s["text"] is None or len(str(r.Findings_EN)) > len(s["text"]):
                    s["text"] = str(r.Findings_EN)
        for study, v in studies.items():
            recs.append({"key": f"ctrate_{study}", "text": v["text"], "scan_ids": sorted(set(v["scans"]))})
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return recs[:limit] if limit else recs


OUT_DIR = os.environ.get("TAGS_LLM_OUT", "llm_mapping/outputs")
DEFAULT_OUTPUT = {d: f"{OUT_DIR}/{d}_categories.jsonl" for d in ("ctrate",)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["ctrate"], default="ctrate")
    p.add_argument("--input-jsonl", default="", help="pre-built {key,text} rows; bypasses load_records")
    p.add_argument("--output", default=None)
    p.add_argument("--label-map", default=LABEL_MAP)
    p.add_argument("--model", default="Qwen/Qwen3-32B-FP8")
    p.add_argument("--max-model-len", type=int, default=12288)
    p.add_argument("--max-tokens", type=int, default=6144)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)          # this shard's index
    p.add_argument("--num-shards", type=int, default=1)     # total shards for this dataset
    p.add_argument("--enforce-eager", action="store_true")  # a100: skip CUDA-graph capture (bad compiled kernel on sm80)
    a = p.parse_args()
    a.output = a.output or DEFAULT_OUTPUT[a.dataset]
    if a.num_shards > 1:                                    # per-shard output file so shards never clobber
        base, ext = os.path.splitext(a.output)
        a.output = f"{base}.shard{a.shard}of{a.num_shards}{ext}"
    os.makedirs(os.path.dirname(a.output), exist_ok=True)

    vocab = build_vocab(a.label_map); vocab_set = set(vocab)
    if a.input_jsonl:                                  # e.g. ReXGroundingCT finding strings
        records = [json.loads(l) for l in open(a.input_jsonl)]
        if a.limit:
            records = records[:a.limit]
    else:
        records = load_records(a.dataset, limit=a.limit)
    if a.num_shards > 1:                                    # deterministic stride split (same record order every run)
        records = records[a.shard::a.num_shards]
    print(f"[data] {a.dataset}: {len(records)} reports to map "
          f"(shard {a.shard}/{a.num_shards}) (-> {a.output})", flush=True)

    done = set()
    if os.path.exists(a.output):
        with open(a.output) as f:
            for line in f:
                try: done.add(json.loads(line)["key"])
                except Exception: pass
        print(f"[resume] {len(done)} reports already mapped", flush=True)
    todo = [r for r in records if r["key"] not in done]
    print(f"[data] {len(todo)} reports remaining", flush=True)
    if not todo:
        return

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import torch
    n_gpus = torch.cuda.device_count() or 1
    print(f"[model] loading {a.model} on {n_gpus} GPU(s)", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, tensor_parallel_size=n_gpus, dtype="bfloat16",
              max_model_len=a.max_model_len, gpu_memory_utilization=0.92,
              enforce_eager=a.enforce_eager)
    sp = SamplingParams(temperature=0, max_tokens=a.max_tokens)

    def render(msgs):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    n_ok = n_bad = n_drop = n_trunc = n_cat = 0
    with open(a.output, "a") as out, open(a.output + ".unparsable.jsonl", "a") as bad:
        for s in range(0, len(todo), a.batch_size):
            chunk = todo[s:s + a.batch_size]
            prompts = [render([{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": make_user_prompt(r["text"], vocab)}]) for r in chunk]
            outs = llm.generate(prompts, sp)
            for r, o in zip(chunk, outs):
                findings, dropped = parse_response(o.outputs[0].text, vocab_set, vocab)
                n_drop += dropped
                fr = o.outputs[0].finish_reason
                if fr == "length":
                    n_trunc += 1
                if findings is None:
                    n_bad += 1
                    bad.write(json.dumps({"key": r["key"], "finish_reason": fr}) + "\n"); bad.flush()
                    continue
                n_ok += 1
                n_cat += sum(1 for f in findings if f["category"])
                out.write(json.dumps({"key": r["key"], "dataset": a.dataset, "scan_ids": r["scan_ids"],
                                      "n_findings": len(findings), "findings": findings}) + "\n")
            out.flush()
            print(f"[gen] {min(s+a.batch_size,len(todo))}/{len(todo)}  ok={n_ok} unparsable={n_bad} "
                  f"categorized={n_cat} oov_dropped={n_drop} truncated={n_trunc}", flush=True)
    print(f"[done] ok={n_ok} unparsable={n_bad} categorized={n_cat} oov_dropped={n_drop} "
          f"truncated={n_trunc} -> {a.output}", flush=True)


if __name__ == "__main__":
    main()

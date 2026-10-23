"""Download the submitted weights from Hugging Face and write them where the scripts expect them.

    python tags/hf_weights.py                 # everything (about 20 GB)
    python tags/hf_weights.py plain fusion24  # only these members (plus the post-processing models)

Writes tags/checkpoints/<member>.pt (12 VoxTell members), tags/checkpoints/comp_head4.pt, tags/results/thr_model_lobe16.pkl
and tapct/results/primus/rex_fttxt_all_z24_e16_25k_s{0..4}/last.pt. The .safetensors files on Hugging Face store each
tensor once; weights that VoxTell shares between its encoder and decoder are restored from the alias list in the metadata.
"""
import ast, json, shutil, sys
from collections import OrderedDict
from pathlib import Path

import numpy as np, torch
from safetensors.torch import load_file

REPO_ID = "fomofo/TAGS-ReXGroundingCT"
MEMBERS = ["plain", "plain_2", "tversky_fp", "tversky_fp_2", "tversky_fn", "aux_effusion", "aux_nodule_effusion",
           "fusion24", "lobe_aux", "nodule_boost15", "nodule_boost20", "category_balance"]
ROOT = Path(__file__).resolve().parent.parent


def _meta(path):
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        return {k: ast.literal_eval(v) for k, v in f.metadata().items()}


def _state(tensors, prefix, order, aliases):
    """OrderedDict in the original key order; aliased keys point at the same tensor object."""
    sd = OrderedDict()
    for k in order:
        src = aliases.get(k, k)
        sd[k] = tensors[prefix + src]
    return sd


def voxtell_checkpoint(path):
    t, m = load_file(path), _meta(path)
    ck = {"network_weights": _state(t, "network_weights.", m["order"]["network_weights"], m["aliases"]),
          "cat_embed": _state(t, "cat_embed.", m["order"]["cat_embed"], {}), "epoch": m["epoch"]}
    ck.update(m["extras"])
    return ck


def tapct_checkpoint(path):
    t, m = load_file(path), _meta(path)
    return {"it": m["it"], "args": m["args"], "specs": m["specs"], "adapters": {}, "opts": {}, "enc_opt": {},
            "heads": _state(t, "heads.", m["order"]["heads"], m["aliases"]),
            "vit": _state(t, "vit.", m["order"]["vit"], m["aliases"])}


def comp_head_checkpoint(path):
    t, m = load_file(path), _meta(path)
    return {"state": _state(t, "state.", m["order"]["state"], {}), "nf": m["nf"], "cin": m["cin"],
            "fmean": t["fmean"].numpy(), "fstd": t["fstd"].numpy()}


def main(members):
    from huggingface_hub import snapshot_download
    src = Path(snapshot_download(REPO_ID, allow_patterns=[f"voxtell/{m}.safetensors" for m in members]
                                 + ["tapct/*", "postprocessing/*"]))
    ck = ROOT / "tags" / "checkpoints"; ck.mkdir(parents=True, exist_ok=True)
    for m in members:
        torch.save(voxtell_checkpoint(src / "voxtell" / f"{m}.safetensors"), ck / f"{m}.pt"); print("wrote", ck / f"{m}.pt")
    torch.save(comp_head_checkpoint(src / "postprocessing" / "comp_head4.safetensors"), ck / "comp_head4.pt")
    (ROOT / "tags" / "results").mkdir(exist_ok=True)
    shutil.copy(src / "postprocessing" / "thr_model_lobe16.pkl", ROOT / "tags" / "results" / "thr_model_lobe16.pkl")
    for k in range(5):
        d = ROOT / "tapct" / "results" / "primus" / f"rex_fttxt_all_z24_e16_25k_s{k}"; d.mkdir(parents=True, exist_ok=True)
        torch.save(tapct_checkpoint(src / "tapct" / f"seed{k}.safetensors"), d / "last.pt"); print("wrote", d / "last.pt")


if __name__ == "__main__":
    main(sys.argv[1:] or MEMBERS)

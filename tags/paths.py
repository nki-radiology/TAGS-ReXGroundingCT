"""Where data, pretrained weights and TAP-CT probability maps live.

Every location can be overridden with an environment variable. The defaults assume this layout
inside the repository (see README):

    data/ReXGroundingCT/     MICCAI_challenge_dataset.json, segmentations/
    data/CT-RATE/dataset/    train_fixed/, valid_fixed/, ts_seg/, radiology_text_reports/
    weights/voxtell_v1.1/    VoxTell v1.1 (plans.json, fold_0/checkpoint_final.pth)
    weights/voxtell_text_bank/   built by tags/build_text_bank.py
    tapct_maps/val200_prob/, tapct_maps/test300_prob/   TAP-CT probability maps (tapct/)
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _path(var, default):
    return Path(os.environ.get(var, str(default)))


REXG_ROOT = _path("TAGS_REXG_ROOT", REPO / "data" / "ReXGroundingCT")
CHALLENGE_JSON = _path("TAGS_CHALLENGE_JSON", REXG_ROOT / "MICCAI_challenge_dataset.json")
CTRATE_ROOT = _path("TAGS_CTRATE_ROOT", REPO / "data" / "CT-RATE" / "dataset")
WEIGHTS = _path("TAGS_WEIGHTS", REPO / "weights")
VOXTELL_DIR = _path("TAGS_VOXTELL_DIR", WEIGHTS / "voxtell_v1.1")
TEXT_BANK_DIR = _path("TAGS_TEXT_BANK", WEIGHTS / "voxtell_text_bank")
TAPCT_VAL_MAPS = _path("TAGS_TAPCT_VAL_MAPS", REPO / "tapct_maps" / "val200_prob")
TAPCT_TEST_MAPS = _path("TAGS_TAPCT_TEST_MAPS", REPO / "tapct_maps" / "test300_prob")
LLM_OUT = _path("TAGS_LLM_OUT", REPO / "llm_mapping" / "outputs")

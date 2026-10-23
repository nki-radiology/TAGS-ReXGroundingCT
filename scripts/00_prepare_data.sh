#!/bin/bash
# Data preparation. Run from the repository root; locations are set in tags/paths.py (env-overridable).
# GPU needed for the text embeddings and the LLM steps (Qwen3-32B-FP8 via vLLM: one 80 GB GPU).
set -euo pipefail
cd tags
python build_t1_index.py            # index/t1_index_{train,val}.jsonl  (one row per finding)
python build_indices.py             # index/t1_index_all.jsonl (train + validation)
python build_text_bank.py           # weights/voxtell_text_bank  (Qwen3-Embedding-4B, train+val sentences)
python prep_test_texts.py           # test_bank/                 (same encoder, test sentences)
python augment_texts.py             # region_prior/augment_variants.json  (text augmentation, from augment_synonyms.json)
python embed_augment.py             # augment_bank/
python build_lobe_bank.py           # lobe_bank/                 (lobe-aux member only)

# Lobe mapping used by the restriction: Qwen3-32B maps every finding sentence to TotalSegmentator lobes.
# The submission's mapping is included in tags/rexg_ts/; to regenerate it instead:
# python build_rexg_findings.py       # rexg_ts/rexg_findings.jsonl
# python ../llm_mapping/map_categories_llm.py --input-jsonl rexg_ts/rexg_findings.jsonl \
#     --output rexg_ts/rexg_ts_labels.jsonl --batch-size 256 --enforce-eager

# CT-RATE auxiliary pool (members aux_effusion and aux_nodule_effusion): split the CT-RATE reports into findings with
# Qwen3-32B, pair nodule/effusion findings with TotalSegmentator masks, keep 8,000 evenly spaced rows, then replace
# rows of challenge test/validation patients by other CT-RATE findings (the pool used for the submission had not
# excluded test patients; see README).
python ../llm_mapping/map_categories_llm.py --dataset ctrate --batch-size 256 --enforce-eager
python build_aux_index.py --out index/aux_ctrate_full.jsonl --limit 0
python prep_aux_texts.py --index index/aux_ctrate_full.jsonl --cap 8000     # aux_bank/
python build_clean_aux.py                                                   # aux_bank_clean/ (used for training)

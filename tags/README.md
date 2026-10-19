# tags/: file guide

Research code; many files and options exist for experiments that did not make it into the final models.
This guide lists what the final submission actually uses. The history is in the top-level README.

## Files

| file | role in the final submission |
|---|---|
| `paths.py` | data, weight and TAP-CT map locations (environment variables) |
| **training** | |
| `train_t1_full.py` | trains every VoxTell member; also defines the model (`T1Model`: VoxTell + category embedding) and the data sampler |
| `train_t1_base.py` | loss functions (Dice + BCE, Tversky, deep supervision) and intensity normalisation |
| `aux_dataset.py` | CT-RATE pseudo-label sampling (members `aux2e`, `auxdil2`, first fine-tuning stage only) |
| `fusion_res.py` | finer 24³ text-image fusion grid (member `fusion24`) |
| `copy_paste.py`, `anat_pos.py`, `region_prior/lung_regions.py` | imported by `train_t1_full.py`, but only for options no final member uses (see below) |
| **inference** | |
| `predict_test.py` | test probability maps (stride 1/8) |
| `cache_val_probs.py` | validation probability maps of the train-only twins (stride 1/4) |
| `eval_t1_val200.py` | sliding-window prediction used by both, the lobe lookup (`named_region`) and validation scoring |
| `build_val_cache.py` | imported by `eval_t1_val200.py` (validation scoring only) |
| **data preparation** | |
| `build_t1_index.py`, `build_indices.py` | one row per finding: train, val, train+val, train + val×5 |
| `build_text_bank.py`, `prep_test_texts.py`, `textsens.py` | Qwen3-Embedding-4B sentence embeddings (`textsens.embed_texts` is the shared encoder; the file name comes from a text-sensitivity probe) |
| `augment_texts.py`, `embed_augment.py`, `region_prior/augment_synonyms.json` | text augmentation: one phrase swapped for an equivalent phrasing seen in train+val |
| `build_lobe_bank.py` | lobe prompts for the lobe-aux member |
| `build_rexg_findings.py` | sentence list for the LLM lobe mapping (`../llm_mapping`) |
| `rexg_ts/` | the LLM lobe mapping used for the submission (CC BY-NC-SA 4.0, see its README) |
| `build_aux_index.py`, `prep_aux_texts.py` | CT-RATE pseudo-label pool, as submitted (includes test patients, see top-level README) |
| `build_clean_aux.py` | the same pool without challenge test/val patients (recommended for new work) |
| `stage_npy.py` | optional: pre-normalised `.norm.npy` copies of the CTs for faster training (identical values) |
| **post-processing** | |
| `build_ensemble_submission.py`, `restrict_rule.py` | member mean, TAP-CT fusion, lobe/lung restriction, per-finding threshold |
| `thr_predictor2.py` | threshold model: features (also used at inference) and fitting |
| `harvest_components_crops.py`, `comp_head.py`, `apply_keep_head.py` | component head: training crops, training, application |
| `apply_case.py` | voxel competition between findings of one scan (`AC_MODE=vox:0.2`) |
| `make_submission_fast.py` | checks the masks against the challenge spec and zips them |
| `subsets_afternoon.txt` | line 1: the 9 train-only twins used to fit the post-processing |

Many post-processing scripts accept further `*_MODE` / env options from experiments; the scripts in
`../scripts/` set exactly the ones used.

## `train_t1_full.py` options

**Used by the final members:**
* basic training: `--epochs --iters --batch --micro --n-crops --workers --optim --lr --index --out --resume --save-every`
* `--no-anat`: required. Without it the model expects 5 extra lobe input channels (an early variant).
* `--enc-from`: stage-0 encoder, four members only.
* sampling: `--fg-prob --pair-prob --cat-boost --cat-balance`
* `--tversky`: `alpha,beta`, where alpha weights false negatives and beta false positives.
* `--text-aug`
* `--aux-index --aux-bank --aux-prob --aux-dilate`: CT-RATE pseudo-labels.
* `--fusion-stage --fusion-lr-mult`: fusion24 member.
* `--lobe-prob --fg-jitter --cat-dropout`: lobe-aux member, first stages only.

**Not used by any final member** (experiments, all off by default):

| option | idea |
|---|---|
| `--text-film`, `--film-lr-mult` | text conditioning (FiLM) of the encoder skip connections |
| `--attn-gate` | spatial gates on the skips driven by the text cross-attention |
| `--anat-pos` | anatomy-relative positional encoding (distance and direction from the hilum, distance to pleura) |
| `--region-aux` | auxiliary prediction of the soft anatomical region named in the sentence |
| `--tn-weight` | penalty on probability in lobes the finding does not occupy |
| `--copy-paste`, `--aug-exclusive` | lesion copy-paste augmentation |
| `--lat-swap` | left/right word swap with an empty target |
| `--excl-weight`, `--attn-excl` | penalties for two findings predicting/attending to the same place |
| `--detect-aux`, `--detect-neg` | prompt-independent "any finding here" auxiliary loss |
| `--boundary-weight` | extra loss weight on the mask boundary |
| `--cap-forgive` | softer loss for findings hitting the annotators' 3-instance cap |
| `--require-same-cat`, `--prefer-same-cat` | paired crops with two findings of the same category |
| `--lora-rank`, `--freeze-enc`, `--bank` | LoRA on the text pathway, frozen encoder, alternative text bank |

# TAGS: text-prompted CT finding segmentation (ReXGroundingCT, MICCAI 2026, Track 1)

Code of team **fomo.fo**, ranked 3rd in Track 1 of the ReXGroundingCT challenge. The final submission, **TAGS_TAPv3**,
scored Dice 0.362 on the full test set.

Given a chest CT and a free-text finding ("small nodule in the right upper lobe"), the method segments that finding.
It has three parts:

1. **Twelve VoxTell models.** VoxTell is a free-text promptable 3D segmentation model (nnU-Net residual encoder;
   a frozen Qwen3-Embedding-4B sentence embedding queries the bottleneck through a transformer decoder). Each of our
   twelve copies was fine-tuned with a different initialisation, loss, sampling or auxiliary data.
2. **TAP-CT.** A 3D ViT-B foundation model encoder, fine-tuned with FiLM text conditioning (Qwen3-Embedding-0.6B).
   Its probability map is averaged over five seeds and fused with the VoxTell mean at weight 0.2.
3. **Post-processing**, in this order:
   * restriction to the named lobe (dilated 16 voxels) or lung (dilated 12 voxels), using an LLM (Qwen3-32B) mapping
     of each sentence to lobes;
   * a per-finding threshold predicted by a gradient-boosted regressor;
   * a small 3D CNN that drops connected components predicted to lower Dice;
   * resolution of voxels claimed by several findings.

## Read this first: why the code looks the way it does

This is research code that grew over two months of experiments. The final ensemble combines models made
at different points in that process, with different recipes, and none of the older models were retrained
when the recipe improved. That explains most of the unevenness below.

**How the twelve VoxTell members came about**

1. **July: the first model.** VoxTell v1.1, fine-tuned on the challenge data. Its encoder was first
   replaced by that of a text-free 14-category nnU-Net (`stage0/`). Training used 85% finding-centred crops,
   half of them showing two findings at once.
2. **Late July to early August: variants.** Four fine-tunes of that model: a cautious Tversky loss (`cautious`),
   CT-RATE pseudo-labels (`aux2e`, `auxdil2`) and a finer text-image fusion grid (`fusion24`). Separately, a
   model with lobe masks as an extra target (`lobeaux`), trained from VoxTell v1.1 without the stage-0 encoder.
3. **6 August: the stage-0 encoder turned out not to matter.** A model trained without it scored the same on
   validation (restricted Dice 0.331 vs 0.335 with it, within noise). All later new models start directly
   from VoxTell v1.1.
   * The four existing variants were nonetheless rebuilt on train + validation with their original recipe
     (`*_all`, via `SUBMISSION_all`), so they still use the stage-0 encoder. They then got 60 more epochs with
     validation oversampled 5× (`*_v5`).
   * `stage0/` is needed only to reproduce these four.
4. **End of August: a better recipe.** One third finding-centred crops, no paired crops, and text
   augmentation. The five existing models got this only as a final 60-epoch stage (`*_aug33`). Seven new
   models were trained with it from VoxTell v1.1: `fleet_*` (plain, two Tversky settings, a repeat run)
   and `mb_*` (oversampling of rare categories / nodules).
5. **September: TAP-CT** was added as a second architecture, and the post-processing was built:
   lobe/lung restriction, per-finding threshold model, component head, voxel competition.

Several ideas were tried and not used in the final models. Their options are still in the code, off by
default; see `tags/README.md`. The pseudo-label and lobe-aux members gave no clear gain on their own but
were kept for ensemble diversity.

**Names you will see**

| name | meaning |
|---|---|
| `_all` | trained on challenge train + validation |
| `_v5` | stage with each validation case repeated 5× |
| `_aug33` / `aug33` | final stage with text augmentation and 1/3 finding-centred crops |
| `fleet_*`, `mb_*` | the seven later members trained from VoxTell v1.1 |
| twin | train-only copy of a member, used only to fit the post-processing on validation |
| `sf125` | test inference with sliding-window stride 1/8 |
| `LOBE16`, `THRL`, `HEAD4`, `CM02` | lobe restriction (16 voxels), threshold model, component head, voxel competition (margin 0.2) |
| `TAGS_TAPv3` | the final submission (internally `XGEN12_TAP_w020_LOBE16_THRL_HEAD4_CM02`) |

## Layout

| path | contents |
|---|---|
| `tags/` | VoxTell members (training `train_t1_full.py`, test inference `predict_test.py`), data preparation, post-processing; file guide in `tags/README.md` |
| `tags/paths.py` | all data and weight locations (environment variables, defaults below) |
| `stage0/` | the 14-category nnU-Net whose encoder initialises four members (legacy, see above) |
| `tapct/` | the TAP-CT member |
| `llm_mapping/` | Qwen3-32B mapping of sentences to lobes and of CT-RATE reports to findings |
| `scripts/` | every stage in order, with the exact arguments used for the submission |

## Setup

```bash
conda create -n tags python=3.11 && conda activate tags
pip install -r requirements.txt
```

Data and weights. These are the default locations; each can be overridden with the variable in brackets.
* `data/ReXGroundingCT/` (`TAGS_REXG_ROOT`): `MICCAI_challenge_dataset.json` and `segmentations/`, from
  [ReXGroundingCT](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT).
* `data/CT-RATE/dataset/` (`TAGS_CTRATE_ROOT`): `train_fixed/`, `valid_fixed/`, `ts_seg/` (TotalSegmentator masks)
  and `radiology_text_reports/`, from [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE).
  The CT volumes come from CT-RATE `*_fixed` (HU and real affine).
* `weights/voxtell_v1.1/` (`TAGS_VOXTELL_DIR`): VoxTell v1.1 (`plans.json`, `fold_0/checkpoint_final.pth`).
* `tapct_maps/{val200_prob,test300_prob}/` (`TAGS_TAPCT_VAL_MAPS`, `TAGS_TAPCT_TEST_MAPS`): TAP-CT maps from `scripts/50_tapct.sh`.

## Running

Run each script from the repository root.

| script | what | hardware |
|---|---|---|
| `scripts/00_prepare_data.sh` | indices, text embeddings, text-augmentation bank, LLM lobe mapping, CT-RATE auxiliary pool | 1 GPU (80 GB for Qwen3-32B) |
| `scripts/10_stage0_category_encoder.sh` | 14-category nnU-Net (challenge train only) | 1 GPU |
| `scripts/20_train_voxtell_members.sh` | the 12 VoxTell members (challenge train + val) | 1 GPU each; stage 1 uses 3 |
| `scripts/30_train_val_twins.sh` | train-only twins and their validation maps, for fitting the post-processing | 1 GPU each |
| `scripts/40_predict_test.sh` | test probability maps, stride 1/8 (about 3.7 min per scan per member on an H100) | 1 GPU |
| `scripts/50_tapct.sh` | TAP-CT: caches, text banks, head initialisation, 5 seeds, test maps | 1 GPU |
| `scripts/60_fit_postprocessing.sh` | threshold model and component head, on validation | CPU |
| `scripts/70_build_submission.sh` | ensemble + post-processing → submission zip | CPU |

## Notes

* **Reproducibility.**
  * The post-processing (scripts 60–70 applied to the same probability maps) reproduces our submission voxel for voxel.
  * Training was not seeded, so retrained models will differ slightly from ours.
* **TAP-CT.** The encoder is downloaded from Hugging Face (`fomofo/tap-ct-b-3d`, revision pinned in
  `tapct/lattice/models.py`). Set `PRIMUS_WIN_Z=24 PRIMUS_Z_MM=1.5` as in `scripts/50_tapct.sh`.
* **Test-set exposure.**
  * The CT-RATE auxiliary pool used by two members excluded validation patients but not test patients:
    80 of the 300 test studies were in it, as image, CT-RATE report sentence and TotalSegmentator pseudo-mask.
  * Retraining both members on a pool without them (`tags/build_clean_aux.py`) changed the final masks by
    mean Dice 0.025, equally on leaked and other studies.
  * Use `build_clean_aux.py` for new work.
* **Included data.** `tags/rexg_ts/` holds the LLM lobe mapping of all challenge sentences used for the
  submission. It contains ReXGroundingCT text and is licensed CC BY-NC-SA 4.0, not Apache-2.0.

## Licence

* **Code:** Apache License 2.0 (`LICENSE`). See `NOTICE` for third-party code.
* **`tags/rexg_ts/`:** CC BY-NC-SA 4.0. Derived from ReXGroundingCT; see `tags/rexg_ts/README.md`.
* **Models and data used:** VoxTell v1.1 weights (CC BY-NC-SA 4.0), TAP-CT weights (CC BY-NC 4.0),
  ReXGroundingCT and CT-RATE (CC BY-NC-SA 4.0) have their own licences.
  Models trained with this code on these weights and data are subject to them.

## Citation

Paper in preparation.

# TAGS: free-text finding segmentation in chest CT

Code of team **fomofo** for Track 1 of the [ReXGroundingCT challenge](https://rexrank.ai/ReXGroundingCT/challenge.html)
(MICCAI 2026): given a chest CT and a sentence from the radiology report, segment the finding the sentence describes.
Our final submission **TAGS_TAPv3** ranked 3rd (Dice 0.362 on the full test set).

## Method

```
CT + finding sentence
   ├─ 12 VoxTell models ─► mean map ─┐
   └─ TAP-CT, 5 seeds  ─► mean map ─┴─► fused map = 0.8·VoxTell + 0.2·TAP-CT
                                            │
                                            ├─ 1. restrict to the named lobe or lung
                                            ├─ 2. per-finding threshold
                                            ├─ 3. drop harmful components (3D CNN)
                                            └─ 4. resolve voxels claimed by several findings
                                                   ▼
                                               final mask
```

**VoxTell** ([Rokuss et al. 2025](https://arxiv.org/abs/2511.11450)) is a text-promptable 3D segmentation model: an nnU-Net
residual encoder, with the Qwen3-Embedding-4B embedding of the sentence querying the bottleneck through a transformer decoder.
We trained twelve copies on the challenge data (train + validation), all with the same recipe
(250 epochs from VoxTell v1.1, one third of the crops centred on a finding, text augmentation), each with one difference:

| member | what is different |
|---|---|
| `plain`, `plain_2` | nothing (two runs) |
| `tversky_fp`, `tversky_fp_2` | Tversky loss 0.3/0.7, penalising false positives (two runs) |
| `tversky_fn` | Tversky loss 0.6/0.4, penalising false negatives |
| `aux_effusion` | + 1,540 CT-RATE effusion pseudo-labels (TotalSegmentator masks, report sentences split by Qwen3-32B) |
| `aux_nodule_effusion` | + 8,000 CT-RATE nodule and effusion pseudo-labels |
| `fusion24` | finer 24³ text–image fusion grid |
| `lobe_aux` | lung-lobe masks as an auxiliary segmentation target |
| `nodule_boost15`, `nodule_boost20` | scans with a nodule finding sampled 1.5× / 2× more often |
| `category_balance` | scans with rare finding categories sampled more often |

The submitted models were trained with the same twelve differences but a longer history: five of them went through
earlier recipes and training stages before the final one, and four of those started from the encoder of a text-free
14-category nnU-Net, which a later ablation showed to make no difference. The scripts here train every member in one
go with the final recipe; see [HISTORY.md](HISTORY.md) for what was actually run and how the names map.

**TAP-CT** ([Veenboer et al. 2025](https://arxiv.org/abs/2512.00872)) is a 3D ViT-B CT foundation model
([`fomofo/tap-ct-b-3d`](https://huggingface.co/fomofo/tap-ct-b-3d)). We added FiLM text conditioning on
Qwen3-Embedding-0.6B and a convolutional decoder, trained the heads for 50k steps with the encoder frozen, then
fine-tuned everything for 25k steps, with five seeds (`tapct/`).

**Post-processing** (`tags/`, fitted on validation predictions of train-only models):
1. restriction to the lobe named in the sentence (dilated 16 voxels), or else to the named lung (dilated 12 voxels);
   the lobe comes from a Qwen3-32B mapping of every sentence, the masks from TotalSegmentator;
2. a per-finding threshold from a gradient-boosted regressor on 41 map and text features;
3. a small 3D CNN that removes connected components predicted to lower Dice;
4. voxels claimed by several findings of one scan go to the most confident one.

## Repository

| path | contents |
|---|---|
| `tags/` | VoxTell members: training, test inference, data preparation, post-processing (file guide: `tags/README.md`) |
| `tags/paths.py` | all data and weight locations (environment variables) |
| `tapct/` | the TAP-CT member |
| `llm_mapping/` | Qwen3-32B mapping of sentences to lobes, and of CT-RATE reports to findings |
| `scripts/` | every stage in order, with the exact arguments of the submission |
| `HISTORY.md` | how the twelve members came about, and the names used in the code |

## Setup

```bash
conda create -n tags python=3.11 && conda activate tags
pip install -r requirements.txt
```

Data and weights (defaults; override with the variable in brackets, see `tags/paths.py`):
* `data/ReXGroundingCT/` (`TAGS_REXG_ROOT`): `MICCAI_challenge_dataset.json`, `segmentations/`
  ([ReXGroundingCT](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT));
* `data/CT-RATE/dataset/` (`TAGS_CTRATE_ROOT`): `train_fixed/`, `valid_fixed/`, `ts_seg/`, `radiology_text_reports/`
  ([CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE));
* `weights/voxtell_v1.1/` (`TAGS_VOXTELL_DIR`): VoxTell v1.1 (`plans.json`, `fold_0/checkpoint_final.pth`);
* `tapct_maps/{val200_prob,test300_prob}/`: TAP-CT probability maps, produced by `scripts/50_tapct.sh`.

## Pretrained weights

The submitted checkpoints (12 VoxTell members, 5 TAP-CT seeds, threshold model, component head; about 22 GB) are on
Hugging Face: [fomofo/TAGS-ReXGroundingCT](https://huggingface.co/fomofo/TAGS-ReXGroundingCT) (CC BY-NC-SA 4.0 /
CC BY-NC 4.0, non-commercial). `python tags/hf_weights.py` downloads them and writes them where the scripts expect them,
so `scripts/40`, `50` (export steps) and `70` can be run without training.

## Running

Run from the repository root, in order.

| script | what | hardware |
|---|---|---|
| `scripts/00_prepare_data.sh` | indices, text embeddings, text augmentation, lobe mapping, CT-RATE pseudo-label pool | 1 GPU (80 GB for the LLM steps) |
| `scripts/20_train_members.sh all` | the 12 members | 1 GPU each |
| `scripts/20_train_members.sh train` | train-only twins and their validation maps (for scripts/60) | 1 GPU each |
| `scripts/40_predict_test.sh` | test probability maps (3.7 min per scan per member on an H100) | 1 GPU |
| `scripts/50_tapct.sh` | TAP-CT: caches, text banks, training, test maps | 1 GPU |
| `scripts/60_fit_postprocessing.sh` | threshold model and component head | CPU |
| `scripts/70_build_submission.sh` | ensemble + post-processing → submission zip | CPU |

## Notes

* **Reproducibility.** Inference and post-processing (scripts 40–70) are the code of the submission: applied to our
  probability maps they reproduce it voxel for voxel, and the weights above are the submitted checkpoints. The training script is the simplified recipe described above,
  and training was not seeded, so retrained members differ from the submitted ones.
* **Test-set exposure.** The CT-RATE pseudo-label pool used for the submission (members `aux_*`) excluded validation
  patients but not test patients: 80 of the 300 test studies were in it, as image, CT-RATE report sentence and
  TotalSegmentator pseudo-mask. Retraining both members on a pool without them changed the final masks by mean Dice
  0.025, equally on exposed and other studies. The scripts here build and use the pool without test patients.
* **Included data.** `tags/rexg_ts/` holds the LLM lobe mapping of all challenge sentences used for the submission
  (CC BY-NC-SA 4.0, derived from ReXGroundingCT).
* **TAP-CT.** The encoder is downloaded from Hugging Face (revision pinned in `tapct/lattice/models.py`);
  set `PRIMUS_WIN_Z=24 PRIMUS_Z_MM=1.5` as in `scripts/50_tapct.sh`.

## Licence

Code: Apache License 2.0 (`LICENSE`, third-party notices in `NOTICE`). `tags/rexg_ts/`: CC BY-NC-SA 4.0.
VoxTell v1.1 weights (CC BY-NC-SA 4.0), TAP-CT weights (CC BY-NC 4.0), ReXGroundingCT and CT-RATE (CC BY-NC-SA 4.0)
have their own licences; models trained with this code on them are subject to those.

## Citation

Paper in preparation.

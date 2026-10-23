# How the submitted models were actually trained

The scripts in this repository train every member in one go with the final recipe. The submitted members were not
all trained that way: this file records what was run, so the submission can be understood. The exact commands and
logs of every stage are kept in our internal archive.

| name in this repository | name in the submission | submitted training |
|---|---|---|
| `tversky_fp_2` | `cautious` | base model with the 14-category encoder (250 ep) → +60 ep Tversky 0.3/0.7 → +60 ep val×5 → +60 ep final recipe |
| `aux_effusion` | `aux2e` | same chain, with CT-RATE effusion pseudo-labels in the first fine-tuning stage |
| `aux_nodule_effusion` | `auxdil2` | same chain, with CT-RATE nodule + effusion pseudo-labels in the first fine-tuning stage |
| `fusion24` | `fusion24` | same chain, with the 24³ fusion grid from the first fine-tuning stage on |
| `lobe_aux` | `lobeaux` | 120 ep train-only with lobe-mask target → 130 ep train+val → +60 ep final recipe |
| `plain`, `plain_2` | `fleet_plain`, `fleet_seed2` | final recipe, as here |
| `tversky_fp`, `tversky_fn` | `fleet_tvA`, `fleet_tvB` | final recipe, as here |
| `nodule_boost15`, `nodule_boost20`, `category_balance` | `mb_2dboost`, `mb_boost20`, `mb_catbal` | final recipe, as here |

The post-processing was fitted on validation maps of nine train-only twins (the first four as single 60-epoch
fine-tunes of a train-only base model); here all twelve twins are used.

## The story

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

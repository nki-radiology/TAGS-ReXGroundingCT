#!/bin/bash
# Ensemble + post-processing -> submission zip (TAGS_TAPv3). CPU only.
set -euo pipefail
cd tags
M="plain;plain_2;tversky_fp;tversky_fp_2;tversky_fn;aux_effusion;aux_nodule_effusion;fusion24;lobe_aux;nodule_boost15;nodule_boost20;category_balance"
N=TAGS_TAPv3
# (1) mean of the 12 members, fused 0.8/0.2 with TAP-CT; (2) restriction to the named lobe (dilated 16) or lung (dilated 12);
# per-finding threshold from the threshold model
LOBE_K=16 python build_ensemble_submission.py --members "${M//;/,}" --thr 0.35 --side --dilate-side 12 \
    --thr-model results/thr_model_lobe16.pkl --tap "${TAGS_TAPCT_TEST_MAPS:-../tapct_maps/test300_prob}" \
    --tap-w 0.2 --workers 6 --end 0 --out submission/VARIANT_$N
# (3) component head drops components predicted to lower Dice
KH_SRC=submission/VARIANT_$N KH_DST=submission/VARIANT_${N}_HEAD4 KH_TAU=0.7 KH_MEMBERS="$M" KH_WORKERS=6 python apply_keep_head.py
# (4) voxels claimed by several findings go to the most confident one (margin 0.2)
AC_SRC=submission/VARIANT_${N}_HEAD4 AC_DST=submission/VARIANT_${N}_FINAL AC_MODE=vox:0.2 AC_MEMBERS="$M" AC_WORKERS=6 python apply_case.py
python make_submission_fast.py --src submission/VARIANT_${N}_FINAL --zip submission/$N.zip

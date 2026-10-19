# Added to nnU-Net (https://github.com/MIC-DKFZ/nnUNet, Apache-2.0) by the fomo.fo team, 2026: registers the VoxTell trainers.
from voxtell.training.voxtell_trainer import (
    VoxTellTrainer,
    VoxTellTrainer_noMirroring,
    VoxTellTrainer_250epochs,
)
from voxtell.training.voxtell_trainer_region import (
    VoxTellTrainer_region_250ep,
    VoxTellTrainer_region_bal_250ep,
)
from voxtell.training.voxtell_trainer_tversky import (
    VoxTellTrainer_region_tversky,
    VoxTellTrainer_region_presence_tversky,
    VoxTellTrainer_region_tversky_b90,
    VoxTellTrainer_region_tversky_b95,
    VoxTellTrainer_region_presence_tversky_b90,
)

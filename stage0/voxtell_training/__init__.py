# Modified from VoxTell (https://github.com/MIC-DKFZ/VoxTell, Apache-2.0) by the fomo.fo team, 2026:
# trainer variants for the 14-category ReXGroundingCT region model. Cluster paths replaced by env vars.
from voxtell.training.voxtell_trainer import (
    VoxTellTrainer,
    VoxTellTrainer_noMirroring,
    VoxTellTrainer_250epochs,
)
from voxtell.training.voxtell_trainer_region import (
    VoxTellTrainer_region_250ep,
    VoxTellTrainer_region_bal_250ep,
    FocalDataLoader,
)
from voxtell.training.voxtell_trainer_tversky import (
    VoxTellTrainer_region_tversky,
    VoxTellTrainer_region_presence_tversky,
    VoxTellTrainer_region_tversky_b90,
    VoxTellTrainer_region_tversky_b95,
    VoxTellTrainer_region_presence_tversky_b90,
    MemoryEfficientTverskyLoss,
    PresenceFocalDataLoader,
)

__all__ = [
    "VoxTellTrainer",
    "VoxTellTrainer_noMirroring",
    "VoxTellTrainer_250epochs",
    "VoxTellTrainer_region_250ep",
    "VoxTellTrainer_region_bal_250ep",
    "FocalDataLoader",
    "VoxTellTrainer_region_tversky",
    "VoxTellTrainer_region_presence_tversky",
    "MemoryEfficientTverskyLoss",
    "PresenceFocalDataLoader",
]

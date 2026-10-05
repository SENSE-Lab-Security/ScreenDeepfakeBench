# Adapted from DeepfakeBench; see NOTICE.md and LICENSE.
from metrics.registry import LOSSFUNC
from .cross_entropy_loss import CrossEntropyLoss
from .l1_loss import L1Loss

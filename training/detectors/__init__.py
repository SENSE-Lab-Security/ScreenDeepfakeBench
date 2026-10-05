# Adapted from DeepfakeBench; see NOTICE.md and LICENSE.
from metrics.registry import DETECTOR
from .xception_detector import XceptionDetector
from .ffd_detector import FFDDetector
from .spsl_detector import SpslDetector

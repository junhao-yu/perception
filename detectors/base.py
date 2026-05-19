from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class Detection:
    bbox: tuple        # (x1, y1, x2, y2) in pixels, int
    label: str         # class name
    score: float       # confidence [0,1]
    mask: np.ndarray = None  # optional segmentation mask


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run detection on a BGR image. Returns list of Detection."""
        ...

    @abstractmethod
    def warmup(self, image: np.ndarray) -> None:
        """Run a warmup pass to compile/prepare."""
        ...

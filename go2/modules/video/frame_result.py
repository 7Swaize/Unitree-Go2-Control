import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class FrameResult:
    """
    Holds the output of a single camera capture.

    Attributes
    ----------
    color : Optional[np.ndarray]
        BGR color image (H, W, 3). Always present for color/RGB cameras.
    depth : Optional[np.ndarray]
        Depth image (H, W) in uint16. Only present for depth cameras.
    """

    color: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None

    @property
    def has_any(self) -> bool:
        """True if any frame is present."""
        return any(f is not None for f in (self.color, self.depth))

    @property
    def has_color(self) -> bool:
        """True if a rgb frame is present."""
        return self.color is not None

    @property
    def has_depth(self) -> bool:
        """True if a depth frame is present."""
        return self.depth is not None
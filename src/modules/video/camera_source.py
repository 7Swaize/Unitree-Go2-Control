"""Camera Source Abstract Base Class"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any
import numpy as np


class CameraSource(ABC):
    """
    Abstract base class for all camera sources.

    A ``CameraSource`` provides a unified interface for acquiring image frames
    from different camera backends (SDK-based cameras, OpenCV webcams,
    depth cameras, etc.).
    """

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the camera source."""
        pass
    
    @abstractmethod
    def get_frames(self) -> Tuple[int, Optional[Any]]:
        """Retrieve the next frame(s) from the camera source."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shut down the camera source and release any resources."""
        pass

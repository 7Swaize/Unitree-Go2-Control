from collections.abc import Iterator

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class FrameResult:
    """Holds the output of a single camera capture."""

    color: Optional[np.ndarray] = None  #: BGR color image (H, W, 3). Always present for color/RGB cameras.
    depth: Optional[np.ndarray] = None  #: Depth image (H, W) in uint16. Only present for depth cameras.

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
    


@dataclass
class MultiFrameResult:
    """
    Holds frame results from a group of named cameras.
    """

    frames: Dict[str, FrameResult] #: Maps camera name to its latest FrameResult, or an empty FrameResult if that camera had no frame ready this cycle.

    def __getitem__(self, key: str) -> FrameResult:
        """
        Direct access by camera name.

        Raises
        ------
        KeyError
            If camera of 'key' name is not in group.
        """
        return self.frames[key] 

    def __iter__(self) -> Iterator[str]:
        return iter(self.frames)
    
    def available_frames(self) -> Iterator[tuple[str, FrameResult]]:
        """An iterable collection of (name, result) pairs where a frame was actually captured."""
        for name, result in self.frames.items():
            if result.has_any:
                yield name, result

    def all_available(self) -> bool:
        """True only if every camera in the group has a frame ready."""
        return all(
            result.has_any for result in self.frames.values()
        )
    
    def missing(self) -> List[str]:
        """Return names of cameras that had no frame this cycle."""
        return [
            name for name, result in self.frames.items() if not result.has_any
        ]
    
    @property
    def camera_names(self) -> List[str]:
        return list(self.frames.keys())
    

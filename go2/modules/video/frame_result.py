import numpy as np
from enum import Enum, auto
from collections.abc import Iterator
from typing import Dict, List, Optional
from dataclasses import dataclass


class FrameStatus(Enum):
    """Describes the state of a :class:`FrameResult` object."""
    OK = auto() #: Frame data is present and usable
    PENDING = auto() #: Source initialised but no frame was currently available in a frame buffer


@dataclass
class FrameResult:
    """Holds one captured moment from a single camera source."""

    status: FrameStatus = FrameStatus.PENDING
    color: Optional[np.ndarray] = None  #: BGR color image (H, W, 3). Always present for color/RGB cameras.
    depth: Optional[np.ndarray] = None  #: Depth image (H, W) in uint16. Only present for depth cameras.

    @classmethod
    def color_only(cls, color: np.ndarray) -> "FrameResult":
        """
        Source is RGB-only; depth will never be present.

        Notes
        -----
        Static factory method for internal use only. Should not be called by the user.
        """
        return cls(status=FrameStatus.OK, color=color)

    @classmethod
    def depth_only(cls, depth: np.ndarray) -> "FrameResult":
        """
        Source is depth-only; color will never be present.
        
        Notes
        -----
        Static factory method for internal use only. Should not be called by the user.
        """
        return cls(status=FrameStatus.OK, depth=depth)

    @classmethod
    def color_and_depth(cls, color: np.ndarray, depth: np.ndarray) -> "FrameResult":
        """
        Source provides both channels (e.g. RealSense aligned frames).
        
        Notes
        -----
        Static factory method for internal use only. Should not be called by the user.
        """
        return cls(status=FrameStatus.OK, color=color, depth=depth)

    @classmethod
    def pending(cls) -> "FrameResult":
        """
        Source is initialised but the capture buffer hasn't filled yet.

        Notes
        -----
        Static factory method for internal use only. Should not be called by the user.
        """
        return cls(status=FrameStatus.PENDING)


    def is_fully_valid(self) -> bool:
        """True only when status is OK and all channels are present."""
        return self.status is FrameStatus.OK and (self.color is not None and self.depth is not None)

    def has_color(self) -> bool:
        """True if a valid rgb frame is present."""
        return self.color is not None

    def has_depth(self) -> bool:
        """True if a valid depth frame is present."""
        return self.depth is not None
    


@dataclass
class MultiFrameResult:
    """Holds frame results from a group of named cameras."""

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
    
    def valid_frames(self) -> dict[str, FrameResult]:
        """Only slots whose FrameResult.is_valid is True."""
        return {name: fr for name, fr in self.frames.items() if fr.is_fully_valid}

    def is_fully_valid(self) -> bool:
        """True only when every slot has all valid frames"""
        return all(fr.is_fully_valid for fr in self.frames.values())

    def pending_slots(self) -> list[str]:
        """Slots that are initialised but haven't produced a frame yet."""
        return [
            name for name, fr in self.frames.items() if fr.status is FrameStatus.PENDING
        ]
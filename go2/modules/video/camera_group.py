from typing import Dict

from .camera_source import CameraSource
from .frame_result import MultiFrameResult


class CameraGroup:
    """
    Manages a named collection of camera sources as a single unit.
    Frames from all cameras are returned together as a :class:`MultiFrameResult` keyed by the names you assign.

    Parameters
    ----------
    sources : Dict[str, CameraSource]
        A mapping of camera name to camera source.
    """

    def __init__(self, sources: Dict[str, CameraSource]) -> None:
        if not sources:
            raise ValueError("[Video] CameraGroup requires atleast one camera source")

        self._sources = sources

    def _initialize(self) -> None:
        for name, source in self._sources.items():
            source._initialize()

    def _get_frames(self) -> MultiFrameResult:
        return MultiFrameResult(frames={
            name: source._get_frames() for name, source in self._sources.items()
        })
    
    def _shutdown(self) -> None:
        for name, source in self._sources.items():
            source._shutdown()

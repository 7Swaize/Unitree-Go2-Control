from abc import ABC, abstractmethod
from ..frame_result import FrameResult


class CameraSource(ABC):
    @abstractmethod
    def _start(self) -> None:
        pass
    
    @abstractmethod
    def _get_frames(self) -> FrameResult:
        pass

    @abstractmethod
    def _shutdown(self) -> None:
        pass



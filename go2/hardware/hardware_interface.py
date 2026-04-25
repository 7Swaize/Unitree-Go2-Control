import sys
import time

from abc import ABC, abstractmethod
from typing import Tuple


class HardwareInterface(ABC):
    """
    Abstract interface for dog hardware control.

    This interface defines the minimum set of motion and posture commands required by the system.

    Notes
    -----
    - This is an internal abstraction.
    - Users should not subclass this directly.
    """
    
    @abstractmethod
    def _initialize(self) -> None:
        """
        Initialize the hardware interface. This is handled automatically and shouldn't be called by users.
        Hardware initialization is linked to the core controller's initialization. The hardware is guaranteed to be 
        initialized before any other system provided by this package.

        Establishes communication with the native or virtual backend.
        """
        pass
    
    @abstractmethod
    def _shutdown(self) -> None:
        """
        Cleanly shut down the hardware interface. This should not be called by users.
        """
        pass
    
    @abstractmethod
    def _move(self, vx: float, vy: float) -> None:
        """
        Move the dog in the horizontal plane.

        Parameters
        ----------
        vx : float
            Forward/backward velocity.
        vy : float
            Left/right velocity.
        """
        pass

    @abstractmethod
    def _rotate(self, vrot: float):
        """
        Rotate the dog in place.

        Parameters
        ----------
        vrot : float
            Rotational velocity.
        """
        pass

    @abstractmethod
    def _stand_up(self) -> None:
        """Command the dog to stand up."""
        pass
    
    @abstractmethod
    def _stand_down(self) -> None:
        """Command the dog to lie down."""
        pass
    
    @abstractmethod
    def _stop_move(self) -> None:
        """Immediately stop all movement and clear internal movement command buffer."""
        pass

    @abstractmethod
    def _damp(self) -> None:
        pass

    @abstractmethod
    def _balance_stand(self) -> None:
        pass

    @abstractmethod
    def _recovery_stand(self) -> None:
        pass

    @abstractmethod
    def _euler(self, roll: float, pitch: float, yaw: float) -> None:
        pass

    @abstractmethod
    def _sit(self) -> None:
        pass

    @abstractmethod
    def _rise_sit(self) -> None:
        pass

    @abstractmethod
    def _speed_level(self, level: int) -> None:
        pass

    @abstractmethod
    def _hello(self) -> None:
        pass

    @abstractmethod
    def _stretch(self) -> None:
        pass

    @abstractmethod
    def _content(self) -> None:
        pass

    @abstractmethod
    def _dance1(self) -> None:
        pass

    @abstractmethod
    def _dance2(self) -> None:
        pass

    @abstractmethod
    def _switch_joystick(self, on: bool) -> None:
        pass

    @abstractmethod
    def _pose(self, flag: bool) -> None:
        pass

    @abstractmethod
    def _scrape(self) -> None:
        pass

    @abstractmethod
    def _front_flip(self) -> None:
        pass

    @abstractmethod
    def _front_jump(self) -> None:
        pass

    @abstractmethod
    def _front_pounce(self) -> None:
        pass

    @abstractmethod
    def _heart(self) -> None:
        pass

    @abstractmethod
    def _left_flip(self) -> None:
        pass

    @abstractmethod
    def _back_flip(self) -> None:
        pass

    @abstractmethod
    def _free_walk(self) -> None:
        pass

    @abstractmethod
    def _free_bound(self, flag: bool) -> None:
        pass

    @abstractmethod
    def _free_jump(self, flag: bool) -> None:
        pass

    @abstractmethod
    def _free_avoid(self, flag: bool) -> None:
        pass

    @abstractmethod
    def _walk_upright(self, flag: bool) -> None:
        pass

    @abstractmethod
    def _cross_step(self, flag: bool) -> None:
        pass

    @abstractmethod
    def _static_walk(self) -> None:
        pass

    @abstractmethod
    def _trot_run(self) -> None:
        pass

    @abstractmethod
    def _hand_stand(self, flag: bool) -> None:
        pass

    @abstractmethod
    def _classic_walk(self, flag: bool) -> None:
        pass

    @abstractmethod
    def _auto_recovery_set(self, enabled: bool) -> None:
        pass

    @abstractmethod
    def _auto_recovery_get(self) -> Tuple[int, bool | None]:
        pass

    @abstractmethod
    def _switch_avoid_mode(self) -> None:
        pass


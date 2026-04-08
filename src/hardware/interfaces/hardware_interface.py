import sys
import time

from abc import ABC, abstractmethod


class HardwareInterface(ABC):
    """
    Abstract interface for dog hardware control.

    This interface defines the minimum set of motion and posture commands required by the system.

    Notes
    -----
    - This is an internal abstraction.
    - Students should not subclass this directly.
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

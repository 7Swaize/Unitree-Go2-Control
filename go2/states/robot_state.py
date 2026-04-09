"""
State Machine Approach for User Use
===================================

Abstract base class for all robot behavior states.

Users create custom states by inheriting from this class and implementing
the :meth:`execute` method. This approach is completely optional. It is solely available
for those who are comfortable with a more advanced approach.  

This framework provides:
    - Automatic loop cancellation
    - Safe shutdown handling
    - Consistent state lifecycle hooks
"""


from abc import ABC, abstractmethod
from typing import Any, final

from ..core.controller import Go2Controller
from .validation import CancellableMeta


class Go2StateAbstract(ABC, metaclass=CancellableMeta):
    """Abstract base class for robot behavior states."""
    
    def __init__(self, unitree_controller: Go2Controller):
        """
        Initialize a robot behavior state.

        Parameters
        ----------
        unitree_controller : Go2Controller
            High-level controller providing robot control APIs
        """
        super().__init__()
        self.unitree_controller = unitree_controller
        self.is_running = False
        self.should_cancel = False

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any: # might have to preserve var args in source gen?
        """
        Main execution logic for the state.

        This method is automatically instrumented to support safe shutdown.
        Any ``for`` or ``while`` loop inside this method will automatically
        check for system shutdown.

        Notes
        -----
        - Long-running logic is safe by default
        """
        pass

    def on_enter(self):
        """
        Called when the state becomes active.

        Sets internal running flags.
        """
        self.is_running = True
        self.should_cancel = False

    def on_exit(self):
        """
        Called when the state exits.

        Cleans up internal state.
        """
        self.is_running = False

    @final
    def check_shutdown(self):
        """
        Check whether the system is shutting down.

        Raises
        ------
        KeyboardInterrupt
            If the controller shutdown event is set
        """
        if self.unitree_controller is None:
            return
        
        if self.unitree_controller._shutdown_event.is_set():
            raise KeyboardInterrupt()

    def cancel(self) -> None:
        """
        Request cancellation of the current state.

        Notes
        -----
        - This flag can be checked manually by user logic
        - Loop-based logic is already auto-cancellable
        """
        self.should_cancel = True

    
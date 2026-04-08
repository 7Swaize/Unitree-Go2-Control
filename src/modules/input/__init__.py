"""Input Module for Student Use"""

from typing import Callable, Optional

from modules.input.callback_manager import InputSignalCallbackManager, UnitreeRemoteControllerInputParser
from modules.input.controller_state import ControllerState
from modules.input.input_signal import InputSignal

from core.module import DogModule
from communication.dds import DDS_TOPICS


class InputModule(DogModule):
    """
    High-level interface for handling controller input.

    Attributes
    ----------
    use_sdk : bool
        Whether to use the Unitree SDK for input.
    """
    
    def __init__(self, use_sdk: bool = False):
        super().__init__("Input")
        self.use_sdk = use_sdk


    def initialize(self) -> None:
        """Set up input parsing and callback management."""
        if self._initialized or not self.use_sdk:
            return
        
        try:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
            from unitree_sdk2py.core.channel import ChannelSubscriber

            self._input_parser = UnitreeRemoteControllerInputParser()
            self._callback_manager = InputSignalCallbackManager()
            self._ControllerState = ControllerState
            self._InputSignal = InputSignal

            self._lowstate_subscriber = ChannelSubscriber(DDS_TOPICS['LOW_STATE'], LowState_)
            self._lowstate_subscriber.Init(self._process_input, 10)
        except ImportError:
            print("[Input] Unitree SDK not available, input module in simulation mode")
            self._input_parser = UnitreeRemoteControllerInputParser()
            self._callback_manager = InputSignalCallbackManager()

        self._initialized = True

    def register_callback(
        self,
        signal: InputSignal,
        callback: Callable[[ControllerState], None],
        name: Optional[str] = None,
        threshold: float = 0.1
    ):
        """
        Register a callback for a specific controller signal.

        Parameters
        ----------
        signal : InputSignal
            The input signal to monitor (e.g., BUTTON_A, LEFT_STICK_X)
        callback : Callable[[ControllerState], None]
            Function to call when the signal triggers
        name : str, optional
            Optional human-readable name for the callback
        threshold : float, optional
            Threshold for analog inputs (default 0.1)
        """
        return self._callback_manager.register(signal, callback, name, threshold)
    
    def unregister_callback(
        self,
        signal: InputSignal,
        callback: Callable[[ControllerState], None]
    ) -> None:
        """Unregister a previously registered callback."""
        self._callback_manager.unregister(signal, callback)
    
    def shutdown(self) -> None:
        """Clean up input resources."""
        if hasattr(self, '_lowstate_subscriber') and self._lowstate_subscriber:
            self._lowstate_subscriber.Close()
        if hasattr(self, '_callback_manager') and self._callback_manager:
            self._callback_manager.shutdown()
        self._initialized = False

    def _process_input(self, msg) -> ControllerState:
        """Internal method: process incoming controller messages."""
        controller_state = self._input_parser.parse(msg.wireless_remote)
        self._callback_manager.handle(controller_state)
        return controller_state


__all__ = ["InputModule", "InputSignal", "ControllerState"]

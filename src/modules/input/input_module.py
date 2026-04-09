from typing import Callable, Optional
from typing_extensions import override

from .callback_manager import InputSignalCallbackManager, UnitreeRemoteControllerInputParser
from .controller_state import ControllerState
from .input_signal import InputSignal

from ...core.module import DogModule
from ...communication.dds.dds import DDS_TOPICS


class InputModule(DogModule):
    """
    High-level interface for handling controller input. 
    This model is really only functional and practical, when on native hardware. It doesn't do anything on virtual hardware.

    Attributes
    ----------
    use_sdk : bool
        Whether to use the Unitree SDK for input.
    """
    
    def __init__(self, use_sdk: bool = False):
        super().__init__("Input")
        self.use_sdk = use_sdk


    @override
    def _initialize(self) -> None:
        """
        Set up input parsing and callback management. This is called internally,
        and should not be called directly by users.

        Notes
        -----
        - If `use_sdk` is False, no live input is initialized
        - Internal: subscribes to DDS LOW_STATE topic for live controller messages
        """
        if self._initialized or not self.use_sdk:
            return
        
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
        from unitree_sdk2py.core.channel import ChannelSubscriber

        self._input_parser = UnitreeRemoteControllerInputParser()
        self._callback_manager = InputSignalCallbackManager()
        self._ControllerState = ControllerState
        self._InputSignal = InputSignal

        self._lowstate_subscriber = ChannelSubscriber(DDS_TOPICS['LOW_STATE'], LowState_)
        self._lowstate_subscriber.Init(self._process_input, 10)

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

        Returns
        -------
        Callback
            Registered callback object
        """
        return self._callback_manager._register(signal, callback, name, threshold)
    
    def unregister_callback(
        self,
        signal: InputSignal,
        callback: Callable[[ControllerState], None]
    ) -> None:
        """
        Unregister a previously registered callback.

        Parameters
        ----------
        signal : InputSignal
            The signal whose callback should be removed
        callback : Callable[[ControllerState], None]
            The previously registered callback function
        """
        self._callback_manager._unregister(signal, callback)
    
    @override
    def _shutdown(self) -> None:
        """
        Clean up input resources. This is handled automatically and shouldn't be called by users.

        Notes
        -----
        - Stops DDS subscription and clears all callbacks
        - Should be called when input is no longer needed
        """
        if self._lowstate_subscriber:
            self._lowstate_subscriber.Close()
        if self._callback_manager:
            self._callback_manager._shutdown()

    def _process_input(self, msg) -> ControllerState:
        """
        Process incoming controller messages.

        Parameters
        ----------
        msg : object
            DDS message containing raw controller data

        Returns
        -------
        ControllerState
            Current controller state after parsing
        """
        controller_state = self._input_parser._parse(msg.wireless_remote)
        self._callback_manager._handle(controller_state)

        return controller_state
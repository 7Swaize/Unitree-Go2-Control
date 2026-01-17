"""
Input Module for Student Use
============================

Handles remote controller input for the robot. This module can only function on the Unitree-Go2, with a correctly paired remote controller. 

Students should not access or construct this class directly. Rather, they should access it through the :class:`~src.core.unitree_control_core.UnitreeGo2Controller` instance.


This module allows students to:
    - Access the current controller state
    - Register callbacks for specific input signals (buttons, sticks, triggers)
    - Cleanly shutdown input resources

Internally, it wraps the low-level Unitree SDK or controller parser, but students
interact only with the high-level :class:`InputModule` interface.

Example
-------
>>> from src.core.unitree_control_core import UnitreeGo2Controller
>>> 
>>> def on_b_pressed(state):
...     print("Button A pressed!")
>>>
>>> unitree_controller = UnitreeGo2Controller(sdk_enabled=True) # SDK enabled is key for functionality
>>> unitree_controller.input.register_callback(InputSignal.BUTTON_B, on_b_pressed)
"""


from typing import Callable, Optional
from src.controller_input_control.callback_manager import InputSignalCallbackManager, UnitreeRemoteControllerInputParser
from src.controller_input_control.controller_state import ControllerState
from src.controller_input_control.input_signal import InputSignal
from src.core.base_module import DogModule
from src.dds.dds_constants import DDS_TOPICS


class InputModule(DogModule):
    """
    High-level interface for handling controller input.

    Attributes
    ----------
    use_sdk : bool
        Whether to use the Unitree SDK for input. This model is really only functional
        and practical, when the sdk mode is enabled.
    """
    
    def __init__(self, use_sdk: bool = False):
        super().__init__("Input")
        self.use_sdk = use_sdk


    def initialize(self) -> None:
        """
        Set up input parsing and callback management. This is called internally,
        and should not be called directly by students.

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

        Example
        -------
        >>> def on_left_stick(state):
        ...     print("Left stick moved:", state.lx, state.ly)
        >>> input_module.register_callback(InputSignal.LEFT_STICK, on_left_stick)
        """
        return self._callback_manager.register(signal, callback, name, threshold)
    
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
        self._callback_manager.unregister(signal, callback)
    
    def shutdown(self) -> None:
        """
        Clean up input resources. This is handled automatically and shouldn't be called by students.

        Notes
        -----
        - Stops DDS subscription and clears all callbacks
        - Should be called when input is no longer needed
        """
        if self._lowstate_subscriber:
            self._lowstate_subscriber.Close()
        if self._callback_manager:
            self._callback_manager.shutdown()

    def _process_input(self, msg) -> ControllerState:
        """
        Internal method: process incoming controller messages.

        Parameters
        ----------
        msg : object
            DDS message containing raw controller data

        Returns
        -------
        ControllerState
            Current controller state after parsing
        """
        controller_state = self._input_parser.parse(msg.wireless_remote)
        self._callback_manager.handle(controller_state)

        return controller_state
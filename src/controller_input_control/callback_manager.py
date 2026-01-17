from dataclasses import dataclass
import struct
from typing import Callable, Dict, List, Optional
from src.controller_input_control.controller_state import ControllerState
from src.controller_input_control.input_signal import InputSignal


_ANALOG_SIGNALS = {
    InputSignal.LEFT_STICK_X, InputSignal.LEFT_STICK_Y,
    InputSignal.RIGHT_STICK_X, InputSignal.RIGHT_STICK_Y,
    InputSignal.LEFT_TRIGGER, InputSignal.RIGHT_TRIGGER
}


@dataclass
class Callback:
    """
    Represents a registered callback for a controller input signal.

    Attributes
    ----------
    callback : Callable[[ControllerState], None]
        Function to call when the signal triggers
    signal : InputSignal
        Signal associated with this callback
    name : Optional[str]
        Human-readable identifier for the callback
    threshold : float
        Minimum change required to trigger for analog inputs
    """
    callback: Callable[[ControllerState], None]
    signal: InputSignal
    name: Optional[str] = None
    threshold: float = 0.1


class InputSignalCallbackManager:
    """
    Internal manager for controller input callbacks.

    Responsibilities:
        - Register/unregister callbacks
        - Detect analog and digital changes
        - Execute callbacks when signals change

    Notes
    -----
    - Analog signals have configurable thresholds
    - Stick movement is measured as vector distance
    - Call ``handle(state)`` with the latest ControllerState each update
    """
    def __init__(self):
        self._callbacks: Dict[InputSignal, List[Callback]] = {}
        self._previous_state = ControllerState()

    def register(
            self,
            signal: InputSignal,
            callback: Callable[[ControllerState], None],
            name: Optional[str] = None,
            threshold: float = 0.1    
        ) -> Callback:
        """
        Register a callback for a signal.

        Parameters
        ----------
        signal : InputSignal
            The controller signal to monitor
        callback : Callable[[ControllerState], None]
            Function to call when signal triggers
        name : Optional[str]
            Human-readable name for the callback
        threshold : float
            Threshold for analog inputs

        Returns
        -------
        Callback
            The registered callback object
        """
        cb = Callback(
            callback=callback,
            signal=signal,
            name=name or getattr(callback, "__name__", f"<lambda:{id(callback)}>"),
            threshold=threshold
        )

        self._callbacks.setdefault(signal, []).append(cb)
        return cb


    def unregister(self, signal: InputSignal, callback: Callable[[ControllerState], None]) -> None:
        """
        Unregister a previously registered callback.

        Parameters
        ----------
        signal : InputSignal
        callback : Callable[[ControllerState], None]
        """
        if signal in self._callbacks:
            self._callbacks[signal] = [cb for cb in self._callbacks[signal] if cb.callback != callback]
            
            if not self._callbacks[signal]:
                del self._callbacks[signal]


    def handle(self, state: ControllerState) -> None:
        """
        Evaluate the current controller state and execute callbacks.

        Parameters
        ----------
        state : ControllerState
            Latest controller state
        """
        if not state.changed:
            return

        for signal, cb_list in self._callbacks.items():
            for cb in cb_list:
                if self.should_trigger(signal, cb, state):
                    self.execute(cb, state)

        self._previous_state = ControllerState(**state.__dict__)


    def shutdown(self) -> None:
        """Clear all callbacks and reset manager state."""
        self._callbacks.clear()


    def execute(self, cb: Callback, state: ControllerState) -> None:
        """Invoke a callback safely with error handling."""
        try:
            cb.callback(state)
        except Exception as e:
            print(f"[CallbackManager] Callback {cb.name} failed: {e}")


    def should_trigger(self, signal: InputSignal, cb: Callback, current_state: ControllerState) -> bool:
        """
        Determine if a callback should fire based on signal changes.

        Parameters
        ----------
        signal : InputSignal
        cb : Callback
        current_state : ControllerState

        Returns
        -------
        bool
            True if the signal change exceeds threshold
        """
        if signal == InputSignal.LEFT_STICK:
            return self.stick_changed('l', current_state, cb.threshold)
            
        if signal == InputSignal.RIGHT_STICK:
            return self.stick_changed('r', current_state, cb.threshold)
        
        signal_attr = signal.value
        if not hasattr(current_state, signal_attr):
            return False
        
        current_value = getattr(current_state, signal_attr, 0.0)
        previous_value = getattr(self._previous_state, signal_attr, 0.0)

        if signal in _ANALOG_SIGNALS:
            return self.check_analog_trigger(current_value, previous_value, cb)
        
        return self.check_digital_trigger(current_value, previous_value, cb)
        

    def stick_changed(self, stick: str, current_state: ControllerState, threshold: float) -> bool:
        """Check if stick movement exceeds threshold (vector magnitude)."""
        x_attr = f"{stick}x"
        y_attr = f"{stick}y"
        
        current_x = getattr(current_state, x_attr, 0.0)
        current_y = getattr(current_state, y_attr, 0.0)
        previous_x = getattr(self._previous_state, x_attr, 0.0)
        previous_y = getattr(self._previous_state, y_attr, 0.0)
        
        # vector magnitude of x and y components
        distance = ((current_x - previous_x) ** 2 + (current_y - previous_y) ** 2) ** 0.5
        return distance > threshold


    def check_analog_trigger(self, current: float, previous: float, cb: Callback) -> bool:
        """Return True if analog change exceeds threshold."""
        return abs(current - previous) > cb.threshold
    

    def check_digital_trigger(self, current: float, previous: float, cb: Callback) -> bool:
        """Return True if digital button transitioned from 0 to 1."""
        return previous == 0.0 and current == 1.0
    

class UnitreeRemoteControllerInputParser:
    """
    Parses raw remote controller data into :class:`~src.controller_input_control.controller_state.ControllerState`.

    Responsibilities:
        - Decode analog and digital signals from raw bytes
        - Track state changes
        - Provide up-to-date ControllerState for callback handling

    Notes
    -----
    - Internal module only
    - Works with Unitree remote controller data format
    """
    def __init__(self):
        self._state = ControllerState()

    def parse_buttons(self, data1: int, data2: int):
        """
        Decode digital buttons from two byte sequences.

        Parameters
        ----------
        data1 : int
        data2 : int
        """
        mapping1 = {
            0: "r1", 1: "l1", 2: "start", 3: "select",
            4: "r2", 5: "l2", 6: "f1", 7: "f3"
        }
        mapping2 = {
            0: "a", 1: "b", 2: "x", 3: "y",
            4: "up", 5: "right", 6: "down", 7: "left"
        }

        for i, attr in mapping1.items():
            setattr(self._state, attr, (data1 >> i) & 1)

        for i, attr in mapping2.items():
            setattr(self._state, attr, (data2 >> i) & 1)


    def parse_analog(self, data: bytes):
        """
        Decode analog stick and trigger values from raw bytes.

        Parameters
        ----------
        data : bytes
        """
        self._state.lx = struct.unpack('<f', data[4:8])[0]
        self._state.ly = struct.unpack('<f', data[20:24])[0]
        self._state.rx = struct.unpack('<f', data[8:12])[0]
        self._state.ry = struct.unpack('<f', data[12:16])[0]
        self._state.l2 = struct.unpack('<f', data[16:20])[0]


    # remote_data is some type that is an array of 8-bit-seqs
    def parse(self, remote_data) -> ControllerState:
        """
        Parse raw remote input into :class:`~src.controller_input_control.controller_state.ControllerState`.

        Parameters
        ----------
        remote_data : Any
            Raw controller input (array of 8-bit sequences)

        Returns
        -------
        ControllerState
            Updated state with `changed` flag indicating modifications
        """
        self._previous_state = ControllerState(**self._state.__dict__)
        
        self.parse_analog(remote_data)
        self.parse_buttons(remote_data[2], remote_data[3])

        # just compares if there is a change
        self._state.changed = any(
            getattr(self._state, attr) != getattr(self._previous_state, attr)
            for attr in vars(self._state) if attr != "changed"
        )
    
        return self._state

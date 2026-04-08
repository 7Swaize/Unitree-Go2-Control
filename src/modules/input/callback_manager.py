"""Callback Manager for Input Signals"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .controller_state import ControllerState
from .input_signal import InputSignal


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
        """Register a callback for a signal."""
        cb = Callback(
            callback=callback,
            signal=signal,
            name=name or getattr(callback, "__name__", f"<lambda:{id(callback)}>"),
            threshold=threshold
        )

        self._callbacks.setdefault(signal, []).append(cb)
        return cb


    def unregister(self, signal: InputSignal, callback: Callable[[ControllerState], None]) -> None:
        """Unregister a previously registered callback."""
        if signal in self._callbacks:
            self._callbacks[signal] = [cb for cb in self._callbacks[signal] if cb.callback != callback]
            
            if not self._callbacks[signal]:
                del self._callbacks[signal]


    def handle(self, state: ControllerState) -> None:
        """Evaluate the current controller state and execute callbacks."""
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
        """Determine if a callback should fire based on signal changes."""
        try:
            current_val = getattr(current_state, signal.value, 0.0)
            previous_val = getattr(self._previous_state, signal.value, 0.0)
            
            if signal in _ANALOG_SIGNALS:
                return abs(current_val - previous_val) > cb.threshold
            else:
                return current_val != previous_val and current_val > 0.5
        except Exception as e:
            print(f"[CallbackManager] Error checking trigger for {signal}: {e}")
            return False


class UnitreeRemoteControllerInputParser:
    """
    Parser for Unitree remote controller messages.
    
    This is a simplified stub - the full implementation would parse
    DDS messages into ControllerState objects.
    """
    def __init__(self):
        pass
    
    def parse(self, wireless_remote_data):
        """Parse raw wireless remote data into ControllerState."""
        # This would be implemented fully with DDS message parsing
        return ControllerState()

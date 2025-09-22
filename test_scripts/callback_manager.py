from enum import Enum
from test_scripts.input_handle import ControllerState

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, TypeVar, cast
from dataclasses import dataclass

T = TypeVar("T")

class InputSignal(Enum):
    LEFT_STICK = "left_stick"
    RIGHT_STICK = "right_stick"
    LEFT_STICK_X = "lx"
    LEFT_STICK_Y = "ly"
    RIGHT_STICK_X = "rx"
    RIGHT_STICK_Y = "ry"
    
    LEFT_TRIGGER = "l2"
    RIGHT_TRIGGER = "r2"
    LEFT_BUMPER = "l1"
    RIGHT_BUMPER = "r1"
    
    BUTTON_A = "a"
    BUTTON_B = "b"
    BUTTON_X = "x"
    BUTTON_Y = "y"
    
    DPAD_UP = "up"
    DPAD_DOWN = "down"
    DPAD_LEFT = "left"
    DPAD_RIGHT = "right"
    
    SELECT = "select"
    START = "start"
    F1 = "f1"
    F3 = "f3"
    
    ANY_BUTTON = "any_button"
    ANY_ANALOG = "any_analog"
    ANY_CHANGE = "any_change"
    
@dataclass
class CallbackInfo[T]:
    callback: Callable[[T], None]
    name: Optional[str]
    enabled: bool

@dataclass
class SignalCallbackInfo[T](CallbackInfo[T]):
    signal: InputSignal
    threshold: float = 0.1

class CallbackInfoFactory[T]:
    def create(
        self,
        callback: Callable[[T], None],
        name: Optional[str] = None,
        enabled: bool = True,
        signal: Optional[InputSignal] = None,
        threshold: float = 0.1,
    ) -> CallbackInfo[T]:
        resolved_name = name or getattr(callback, "__name__", f"<lambda:{id(callback)}>")

        if signal is not None: 
            return SignalCallbackInfo(
                callback=callback,
                name=resolved_name,
                enabled=enabled,
                signal=signal,
                threshold=threshold,
            )

        return CallbackInfo(
            callback=callback,
            name=resolved_name,
            enabled=enabled
        )   


class InputSignalCallbackManager:
    def __init__(self) -> None:
        self.callbacks: Dict[InputSignal, List[SignalCallbackInfo]] = {}
        self.callback_info_factory = CallbackInfoFactory[ControllerState]()
        self._previous_state = ControllerState()

    def register_signal_callback(
            self,
            callback: Callable[[ControllerState], None],
            signal: InputSignal,
            name: Optional[str] = None,
            threshold: float = 0.1
    ) -> None:
        callback_info = cast(
            SignalCallbackInfo, 
            self.callback_info_factory.create(
                callback=callback,
                name=name,
                signal=signal,
                threshold=threshold
            )
        )

        if signal not in self.callbacks:
            self.callbacks[signal] = []

        self.callbacks[signal].append(callback_info)

    def unregister_signal_callback(
            self,
            signal: InputSignal,
            callback: Callable[[ControllerState], None]
    ) -> None:
        callback_id = id(callback)

        if signal in self.callbacks:
            self.callbacks[signal] = [
                cb for cb in self.callbacks[signal] 
                if id(cb.callback) != callback_id
            ]

            if not self.callbacks[signal]:
                del self.callbacks[signal]


class CallbackManager:
    def __init__(self) -> None:
        self.callbacks: Dict[int, CallbackInfo] = {}
        self.callback_info_factory = CallbackInfoFactory()
        self._execution_lock = threading.Lock()
        self._execution_pool = ThreadPoolExecutor(max_workers=4)

    def register_callback(self, callback_info: CallbackInfo) -> None:
        self.callbacks[id(callback_info.callback)] = callback_info

    def unregister_callback(self, callback: Callable[[ControllerState], None]):
        self.callbacks.pop(id(callback), None)

    def enable_callback(self, callback: Callable[[ControllerState], None], enabled: bool = True):
        cb_info = self.callbacks.get(id(callback))
        if cb_info:
            cb_info.enabled = enabled

    def clear_callbacks(self):
        self.callbacks.clear()

    def handle_callbacks(self, controller_state: ControllerState):
        if not controller_state.changed:
            return

        to_call = self.determine_callbacks_to_call(controller_state)

        with self._execution_lock:
            for callback_info in to_call:
                self._execution_pool.submit(
                    self._execute_callback_sync,
                    callback_info,
                    controller_state
                )

    def _execute_callback_sync(self, callback_info: CallbackInfo, controller_state: ControllerState) -> None:
        try:
            callback_info.callback(controller_state)
        except Exception as e:
            print(f"Callback {callback_info.name} failed: {e}")


    def determine_callbacks_to_call(self, controller_state: ControllerState) -> List[CallbackInfo]:
        to_call = []

        for callback in self.callbacks.values():
            if callback.enabled:
                to_call.append(callback)

        return to_call
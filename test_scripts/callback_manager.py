import threading

from test_scripts.input_handle import ControllerState, UnitreeRemoteControllerInputParser

from enum import Enum
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, TypeVar, cast
from dataclasses import dataclass

from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ # specific to using Go2
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ # specific to using Go2
from unitree_sdk2py.core.channel import ChannelSubscriber


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
    
@dataclass
class CallbackInfo[T]:
    callback: Callable[[T], None]
    name: Optional[str]

@dataclass
class SignalCallbackInfo[T](CallbackInfo[T]):
    signal: InputSignal
    threshold: float = 0.1

class CallbackInfoFactory[T]:
    def create(
        self,
        callback: Callable[[T], None],
        name: Optional[str] = None,
        signal: Optional[InputSignal] = None,
        threshold: float = 0.1,
    ) -> CallbackInfo[T]:
        resolved_name = name or getattr(callback, "__name__", f"<lambda:{id(callback)}>")

        if signal is not None: 
            return SignalCallbackInfo(
                callback=callback,
                name=resolved_name,
                signal=signal,
                threshold=threshold
            )

        return CallbackInfo(
            callback=callback,
            name=resolved_name
        )   

# if callbacks are complex (AI calculations, for example), callback execution can be offloaded to a worker thread as outlined here: https://docs.python.org/3/library/concurrent.futures.html
class InputSignalCallbackManager:
    def __init__(self) -> None:
        self.callbacks: Dict[InputSignal, List[SignalCallbackInfo]] = {}
        self.callback_info_factory = CallbackInfoFactory[ControllerState]()
        self.previous_state = ControllerState()
    
    def register_signal_callback(
            self,
            callback: Callable[[ControllerState], None],
            signal: InputSignal,
            name: Optional[str] = None,
            threshold: float = 0.1,
    ) -> SignalCallbackInfo:
        callback_info = cast(
            SignalCallbackInfo, 
            self.callback_info_factory.create(
                callback=callback,
                name=name,
                signal=signal,
                threshold=threshold,
            )
        )

        if signal not in self.callbacks:
            self.callbacks[signal] = []

        self.callbacks[signal].append(callback_info)

        return callback_info

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

    def determine_callbacks_to_call(self, controller_state: ControllerState) -> List[SignalCallbackInfo]:
        to_call = []

        for signal, callbacks in self.callbacks.items():
            for callback_info in callbacks:           
                if self.should_trigger_callback(signal, callback_info, controller_state):
                    to_call.append(callback_info)

        self.previous_state = ControllerState(**controller_state.__dict__) # see what this actually does
        return to_call
    
    def handle_callbacks(self, controller_state: ControllerState):
        if not controller_state.changed:
            return

        to_call = self.determine_callbacks_to_call(controller_state)

        for callback_info in to_call:
            self._execute_callback_sync(callback_info, controller_state)
                

    def _execute_callback_sync(self, callback_info: CallbackInfo, controller_state: ControllerState) -> None:
        try:
            callback_info.callback(controller_state)
        except Exception as e:
            print(f"Callback {callback_info.name} failed: {e}")

    def should_trigger_callback(
            self,
            signal: InputSignal,
            callback_info: SignalCallbackInfo,
            current_state: ControllerState
    ) -> bool:        
        if signal == InputSignal.LEFT_STICK:
            return self.stick_changed('l', current_state, callback_info.threshold)
            
        if signal == InputSignal.RIGHT_STICK:
            return self.stick_changed('r', current_state, callback_info.threshold)
        
        signal_attr = signal.value
        if not hasattr(current_state, signal_attr):
            return False
            
        current_value = getattr(current_state, signal_attr)
        previous_value = getattr(self.previous_state, signal_attr, 0.0)
        
        # For analog inputs
        if self.is_analog_input(signal):
            return self.check_analog_trigger(
                current_value, previous_value, callback_info
            )
        
        # For digital inputs (buttons)
        return self.check_digital_trigger(
            current_value, previous_value, callback_info
        )
    
    def stick_changed(self, stick: str, current_state: ControllerState, threshold: float) -> bool:
        x_attr = f"{stick}x"
        y_attr = f"{stick}y"
        
        current_x = getattr(current_state, x_attr, 0.0)
        current_y = getattr(current_state, y_attr, 0.0)
        previous_x = getattr(self.previous_state, x_attr, 0.0)
        previous_y = getattr(self.previous_state, y_attr, 0.0)
        
        # vector magnitude of x and y components
        distance = ((current_x - previous_x) ** 2 + (current_y - previous_y) ** 2) ** 0.5
        return distance > threshold
    
    def is_analog_input(self, signal: InputSignal) -> bool:
        analog_signals = {
            InputSignal.LEFT_STICK_X,
            InputSignal.LEFT_STICK_Y,
            InputSignal.RIGHT_STICK_X, 
            InputSignal.RIGHT_STICK_Y,
            InputSignal.LEFT_TRIGGER,
            InputSignal.RIGHT_TRIGGER
        }

        return signal in analog_signals
    
    def check_analog_trigger(self, current: float, previous: float, callback_info: SignalCallbackInfo) -> bool:
        return abs(current - previous) > callback_info.threshold
    
    def check_digital_trigger(self, current: float, previous: float, callback_info: SignalCallbackInfo) -> bool:
        return previous == 0.0 and current == 1.0
    
    def shutdown(self):
        self.callbacks.clear()
        

class InputHandler:
    def __init__(self) -> None:
        self.input_parser = UnitreeRemoteControllerInputParser()
        self.callback_manager = InputSignalCallbackManager()

        self.lowstate_subscriber = ChannelSubscriber("rt/lf/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.process_input, 10)

    def register_callback(
        self,
        signal: InputSignal,
        callback: Callable[[ControllerState], None],
        name: Optional[str] = None,
        threshold: float = 0.1,
        *args: Any,
        **kwargs: Any
    ) -> SignalCallbackInfo:
        return self.callback_manager.register_signal_callback(
            signal=signal,
            callback=callback,
            name=name,
            threshold=threshold,
        )
    
    def unregister_callback(self, signal: InputSignal, callback: Callable[[ControllerState], None]) -> None:
        self.callback_manager.unregister_signal_callback(signal, callback)

    def process_input(self, msg: LowState_) -> ControllerState:
        controller_state = self.input_parser.parse(msg.wireless_remote)
        self.callback_manager.handle_callbacks(controller_state)
        return controller_state
    
    def shutdown(self) -> None:
        self.lowstate_subscriber.Close() # cleanup resources and unsubscribe from Lowstate_ topic
        self.callback_manager.shutdown()



from enum import Enum


class InputSignal(Enum):
    """
    Enumeration of all possible controller input signals.

    Internal mapping from logical signal names to string identifiers
    used by the system for tracking and callback triggering.

    Members
    -------
    - LEFT_STICK, RIGHT_STICK : Stick identifiers
    - LEFT_STICK_X, LEFT_STICK_Y, RIGHT_STICK_X, RIGHT_STICK_Y : Stick axes
    - LEFT_TRIGGER, RIGHT_TRIGGER : Analog triggers
    - LEFT_BUMPER, RIGHT_BUMPER : Bumper buttons
    - BUTTON_A, BUTTON_B, BUTTON_X, BUTTON_Y : Face buttons
    - DPAD_UP, DPAD_DOWN, DPAD_LEFT, DPAD_RIGHT : Directional pad
    - SELECT, START, F1, F3 : Other buttons
    """
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
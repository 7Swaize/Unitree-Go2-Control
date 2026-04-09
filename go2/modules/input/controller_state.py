from dataclasses import dataclass


@dataclass
class ControllerState:
    """
    Represents the current state of a controller's inputs.

    Tracks analog and digital values for sticks, triggers, and buttons,
    along with a flag indicating whether any input has changed since the last update.
    """

    # Analog sticks (-1.0 to 1.0)
    lx: float = 0.0  #: Left stick X-axis
    ly: float = 0.0  #: Left stick Y-axis
    rx: float = 0.0  #: Right stick X-axis
    ry: float = 0.0  #: Right stick Y-axis

    # Triggers / bumpers (0.0 to 1.0)
    l1: float = 0.0  #: Left bumper
    l2: float = 0.0  #: Left trigger
    r1: float = 0.0  #: Right bumper
    r2: float = 0.0  #: Right trigger

    # Face buttons (digital)
    a: float = 0.0  #: 'A' button
    b: float = 0.0  #: 'B' button
    x: float = 0.0  #: 'X' button
    y: float = 0.0  #: 'Y' button

    # D-pad buttons
    up: float = 0.0  #: D-pad up
    down: float = 0.0  #: D-pad down
    left: float = 0.0  #: D-pad left
    right: float = 0.0  #: D-pad right

    # System buttons
    select: float = 0.0  #: Select button
    start: float = 0.0  #: Start button

    # Function buttons
    f1: float = 0.0  #: F1 button
    f3: float = 0.0  #: F3 button
    changed: bool = False
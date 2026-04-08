"""Controller State Representation"""

from dataclasses import dataclass


@dataclass
class ControllerState:
    """
    Represents the current state of a controller's inputs.

    Attributes
    ----------
    lx, ly : float
        Left stick X/Y axes (-1.0 to 1.0)
    rx, ry : float
        Right stick X/Y axes (-1.0 to 1.0)
    l1, l2, r1, r2 : float
        Trigger/bumper values (0.0 to 1.0)
    a, b, x, y : float
        Face button values (0.0 or 1.0)
    up, down, left, right : float
        D-pad values (0.0 or 1.0)
    select, start, f1, f3 : float
        System button values (0.0 or 1.0)
    changed : bool
        True if any value changed since the last update.
    """
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    
    l1: float = 0.0
    l2: float = 0.0
    r1: float = 0.0
    r2: float = 0.0
    
    a: float = 0.0
    b: float = 0.0
    x: float = 0.0
    y: float = 0.0
    
    up: float = 0.0
    down: float = 0.0
    left: float = 0.0
    right: float = 0.0
    
    select: float = 0.0
    start: float = 0.0
    f1: float = 0.0
    f3: float = 0.0

    changed: bool = False

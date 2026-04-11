from enum import Enum, auto

# We can just expose this from the "core" module

class HardwareType(Enum):
    """Enumeration passed upon creation of a :class:`~core.controller.Go2Controller` instance."""
    NATIVE = auto() #: Commands should act upon an actual Unitree-Go2 robot.
    VIRTUAL = auto() #: Commands should act upon the simulator (launched on controller creation in 'VIRTUAL' mode).
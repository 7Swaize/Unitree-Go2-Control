import os
import signal
import subprocess
from typing_extensions import override

from hardware.interfaces.hardware_interface import HardwareInterface


class VirtualHardware(HardwareInterface):
    """
    Virtual hardware backend for development and testing.
     
    This implementation mimics robot functionality using the Unitree provided Mujoco simulator.
    The simulator is launched via seperate process during initialization of the hardware.

    Dependencies
    ------------
    - `Unitree MuJoCo library <https://github.com/7Swaize/unitree_mujoco.git>`_
    """
    
    def __init__(self):
        self.sim_proc = subprocess.Popen(
            ["/home/gsmst/unitree_mujoco/simulate/build/unitree_mujoco", "-r", "go2", "-s", "scene_terrain.xml"], # TODO: Refactor into using git submodule and scikit build
        )

        self._initialized = True
    
    @override
    def _initialize(self) -> None:
        self._initialized = True

    @override
    def _move(self, vx: float, vy: float) -> None:
        pass

    @override
    def _rotate(self, vrot: float):
        pass
    
    @override
    def _stand_up(self) -> None:
        pass
    
    @override
    def _stand_down(self) -> None:
        pass
    
    @override
    def _stop_move(self) -> None:
        pass

    @override
    def _shutdown(self) -> None:
        self._initialized = False

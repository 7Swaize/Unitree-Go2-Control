import os
import signal
import subprocess

from .hardware_interface import HardwareInterface


class VirtualHardware(HardwareInterface):
    """
    Virtual hardware backend for development and testing.
     
    This implementation mimics robot functionality using the Unitree provided Mujoco simulator.
    The simulator is launched during initialization of the hardware.

    Dependencies
    ----------
    - Unitree MuJoCo library: https://github.com/7Swaize/unitree_mujoco.git
    """
    
    def __init__(self):
        self.sim_proc = subprocess.Popen(
            ["/home/gsmst/unitree_mujoco/simulate/build/unitree_mujoco", "-r", "go2", "-s", "scene_terrain.xml"], # TODO: Refactor into using git submodule and scikit build
        )

        self._initialized = True
    
    def initialize(self) -> None:
        self._initialized = True

    def move(self, vx: float, vy: float) -> None:
        pass

    def rotate(self, vrot: float):
        pass
    
    def stand_up(self) -> None:
        pass
    
    def stand_down(self) -> None:
        pass
    
    def stop_move(self) -> None:
        pass

    def shutdown(self) -> None:
        self._initialized = False
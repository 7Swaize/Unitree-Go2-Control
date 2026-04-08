import sys
from typing_extensions import override

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

from hardware.interfaces.hardware_interface import HardwareInterface


class NativeHardware(HardwareInterface):
    """
    Hardware interface to execute on the Unitree GO2 itself.

    This implementation communicates with the real robot using ``unitree_sdk2py``. 
    It can only used for native execution on the robot.

    Warnings
    --------
    - This class issues real hardware commands.
    - Improper use may cause unexpected robot motion.
    """    
    def __init__(self):
        self._sport_client = None
        self._initialized = False
    
    @override
    def _initialize(self) -> None:
        """
        Initialize the Unitree SDK connection.

        This method sets up DDS communication and initializes the sport client.
        """
        if self._initialized:
            return
        
        # TODO: Pull this into yaml or ctor?
        if len(sys.argv) < 2:
            ChannelFactoryInitialize(1, "lo")
        else:
            ChannelFactoryInitialize(0, sys.argv[1])
        
        self._sport_client = SportClient()
        self._sport_client.Init()
        self._sport_client.SetTimeout(3.0)
        
        self._initialized = True
    
    @override
    def _shutdown(self) -> None:
        if self._sport_client:
            self._sport_client.StopMove()
    
    @override
    def _move(self, vx: float, vy: float) -> None:
        if self._sport_client:
            self._sport_client.Move(vx, vy, 0)

    @override
    def _rotate(self, vrot: float):
        if self._sport_client:
            self._sport_client.Move(0, 0, vrot)
    
    @override
    def _stand_up(self) -> None:
        if self._sport_client:
            self._sport_client.StandUp()
    
    @override
    def _stand_down(self) -> None:
        if self._sport_client:
            self._sport_client.StandDown()
    
    @override
    def _stop_move(self) -> None:
        if self._sport_client:
            self._sport_client.StopMove()

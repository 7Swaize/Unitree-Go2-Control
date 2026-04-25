import sys
from typing import Tuple
from typing_extensions import override

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

from ..hardware_interface import HardwareInterface


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

    @override
    def _damp(self) -> None:
        if self._sport_client:
            self._sport_client.Damp()

    @override
    def _balance_stand(self) -> None:
        if self._sport_client:
            self._sport_client.BalanceStand()

    @override
    def _recovery_stand(self) -> None:
        if self._sport_client:
            self._sport_client.RecoveryStand()

    @override
    def _euler(self, roll: float, pitch: float, yaw: float) -> None:
        if self._sport_client:
            self._sport_client.Euler(roll, pitch, yaw)

    @override
    def _sit(self) -> None:
        if self._sport_client:
            self._sport_client.Sit()

    @override
    def _rise_sit(self) -> None:
        if self._sport_client:
            self._sport_client.RiseSit()

    @override
    def _speed_level(self, level: int) -> None:
        if self._sport_client:
            self._sport_client.SpeedLevel(level)

    @override
    def _hello(self) -> None:
        if self._sport_client:
            self._sport_client.Hello()

    @override
    def _stretch(self) -> None:
        if self._sport_client:
            self._sport_client.Stretch()

    @override
    def _content(self) -> None:
        if self._sport_client:
            self._sport_client.Content()

    @override
    def _dance1(self) -> None:
        if self._sport_client:
            self._sport_client.Dance1()

    @override
    def _dance2(self) -> None:
        if self._sport_client:
            self._sport_client.Dance2()

    @override
    def _switch_joystick(self, on: bool) -> None:
        if self._sport_client:
            self._sport_client.SwitchJoystick(on)

    @override
    def _pose(self, flag: bool) -> None:
        if self._sport_client:
            self._sport_client.Pose(flag)

    @override
    def _scrape(self) -> None:
        if self._sport_client:
            self._sport_client.Scrape()

    @override
    def _front_flip(self) -> None:
        if self._sport_client:
            self._sport_client.FrontFlip()

    @override
    def _front_jump(self) -> None:
        if self._sport_client:
            self._sport_client.FrontJump()

    @override
    def _front_pounce(self) -> None:
        if self._sport_client:
            self._sport_client.FrontPounce()

    @override
    def _heart(self) -> None:
        if self._sport_client:
            self._sport_client.Heart()

    @override
    def _left_flip(self) -> None:
        if self._sport_client:
            self._sport_client.LeftFlip()

    @override
    def _back_flip(self) -> None:
        if self._sport_client:
            self._sport_client.BackFlip()

    @override
    def _free_walk(self) -> None:
        if self._sport_client:
            self._sport_client.FreeWalk()

    @override
    def _free_bound(self, flag: bool) -> None:
        if self._sport_client:
            self._sport_client.FreeBound(flag)

    @override
    def _free_jump(self, flag: bool) -> None:
        if self._sport_client:
            self._sport_client.FreeJump(flag)

    @override
    def _free_avoid(self, flag: bool) -> None:
        if self._sport_client:
            self._sport_client.FreeAvoid(flag)

    @override
    def _walk_upright(self, flag: bool) -> None:
        if self._sport_client:
            self._sport_client.WalkUpright(flag)

    @override
    def _cross_step(self, flag: bool) -> None:
        if self._sport_client:
            self._sport_client.CrossStep(flag)

    @override
    def _static_walk(self) -> None:
        if self._sport_client:
            self._sport_client.StaticWalk()

    @override
    def _trot_run(self) -> None:
        if self._sport_client:
            self._sport_client.TrotRun()

    @override
    def _hand_stand(self, flag: bool) -> None:
        if self._sport_client:
            self._sport_client.HandStand(flag)

    @override
    def _classic_walk(self, flag: bool) -> None:
        if self._sport_client:
            self._sport_client.ClassicWalk(flag)

    @override
    def _auto_recovery_set(self, enabled: bool) -> None:
        if self._sport_client:
            self._sport_client.AutoRecoverySet(enabled)

    @override
    def _auto_recovery_get(self) -> Tuple[int, bool | None]:
        if self._sport_client:
            return self._sport_client.AutoRecoveryGet()
        return 1, None

    @override
    def _switch_avoid_mode(self) -> None:
        if self._sport_client:
            self._sport_client.SwitchAvoidMode()
        
    

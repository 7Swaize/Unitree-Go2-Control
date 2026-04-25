import iceoryx2 as iox2
from typing import Tuple
from typing_extensions import override
from iceoryx_interfaces.mappings import SportCommand
from iceoryx_interfaces.qos import SportQoS
from iceoryx_interfaces.sport_cmds import (
    SportCommandHeader_,
    NoArgsData_,
    FloatArgsData_
)

from ..hardware_interface import HardwareInterface


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
        iox2.set_log_level_from_env_or(iox2.LogLevel.Error)
    
    @override
    def _initialize(self) -> None:
        self._node = iox2.NodeBuilder.new() \
                        .signal_handling_mode(iox2.SignalHandlingMode.Disabled) \
                        .create(iox2.ServiceType.Ipc)
        
        self._noargs_service = self._node.service_builder(iox2.ServiceName.new(SportQoS.TOPIC_SIM_NOARGS_CMD)) \
                                    .publish_subscribe(NoArgsData_) \
                                    .user_header(SportCommandHeader_) \
                                    .open_or_create()
        
        self._floatargs_service = self._node.service_builder(iox2.ServiceName.new(SportQoS.TOPIC_SIM_FLOATARGS_CMD)) \
                                    .publish_subscribe(FloatArgsData_) \
                                    .user_header(SportCommandHeader_) \
                                    .open_or_create()
        
        self._noargs_pub = self._noargs_service.publisher_builder().create()
        self._floatargs_pub = self._floatargs_service.publisher_builder().create()
        self._initialized = True

    @override
    def _move(self, vx: float, vy: float) -> None:
        sample = self._floatargs_pub.loan_uninit()
        
        sample.user_header().contents.command = SportCommand.MOVE
        sample = sample.write_payload(
            FloatArgsData_(arg1=vx, arg2=vy)
        )

        sample.send()

    @override
    def _rotate(self, vrot: float):
        sample = self._floatargs_pub.loan_uninit()

        sample.user_header().contents.command = SportCommand.ROTATE
        sample = sample.write_payload(
            FloatArgsData_(arg1=vrot, arg2=0)
        )

        sample.send()
    
    @override
    def _stand_up(self) -> None:
        sample = self._noargs_pub.loan_uninit()

        sample.user_header().contents.command = SportCommand.STAND_UP
        sample = sample.write_payload(
            NoArgsData_(null=0)
        )

        sample.send()
    
    @override
    def _stand_down(self) -> None:
        sample = self._noargs_pub.loan_uninit()

        sample.user_header().contents.command = SportCommand.STAND_DOWN
        sample = sample.write_payload(
            NoArgsData_(null=0)
        )

        sample.send()
   
    
    @override
    def _stop_move(self) -> None:
        sample = self._noargs_pub.loan_uninit()

        sample.user_header().contents.command = SportCommand.STOP
        sample = sample.write_payload(
            NoArgsData_(null=0)
        )

        sample.send()
   

    @override
    def _damp(self) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _balance_stand(self) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _recovery_stand(self) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _euler(self, roll: float, pitch: float, yaw: float) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _sit(self) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _rise_sit(self) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _speed_level(self, level: int) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _hello(self) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _stretch(self) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _content(self) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _dance1(self) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _dance2(self) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _switch_joystick(self, on: bool) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _pose(self, flag: bool) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _scrape(self) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _front_flip(self) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _front_jump(self) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _front_pounce(self) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _heart(self) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _left_flip(self) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _back_flip(self) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _free_walk(self) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _free_bound(self, flag: bool) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _free_jump(self, flag: bool) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _free_avoid(self, flag: bool) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _walk_upright(self, flag: bool) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _cross_step(self, flag: bool) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _static_walk(self) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _trot_run(self) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _hand_stand(self, flag: bool) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _classic_walk(self, flag: bool) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _auto_recovery_set(self, enabled: bool) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _auto_recovery_get(self) -> Tuple[int, bool | None]:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _switch_avoid_mode(self) -> None:
        raise NotImplementedError("Virtual hardware does not support this command yet.")

    @override
    def _shutdown(self) -> None:
        self._initialized = False

import iceoryx2 as iox2
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
    def _shutdown(self) -> None:
        self._initialized = False

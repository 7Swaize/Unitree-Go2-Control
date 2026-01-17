import sys
import time

from abc import ABC, abstractmethod


class HardwareInterface(ABC):
    """
    Abstract interface for dog hardware control.

    This interface defines the minimum set of motion and posture commands
    required by the system. Concrete implementations may communicate with
    real hardware (SDK) or simulated backends.

    Notes
    -----
    - This is an internal abstraction.
    - Students should not subclass this directly.
    """
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the hardware interface. This is handled automatically and shouldn't be called by students.

        Establishes communication with the hardware or simulation backend.
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """
        Cleanly shut down the hardware interface.
        """
        pass
    
    @abstractmethod
    def move(self, vx: float, vy: float) -> None:
        """
        Move the dog in the horizontal plane.

        Parameters
        ----------
        vx : float
            Forward/backward velocity.
        vy : float
            Left/right velocity.
        """
        pass

    @abstractmethod
    def rotate(self, vrot: float):
        """
        Rotate the dog in place.

        Parameters
        ----------
        vrot : float
            Rotational velocity.
        """
        pass

    
    @abstractmethod
    def stand_up(self) -> None:
        """Command the dog to stand up."""
        pass
    
    @abstractmethod
    def stand_down(self) -> None:
        """Command the dog to lie down."""
        pass
    
    @abstractmethod
    def stop_move(self) -> None:
        """Immediately stop all movement and clear internal movement command buffer."""
        pass


class UnitreeSDKHardware(HardwareInterface):
    """
    Hardware interface backed by the Unitree SDK.

    This implementation communicates with the real robot using
    ``unitree_sdk2py``. It is only available when the SDK is installed
    and properly configured.

    Warnings
    --------
    - This class issues real hardware commands.
    - Improper use may cause unexpected robot motion.
    """    
    def __init__(self):
        self._sport_client = None
        self._initialized = False
    
    def initialize(self) -> None:
        """
        Initialize the Unitree SDK connection.

        This method sets up DDS communication, initializes the sport client,
        and brings the robot to a safe standing state.
        """
        if self._initialized:
            return
            
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        
        if len(sys.argv) > 1:
            ChannelFactoryInitialize(0, sys.argv[1])
        else:
            ChannelFactoryInitialize(0)
        
        print("[SDK] Creating SportClient")
        self._sport_client = SportClient()
        self._sport_client.Init()
        self._sport_client.SetTimeout(3.0)
        
        print("[SDK] Standing up")
        self._sport_client.StandUp()
        time.sleep(1)
        self._sport_client.StopMove()
        time.sleep(0.5)
        
        self._initialized = True
    
    def shutdown(self) -> None:
        if self._sport_client:
            self._sport_client.StopMove()
    
    def move(self, vx: float, vy: float) -> None:
        if self._sport_client:
            self._sport_client.Move(vx, vy, 0)

    def rotate(self, vrot: float):
        if self._sport_client:
            self._sport_client.Move(0, 0, vrot)
    
    def stand_up(self) -> None:
        if self._sport_client:
            self._sport_client.StandUp()
    
    def stand_down(self) -> None:
        if self._sport_client:
            self._sport_client.StandDown()
    
    def stop_move(self) -> None:
        if self._sport_client:
            self._sport_client.StopMove()


class SimulatedHardware(HardwareInterface):
    """
    Simulated hardware backend for development and testing.

    This implementation mimics robot motion using simple kinematic updates
    and console output. It allows development without access to physical
    hardware or the Unitree SDK.

    Notes
    -----
    - Used automatically when SDK support is disabled.
    - Motion is not physically accurate.
    """
    
    def __init__(self):
        self._initialized = False
        self._position = [0.0, 0.0]
        self._rotation = 0
        self._is_standing = False
    
    def initialize(self) -> None:
        print("[SIM] Initializing simulated hardware")
        self._initialized = True
        self._is_standing = True
    
    def shutdown(self) -> None:
        print("[SIM] Shutting down simulation")
        self._initialized = False
    
    def move(self, vx: float, vy: float) -> None:
        self._position[0] += vx * 0.1
        self._position[1] += vy * 0.1
        print(f"[SIM] Move: vx={vx:.2f}, vy={vy:.2f} -> pos={self._position}")

    def rotate(self, vrot: float):
        self._rotation += vrot * 0.1
        print(f"[SIM] Rotate: vrot={vrot:.2f} -> rotation={self._rotation}")
    
    def stand_up(self) -> None:
        print("[SIM] Standing up")
        self._is_standing = True
    
    def stand_down(self) -> None:
        print("[SIM] Standing down")
        self._is_standing = False
    
    def stop_move(self) -> None:
        print("[SIM] Stopping movement")
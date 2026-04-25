from typing import Tuple
from typing_extensions import override

from ...core.module import DogModule
from ...hardware.hardware_interface import HardwareInterface


class MovementModule(DogModule):
    """
    High-level movement control for the robot.

    Parameters
    ----------
    hardware : HardwareInterface
        The underlying hardware interface that communicates with the robot.
    """

    def __init__(self, hardware: HardwareInterface) -> None:
        """
        Create the MovementModule. 

        Sets up maximum safe limits for rotation and translation.
        """
        super().__init__("Movement")
        self.hardware = hardware
        self.max_rotation = 5.0
        self.max_translation = 5.0

    @override
    def _initialize(self) -> None:
        """
        Prepare the movement module for use. This is called internally,
        and should not be called directly by users.

        This marks the module as initialized. It must be called
        before issuing movement commands.
        """
        self._initialized = True

    def rotate(self, amount: float) -> None:
        """
        Rotate the robot around its yaw axis.

        Parameters
        ----------
        amount : float
            Desired rotation amount. Automatically clamped to
            [-max_rotation, max_rotation].

        Notes
        -----
        - The rotation is relative to the robot’s current orientation.
        """
        amount = max(-self.max_rotation, min(amount, self.max_rotation))
        self.hardware._rotate(amount)

    def move(self, amount_x: float = 0.0, amount_y: float = 0.0) -> None:
        """
        Move the robot forward/backward and laterally.

        Parameters
        ----------
        amount_x : float, optional
            Forward/backward movement amount. Positive is forward.
            Automatically clamped to [-max_translation, max_translation].
        amount_y : float, optional
            Lateral movement amount. Positive is right.
            Automatically clamped to [-max_translation, max_translation].

        Notes
        -----
        - Movements are relative to the robot’s current orientation.
        """
        amount_x = max(-self.max_translation, min(amount_x, self.max_translation))
        amount_y = max(-self.max_translation, min(amount_y, self.max_translation))
        self.hardware._move(amount_x, amount_y)

    def stand_up(self) -> None:
        """
        Command the robot to stand up.

        Notes
        -----
        - The robot must be powered on and initialized.
        - It's best practice to wait a few seconds after standing up before issuing movement commands.
        """
        self.hardware._stand_up()

    def stand_down(self) -> None:
        """
        Command the robot to lay down.

        Notes
        -----
        - The robot must be powered on and initialized.
        """
        self.hardware._stand_down()

    def stop(self) -> None:
        """
        Stop all robot movement immediately.

        Notes
        -----
        - Can be called at any time to halt translation and rotation. Can be used to clear internal movement command buffer.
        """
        self.hardware._stop_move()

    def damp(self) -> None:
        """Command the robot to damp."""
        self.hardware._damp()

    def balance_stand(self) -> None:
        """Command the robot to balance stand."""
        self.hardware._balance_stand()

    def recovery_stand(self) -> None:
        """Command the robot to recovery stand."""
        self.hardware._recovery_stand()

    def euler(self, roll: float, pitch: float, yaw: float) -> None:
        """Set the robot's Euler angles."""
        self.hardware._euler(roll, pitch, yaw)

    def sit(self) -> None:
        """Command the robot to sit."""
        self.hardware._sit()

    def rise_sit(self) -> None:
        """Command the robot to rise from sit."""
        self.hardware._rise_sit()

    def speed_level(self, level: int) -> None:
        """Set the robot's speed level."""
        self.hardware._speed_level(level)

    def hello(self) -> None:
        """Command the robot to perform hello."""
        self.hardware._hello()

    def stretch(self) -> None:
        """Command the robot to stretch."""
        self.hardware._stretch()

    def content(self) -> None:
        """Command the robot to perform content."""
        self.hardware._content()

    def dance1(self) -> None:
        """Command the robot to perform dance1."""
        self.hardware._dance1()

    def dance2(self) -> None:
        """Command the robot to perform dance2."""
        self.hardware._dance2()

    def switch_joystick(self, on: bool) -> None:
        """Switch joystick mode."""
        self.hardware._switch_joystick(on)

    def pose(self, flag: bool) -> None:
        """Set pose mode."""
        self.hardware._pose(flag)

    def scrape(self) -> None:
        """Command the robot to scrape."""
        self.hardware._scrape()

    def front_flip(self) -> None:
        """Command the robot to perform front flip."""
        self.hardware._front_flip()

    def front_jump(self) -> None:
        """Command the robot to perform front jump."""
        self.hardware._front_jump()

    def front_pounce(self) -> None:
        """Command the robot to perform front pounce."""
        self.hardware._front_pounce()

    def heart(self) -> None:
        """Command the robot to perform heart."""
        self.hardware._heart()

    def left_flip(self) -> None:
        """Command the robot to perform left flip."""
        self.hardware._left_flip()

    def back_flip(self) -> None:
        """Command the robot to perform back flip."""
        self.hardware._back_flip()

    def free_walk(self) -> None:
        """Command the robot to free walk."""
        self.hardware._free_walk()

    def free_bound(self, flag: bool) -> None:
        """Set free bound mode."""
        self.hardware._free_bound(flag)

    def free_jump(self, flag: bool) -> None:
        """Set free jump mode."""
        self.hardware._free_jump(flag)

    def free_avoid(self, flag: bool) -> None:
        """Set free avoid mode."""
        self.hardware._free_avoid(flag)

    def walk_upright(self, flag: bool) -> None:
        """Set walk upright mode."""
        self.hardware._walk_upright(flag)

    def cross_step(self, flag: bool) -> None:
        """Set cross step mode."""
        self.hardware._cross_step(flag)

    def static_walk(self) -> None:
        """Command the robot to static walk."""
        self.hardware._static_walk()

    def trot_run(self) -> None:
        """Command the robot to trot run."""
        self.hardware._trot_run()

    def hand_stand(self, flag: bool) -> None:
        """Set hand stand mode."""
        self.hardware._hand_stand(flag)

    def classic_walk(self, flag: bool) -> None:
        """Set classic walk mode."""
        self.hardware._classic_walk(flag)

    def auto_recovery_set(self, enabled: bool) -> None:
        """Set auto recovery mode."""
        self.hardware._auto_recovery_set(enabled)

    def get_auto_recovery(self) -> Tuple[int, bool | None]:
        """Get auto recovery status."""
        return self.hardware._auto_recovery_get()

    def switch_avoid_mode(self) -> None:
        """Switch avoid mode."""
        self.hardware._switch_avoid_mode()

    @override
    def _shutdown(self) -> None:
        """
        Shut down the movement module safely. This should not be called by users

        Stops all movement and marks the module as uninitialized.
        """
        self.stop()
        self._initialized = False
"""
Movement Module for User Use
============================

This module provides high-level movement controls for the dog robot.

Users should interact only with the ``MovementModule`` class to
command the robot. It wraps the underlying hardware interface and
performs safety checks to prevent unsafe motions.

Users should not access or construct this class directly.
Rather, they should access it through the :class:`~core.controller.Go2Controller` instance.
"""

from typing_extensions import override

from core.module import DogModule
from hardware.interfaces.hardware_interface import HardwareInterface


class MovementModule(DogModule):
    """
    High-level movement control for the robot.

    Parameters
    ----------
    hardware : HardwareInterface
        The underlying hardware interface that communicates with the robot.
    """

    def __init__(self, hardware: HardwareInterface):
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

    @override
    def _shutdown(self) -> None:
        """
        Shut down the movement module safely. This should not be called by users

        Stops all movement and marks the module as uninitialized.
        """
        self.stop()
        self._initialized = False
import time

from go2.core import Go2Controller, HardwareType


def create_controller() -> Go2Controller:
    # Create a Go2 Controller instance to access all provided funtionalities.
    # Here, we choose HardwareType.NATIVE because we want movements commands to be executed on the actual robot.
    controller = Go2Controller(hardware_type=HardwareType.NATIVE)

    # NOTE: The movement module is a default module. It is already added onto the controller.
    # Therefore, we do not need to explictly add it.

    return controller


def run_movement_sequence(controller: Go2Controller) -> None:
    # Stand Up
    controller.movement.stand_up()
    time.sleep(3)

    # Move forward
    # Velocity is held for 1s. We loop to maintain motion for approx 3 seconds
    for _ in range(3):
        # Paramter in units of m/s
        controller.movement.move(1, 0)
        time.sleep(1.0)

    # Clear internal velocity command buffer
    controller.movement.stop()

    # Rotate
    # Rotation is held for 1s. We loop to maintain motion for approx 3 seconds
    for _ in range(3):
        # Parameter in units of rad/s
        controller.movement.rotate(0.5)
        time.sleep(1.0)

    # Clear internal velocity command buffer
    controller.movement.stop()

    # Stand down
    controller.movement.stand_down()


if __name__ == "__main__":
    # Create a controller instance
    controller = create_controller()

    # Run our test movement sequence
    run_movement_sequence(controller)

    # Shutdown the controller
    # It is critical to always shutdown the control on script exit
    controller.safe_shutdown()